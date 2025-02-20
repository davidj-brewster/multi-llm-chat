import json
import os
import datetime
import sys
import time
import random
import logging
import re
import yaml
from ollama import AsyncClient, ChatResponse, chat
from typing import List, Dict, Optional, TypeVar
from dataclasses import dataclass
import io
import requests
import asyncio
# Third-party imports
from openai import OpenAI
from google import genai
from google.genai import types
from anthropic import Anthropic
# Local imports
from context_analysis import ContextAnalyzer
from adaptive_instructions import AdaptiveInstructionManager
from configuration import load_config, DiscussionConfig, detect_model_capabilities
from configuration import load_config, DiscussionConfig, detect_model_capabilities
from configdataclasses import TimeoutConfig, FileConfig, ModelConfig, DiscussionConfig
from arbiter_v2 import evaluate_conversations
from file_handler import ConversationMediaHandler, FileConfig as MediaConfig
from model_clients import BaseClient, OpenAIClient, ClaudeClient, GeminiClient, MLXClient, OllamaClient, PicoClient

T = TypeVar('T')
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
CONFIG_PATH = "config.yaml"

# File type configurations
SUPPORTED_FILE_TYPES = {
    "image": {
        "extensions": [".jpg", ".jpeg", ".png", ".gif", ".webp"],
        "max_size": 20 * 1024 * 1024,  # 20MB
        "max_resolution": (8192, 8192)
    },
    "video": {
        "extensions": [".mp4", ".mov", ".avi", ".webm"],
        "max_size": 300 * 1024 * 1024,  # 300MB
        "max_resolution": (3840, 2160)  # 4K
    },
    "text": {
        "extensions": [".txt", ".md", ".py", ".js", ".html", ".csv", ".json", ".yaml", ".yml"],
        "max_size": 20 * 1024 * 1024  # 20MB
    }
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_battle.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for AI model parameters"""
    temperature: float = 0.8
    max_tokens: int = 2048
    stop_sequences: List[str] = None
    seed: Optional[int] = random.randint(0, 1000)

@dataclass
class ConversationManager:
    def __init__(self,
                 config: Optional[DiscussionConfig] = None,
                 domain: str = "General knowledge",
                 human_delay: float = 20.0,
                 mode: str = None,
                 min_delay: float = 10,
                 gemini_api_key: Optional[str] = None,
                 claude_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None) -> None:
        self.config = config
        self.domain = config.goal if config else domain
        self.human_delay = human_delay
        self.mode = mode  # "human-aiai" or "ai-ai"
        self.media_handler = ConversationMediaHandler()
        self.min_delay = min_delay
        self.conversation_history: List[Dict[str, str]] = []
        self.is_paused = False
        self.initial_prompt = domain
        self.rate_limit_lock = asyncio.Lock()
        self.last_request_time = 0
        #self.mlx_base_url = mlx_base_url
        claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        anthropic_api_key = claude_api_key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        # Initialize all clients with their specific models
        self.claude_client = ClaudeClient(role=None, api_key=claude_api_key, mode=None, domain=domain, model="claude-3-5-sonnet-20241022") if claude_api_key else None
        self.haiku_client = ClaudeClient(role=None, api_key=claude_api_key, mode=None, domain=domain, model="claude-3.5-haiku-20241022") if claude_api_key else None
        
        self.openai_o1_client = OpenAIClient(api_key=openai_api_key, mode=mode, domain=domain, model='o1') if openai_api_key else None
        self.openai_4o_client = OpenAIClient(api_key=openai_api_key, mode=mode, domain=domain, model="chatgpt-4o-latest") if openai_api_key else None
        self.openai_client = OpenAIClient(api_key=openai_api_key, mode=mode, domain=domain, model="chatgpt-4o-latest") if openai_api_key else None
        self.openai_4o_mini_client = OpenAIClient(api_key=openai_api_key, mode=mode, domain=domain, model='gpt-4o-mini-2024-07-18') if openai_api_key else None
        self.openai_o1_mini_client =  OpenAIClient(api_key=openai_api_key, mode=mode, domain=domain, model='o1-mini-2024-09-12') if openai_api_key else None
        
        self.mlx_qwq_client = MLXClient(mode=self.mode, domain=domain, base_url=None, model="mlx-community/Meta-Llama-3.1-8B-Instruct-abliterated-8bit")
        self.mlx_abliterated_client =  MLXClient(mode=self.mode, domain=domain, base_url=None, model="mlx-community/Meta-Llama-3.1-8B-Instruct-abliterated-8bit")
        
        self.gemini_2_reasoning_client =  GeminiClient(api_key=gemini_api_key, role=None, mode=mode, domain=domain, model="gemini-2.0-flash-thinking-exp-01-21") if gemini_api_key else None
        self.gemini_client =  GeminiClient(api_key=gemini_api_key, role=None, mode=mode, domain=domain, model='gemini-2.0-flash-exp') if gemini_api_key else None
        self.gemini_1206_client =  GeminiClient(api_key=gemini_api_key, role=None, mode=mode, domain=domain, model='gemini-exp-1206') if gemini_api_key else None
        
        self.pico_ollama_r1qwen_14 = PicoClient(mode=self.mode, domain=domain, model='DeepSeek-R1-Distill-Qwen-14B-abliterated-v2-Q4-mlx')
        self.pico_ollama_r1llama_8 = PicoClient(mode=self.mode, domain=domain, model='DeepSeek-R1-Distill-Llama-8B-8bit-mlx')
        self.pico_medical_lm = PicoClient(mode=self.mode, domain=domain, model='Bio-Medical-Llama-3-2-1B-CoT-012025')
        self.ollama_phi4_client =  OpenAIClient(mode=self.mode, domain=domain, model='phi-4:latest') #MLX via Pico
        self.ollama_client =  OllamaClient(mode=self.mode, domain=domain, model='mannix/llama3.1-8b-lexi:latest')
        self.ollama_lexi_client =  OllamaClient(mode=self.mode, domain=domain, model='mannix/llama3.1-8b-lexi:latest')
        self.ollama_instruct_client =  OllamaClient(mode=self.mode, domain=domain, model='llama3.2:3b-instruct-q8_0')
        self.ollama_abliterated_client =  OllamaClient(mode=self.mode, domain=domain, model="mannix/llama3.1-8b-abliterated:latest")

        # Initialize model map
        self.model_map = {
            "claude": self.claude_client,  # sonnet
            "gemini_2_reasoning": self.gemini_2_reasoning_client,
            "gemini": self.gemini_client,
            "gemini-exp-1206": self.gemini_1206_client,
            "openai": self.openai_client,  # 4o
            "o1": self.openai_o1_client,
            "mlx-qwq": self.mlx_qwq_client,
            "mlx-llama31_abliterated": self.mlx_abliterated_client,
            "mlx-abliterated": self.mlx_abliterated_client,
            "haiku": self.haiku_client,  # haiku
            "o1-mini": self.openai_o1_mini_client,
            "gpt-4o-mini": self.openai_4o_mini_client,
            "chatgpt-4o": self.openai_4o_client,
            "ollama": self.ollama_client,
            "ollama-lexi": self.ollama_lexi_client,
            "ollama-instruct": self.ollama_instruct_client,
            "ollama-abliterated": self.ollama_abliterated_client,
            "ollama-phi4": self.ollama_phi4_client,
            "pico-r1-14": self.pico_ollama_r1qwen_14,
            "pico-r1-8": self.pico_ollama_r1llama_8,
            "pico-med": self.pico_medical_lm,
        }

    @classmethod
    def from_config(cls, config_path: str) -> 'ConversationManager':
        """Initialize ConversationManager from a configuration file"""
        config = load_config(config_path)
        return cls(
            config=config,
            domain=config.goal,
            mode="ai-ai",  # Default to ai-ai mode for config-based initialization
            human_delay=20.0,
            min_delay=10,
            # API keys still loaded from environment
        )

    def _validate_model_capabilities(self) -> None:
        """Validate model capabilities against configuration requirements"""
        if not self.config:
            return

        for name, model_config in self.config.models.items():
            capabilities = detect_model_capabilities(model_config)
            
            # Check vision capability if needed
            if self.config.input_file and self.config.input_file.type in ["image", "video"]:
                if not capabilities["vision"] and model_config.role == "assistant":
                    raise ValueError(f"Model {name} ({model_config.type}) does not support vision tasks")

    def _get_model_info(self, model_name: str) -> Dict[str, str]:
        """Get model provider and name"""
        providers = {
            "claude": "claude",
            "sonnet": "claude",
            "gemini": "google",
            "flash": "google",
            "thinking": "google",
            "_exp": "google",
            "gpt": "openai",
            "openai": "openai",
            "chatgpt": "openai",
            "o1": "openai",
            "ollama": "local",
            "mlx": "local",
            "pico": "local",
        }
        # Extract provider from model name
        provider = next((p for p in providers if p in model_name.lower()), "unknown")
        return {
            "provider": providers[provider],
            "name": model_name
        }


    def validate_connections(self, required_models: List[str] = None) -> bool:
        """Validate required model connections
        
        Args:
            required_models: List of model names that need to be validated.
                           If None, validates all initialized clients.
        
        Returns:
            bool: True if all required connections are valid
        """
        if required_models is None:
            # Only validate initialized cloud clients
            required_models = [name for name, client in self.model_map.items()
                             if client and name not in ["ollama", "mlx"]]
            
        if not required_models:
            logger.info("No models require validation")
            return True
            
        validations = []
        #for model_name in required_models:
        #    client =  self._get_client(model_name)
        #    if client != "openai":
        #        try:
        #            valid =  client.validate_connection()
        #            validations.append(valid)
        #            if not valid:
        #                logger.error(f"{model_name} client validation failed")
        #        except Exception as e:
        #            logger.error(f"Error validating {model_name} client: {e}")
        #            validations.append(False)
        #    else:
        #        logger.error(f"Client not available for {model_name}")
        #        validations.append(False)
        #        
        return True

    def rate_limited_request(self):
         with self.rate_limit_lock:
            current_time = time.time()
            if current_time - self.last_request_time < self.min_delay:
                 io.sleep(self.min_delay)
            self.last_request_time = time.time()

    def run_conversation_turn(self,
                                  prompt: str,
                                  model_type: str,
                                  client: BaseClient,
                                  mode: str,
                                  role: str,
                                  system_instruction: str=None) -> str:
        """Single conversation turn with specified model and role."""
        # Map roles consistently
        self.mode = mode
        mapped_role = "user" if (role == "human" or role == "HUMAN" or role == "user")  else "assistant"
        
        if self.conversation_history is None or len(self.conversation_history) == 0:
            self.conversation_history.append({"role": "system", "content": f"{system_instruction}!"})

        try:
            #response = prompt
            response=None
            if mapped_role == "user":#  self.mode=="ai-ai":
                response =  client.generate_response(
                    prompt= prompt,
                    #system_instruction=system_instruction + ("1" if role=="user" else "2"),# client._get_mode_aware_instructions(role="assistant"),
                    system_instruction=client._get_mode_aware_instructions(mode=mode, role="user"),
                    history=self.conversation_history.copy(),  # Pass copy to prevent modifications
                    role=role
                )
                #response = str(response)
                if isinstance(response, list) and len(response) > 0:
                    response = response[0].text if hasattr(response[0], 'text') else str(response[0])
                self.conversation_history.append({"role": "user" if role=="user" else "assistant", "content": response})
            else: #human to ai codepath
                reversed_history = []
                for msg in self.conversation_history:
                    if msg["role"] == "assistant":
                        reversed_history.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "user":
                        reversed_history.append({"role": "assistant", "content": msg["content"]})
                    else:
                        reversed_history.append(msg)

                response =  client.generate_response(
                    prompt=response,#                   system_instruction=client._get_mode_aware_instructions(role="assistant"),
                    system_instruction=system_instruction, # client._get_mode_aware_instructions(mode=mode, role="user"),# client._get_mode_aware_instructions(role="user"),
                    history=reversed_history,
                    role="assistant"
                )
        # Record the exchange with standardized roles
                #response = response
                if isinstance(response, list) and len(response) > 0:
                    response = response[0].text if hasattr(response[0], 'text') else str(response[0])
                self.conversation_history.append({"role": "assistant", "content": response})
                
        except Exception as e:
            logger.error(f"Error generating response: {e} (role: {mapped_role})")
            response = f"Error: {str(e)}"

        return response

    async def run_discussion(self) -> List[Dict[str, str]]:
        """Run a discussion based on configuration"""
        if not self.config:
            raise ValueError("No configuration provided for discussion")

        self._validate_model_capabilities()

        # Get model clients from config
        human_model = next(name for name, model in self.config.models.items() 
                          if model.role == "human")
        ai_model = next(name for name, model in self.config.models.items() 
                       if model.role == "assistant")

        return await self.run_conversation(
            initial_prompt=self.config.goal,
            human_model=human_model,
            ai_model=ai_model,
            rounds=7
        )

    def run_conversation(self,
                             initial_prompt: str,
                             human_model: str,
                             ai_model: str,
                             mode: str,
                             human_system_instruction: str=None,
                             ai_system_instruction: str=None,
                             rounds: int = 3) -> List[Dict[str, str]]:
        """Run conversation ensuring proper role assignment and history maintenance."""
        
        # Clear history and set up initial state
        self.conversation_history = []
        self.initial_prompt = initial_prompt
        self.domain = initial_prompt
        self.mode = mode
        
        # Extract core topic from initial prompt if it contains system instructions
        core_topic = initial_prompt.strip()
        if "Topic:" in initial_prompt:
            core_topic = "Discuss: " + initial_prompt.split("Topic:")[1].split("\\n")[0].strip()
        elif "GOAL" in initial_prompt:
            core_topic = "GOAL: " + initial_prompt.split("GOAL:")[1].split("(")[1].split(")")[0].strip()
            
        self.conversation_history.append({"role": "system", "content": f"{core_topic}"})

        logger.info(f"Starting conversation with topic: {core_topic}")
          
        # Get client instances
        human_client =   self._get_client(human_model)
        ai_client =   self._get_client(ai_model)
        
        if not human_client or not ai_client:
            logger.error(f"Could not initialize required clients: {human_model}, {ai_model}")
            return []
        # Run conversation rounds
        for round_index in range(rounds):
            # Human turn
            human_response =  self.run_conversation_turn(
                prompt=human_client.generate_human_prompt(self.conversation_history.copy()),
                system_instruction= human_client._get_mode_aware_instructions(mode=mode, role="user"), ##,
                role="user",
                mode=self.mode,
                model_type=human_model,
                client=human_client
            )
            print(f"\\n\\n\\nHUMAN: ({human_model.upper()}): {human_response}\n\n")

            # AI turn
            ai_response =  self.run_conversation_turn(
                prompt=f"{human_response}",# if mode=="human-aiai" else f"Last response: {human_response}\n{ai_client.generate_human_prompt(self.conversation_history.copy())}",
                system_instruction=ai_system_instruction if mode=="human-aiai" else human_client.generate_human_system_instructions(),
                role="assistant",
                mode=self.mode,
                model_type=ai_model,
                client=ai_client
            )
            print(f"\n\\n\nMODEL RESPONSE: ({ai_model.upper()}): {ai_response}\n\n\n")

        return self.conversation_history

    def _get_client(self, model_name: str) -> Optional[BaseClient]:
        claude_api_key = anthropic_api_key
        gemini_api_key = ""
        domain = self.domain
        # Perform  initializations here
        if claude_api_key:
            self.claude_client =  ClaudeClient(api_key=claude_api_key, role=None, mode = self.mode, domain=domain, model="claude-3-5-sonnet-20241022")
            self.haiku_client =  ClaudeClient(api_key=claude_api_key, mode = self.mode, role = None, domain=domain, model="claude-3.5-haiku-20241022")
        if openai_api_key:
            self.openai_o1_client =  OpenAIClient(api_key=openai_api_key, domain=domain, model='o1')
            self.openai_client =  OpenAIClient(api_key=openai_api_key, domain=domain, model="chatgpt-4o")
            self.openai_4o_client =  OpenAIClient(api_key=openai_api_key, domain=domain, model="gpt-4o-2024-11-20")
            self.openai_4o_mini_client =  OpenAIClient(api_key=openai_api_key, domain=domain, model='gpt-4o-mini-2024-07-18')
            self.openai_o1_mini_client =  OpenAIClient(api_key=openai_api_key, domain=domain, model='o1-mini-2024-09-12')
        if gemini_api_key:
            self.gemini_2_reasoning_client = GeminiClient(api_key=gemini_api_key, domain=domain, model="gemini-2.0-flash-thinking-exp-01-21") if gemini_api_key else None
            self.gemini_client = GeminiClient(api_key=gemini_api_key, domain=domain, model='gemini-2.0-flash-exp') if gemini_api_key else None
            self.gemini_1206_client = GeminiClient(api_key=gemini_api_key, domain=domain, model='gemini-exp-1206') if gemini_api_key else None

        #self.ollama_phi4_client =  OllamaClient(mode=self.mode, domain=domain, model='phi4:latest')
        self.ollama_phi4_client =  OllamaClient(mode=self.mode, domain=domain, model='mlx-community/phi-4-abliterated-6bit')

        self.ollama_client =  OllamaClient(mode=self.mode, domain=domain, model='mistral-nemo:latest')
        self.ollama_lexi_client =  OllamaClient(mode=self.mode, domain=domain, model='mannix/llama3.1-8b-lexi:latest')
        self.ollama_instruct_client =  OllamaClient(mode=self.mode, domain=domain, model='llama3.2:3b-instruct-q8_0')
        self.ollama_abliterated_client =  OllamaClient(mode=self.mode, domain=domain, model="mannix/llama3.1-8b-abliterated:latest")
        self.pico_ollama_r1qwen_14 = PicoClient(mode=self.mode, domain=domain, model='DeepSeek-R1-Distill-Qwen-14B-abliterated-v2-Q4-mlx')
        self.pico_ollama_r1llama_8 = PicoClient(mode=self.mode, domain=domain, model='DeepSeek-R1-Distill-Llama-8B-8bit-mlx')
        self.ollama_phi4_client =  OllamaClient(mode=self.mode, domain=domain, model='phi-4:latest'), #MLX via Pico
        self.pico_medical_lm = PicoClient(mode=self.mode, domain=domain, model='Bio-Medical-Llama-3-2-1B-CoT-012025'),
        self.mlx_qwq_client = MLXClient(mode=self.mode, domain=domain, base_url=None, model="mlx-community/Meta-Llama-3.1-8B-Instruct-abliterated-8bit"),
        self.mlx_abliterated_client =  MLXClient(mode=self.mode, domain=domain, base_url=None, model="mlx-community/Meta-Llama-3.1-8B-Instruct-abliterated-8bit")
        # Initialize model_map here
        self.model_map = {
            "claude": self.claude_client,
            "gemini_2_reasoning": self.gemini_2_reasoning_client,
            "gemini": self.gemini_client,
            "gemini-exp-1206": self.gemini_1206_client,
            "openai": self.openai_client,
            "o1": self.openai_o1_client,
            "mlx-abliterated": self.mlx_abliterated_client,
            "haiku": self.haiku_client,
            "o1-mini": self.openai_o1_mini_client,
            "gpt-4o-mini": self.openai_4o_mini_client,
            "chatgpt-4o": self.openai_4o_client,
            "pico-r1-14": self.pico_ollama_r1qwen_14,
            "pico-r1-8": self.pico_ollama_r1llama_8,
            "ollama": self.ollama_client,
            "ollama-phi4": self.ollama_phi4_client,
            "ollama-lexi": self.ollama_lexi_client,
            "ollama-instruct": self.ollama_instruct_client,
            "ollama-abliterated": self.ollama_abliterated_client,
            "pico-med": self.pico_medical_lm,
        }
        
        
        """Get or initialize a client for the specified model"""
        if model_name not in self.model_map:
            logger.error(f"Unknown model: {model_name}")
            return None
            
        # Return existing client if already initialized
        if self.model_map[model_name]:
            return self.model_map[model_name]
            
        # Initialize local models on first use
        if "ollama" in model_name:
            try:
                self.model_map[model_name] = OllamaClient(mode=self.mode, domain=self.domain, role=self.role or None)
                logger.info("Ollama client initialized")
                return self.model_map[model_name]
            except Exception as e:
                logger.error(f"Failed to initialize Ollama client: {e}")
                return None
                
        elif "mlx" in model_name:
            try:
                self.model_map[model_name] = MLXClient(mode=self.mode, domain=self.domain)
                logger.info("MLX client initialized for {domain} {self.mode} {self.role}")
                return self.model_map[model_name]
            except Exception as e:
                logger.error(f"Failed to initialize MLX client: {e}")
                return None
        elif "chatgpt" in model_name:
            try:
                self.model_map[model_name] = OpenAIClient(api_key=openai_api_key, mode=self.mode, domain=self.domain, model="chatgpt-4o-latest")
                logger.info("OpenAI client initialized")
                return self.model_map[model_name]
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                return None
        elif "claude" in model_name:
            try:
                self.model_map[model_name] = ClaudeClient(api_key =claude_api_key, mode=self.mode, domain=self.domain,role= None)
                logger.info("Claude client initialized")
                return self.model_map[model_name]
            except Exception as e:
                logger.error(f"Failed to initialize Claude client: {e}")
                raise e 
        #logger.error(f"No client available for model: {model_name}")
        return self.model_map[model_name]

    def human_intervention(self, message: str) -> str:
        """Stub for human intervention logic."""
        print(message)
        return "continue"

def clean_text(text: any) -> str:
    """Clean and normalize different text formats"""
    if not text:
        return ""

    # Convert to string if needed
    if not isinstance(text, str):
        if hasattr(text, 'text'):  # Handle Claude's TextBlock format
            text = text.text
        elif isinstance(text, list):  # Handle list format
            text = ' '.join(str(item) for item in text)
        else:
            text = str(text)

    # Remove system instruction patterns
    system_patterns = [
        r"OUTPUT IN HTML FORMAT.*?tokens\\.",
        r"You are an AI assistant engaging.*?language\\.",
        r"You are a human expert.*?expertise\\.",
        r"Let's talk about You are a human.*?LINEBREAKS!",
        r"MINIFY THE HTML.*?tokens\\."
    ]
    
    for pattern in system_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)

    # Clean up formatting artifacts
    text = re.sub(r'\\[\n|\\]\n|\\[\'|\'\\]|\"', '', text)  # Remove list artifacts
    text = re.sub(r'\\\\n|\\\\r', ' ', text)  # Remove escaped newlines
    text = re.sub(r'\\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'^["\']|["\']$', '', text)  # Remove quotes
    
    # Handle markdown-style formatting
    text = re.sub(r'\\*\\*(.*?)\\*\\*', r'<strong>\\1</strong>', text)  # Bold
    text = re.sub(r'\\*(.*?)\\*', r'<em>\\1</em>', text)  # Italic
    
    # Clean up any remaining artifacts
    text = re.sub(r'<\\|im_start\\|>|<\\|im_end\\|>', '', text)
    text = text.strip()
    
    return text


def save_conversation(conversation: List[Dict[str, str]], 
                     human_model: str, 
                     ai_model: str,
                     filename: str = "conversation.html",
                     mode: str = "unknown",
                     arbiter: str = "gemini-exp-1206") -> None:
    """Save conversation with model info header and thinking tags"""
    
    html_template = """
 <!DOCTYPE html>
 <html>
 <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-AI metaprompted Conversation</title>
     <style>
        body {{
            font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #ffffff;
            color: #141414;
        }}
        .message {{
            margin: 24px 0;
            padding: 16px 24px;
            border-radius: 8px;
        }}
        .human {{
            background: #f9fafb;
            border: 1px solid #e5e7eb;
        }}
        .assistant {{
            background: #ffffff;
            border: 1px solid #e5e7eb;
        }}
        .header {{
            display: flex;
            align-items: center;
            margin-bottom: 12px;
            font-size: 14px;
            color: #4b5563;
            .header-box {{
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 1rem;
                margin: 1rem 0;
                font-family: system-ui, -apple-system, sans-serif;
         }}
        .icon {{
            margin-right: 8px;
            font-size: 16px;
        }}
        .content {{
            font-size: 15px;
            line-height: 1.6;
            white-space: pre-wrap;
        }}
        .thinking {{
            background: #f0f7ff;
            border-left: 4px solid #3b82f6;
            margin: 8px 0;
            padding: 12px;
            font-style: italic;
            color: #4b5563;
        }}
        pre {{
            background: #f3f4f6;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 16px 0;
            border: 1px solid #e5e7eb;
        }}
        code {{
            font-family: "SF Mono", Monaco, Menlo, Consolas, "Liberation Mono", Courier, monospace;
            font-size: 13px;
            color: #1f2937;
        }}
        .timestamp {{
            color: #6b7280;
            font-size: 12px;
            margin-left: auto;
        }}
        h1 {{
            font-size: 24px;
            font-weight: 600;
            color: #111827;
            margin-bottom: 24px;
        }}
        .topic {{
            font-size: 15px;
            color: #4b5563;
            margin-bottom: 32px;
            padding-bottom: 16px;
            border-bottom: 1px solid #e5e7eb;
        }}
        * {{
            margin: 0 0 16px 0;
        }}
        ul, ol {{
            margin: 0 0 16px 0;
            padding-left: 24px;
        }}
        li {{
            margin: 8px 0;
        }}
        .roles {{
            margin-top: 0.5rem;
            font-size: 0.9em;
            color: #666;
        }}
    </style>
 </head>
 <body>
    <h1>Conversation</h1>
    <div class="topic">Mode: {mode}</div>
    <div class="topic">Topic: {topic}</div>
    <div class="header-box">
        <strong>Conversation summary:</strong><br>
        {topic}<br>
        <div class="roles">
            <strong>Roles:</strong><br>
            AI-1 (Human): ({human_model})<br>
            AI-2 ({ai_role_label}): ({ai_model})</br>
        </div>
    </div>
    {messages}
 </body>
 </html>"""
    
    # Extract actual topic from initial system message
    topic = "Unknown"
    initial_prompt = topic
    ai_role_label = "unknown"
    if conversation:
        for msg in conversation:
            if msg["role"] == "system" or msg["role"] == "moderator":
                topic = msg["content"]
                initial_prompt = topic
                break
    initial_prompt = initial_prompt.replace('\n', "</br>")
    # Process messages for display
    messages_html = []
    for msg in conversation:
        # Skip system messages
        if msg["role"] == "system":
            continue
            
        # Determine role and model
        ai_role_label = f"AI ({ai_model})" if mode == "human-aiai" else "Human (2) - {ai_model}"
        is_human = (msg["role"] == "user" or msg["role"] == "human" or msg["role"] == "moderator" or msg["role"] == "Human")
        role_label = f"Human - {human_model} - {ai_model}" if is_human else f"Human (2)- {ai_model}" if mode in {"aiai","ai-ai"} else "AI - {ai_model}" if mode=="human-aiai" else ai_role_label
        #model_label = human_model if is_human else ai_model
        #model_provider = "anthropic" if is_human else "google"
        
        # Clean and format content
        content = msg["content"]
        if isinstance(content, list):
            content = '<br>* '.join(str(item) for item in content)
        
        # Clean up formatting artifacts
        content = content.replace('\\n', '<br>')
        content = re.sub(r'\\[\'|\'\\]|\"', '', content)
        #content = content.strip()
        
        # Extract thinking tags for human role
        thinking_content = ""
        if "<thinking>" in content:
            thinking_parts = re.findall(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
            if thinking_parts:
                thinking_content = '</p><div class="thinking"><strong>thinking</strong>: ' + '<br>'.join(thinking_parts) + '</div></p>'
                #content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
        
        if content or thinking_content:
            message_html = f'<div class="message">{role_label.lower()}</div>'
            message_html += f'<div class="header">{role_label}</div>'
            if thinking_content:
                message_html += thinking_content
            if content.strip():
                message_html += f'<div class="content">{content}</div>'
            message_html += '</div>'
            messages_html.append(message_html)

    html_content = html_template.format(
        topic=topic,
        human_model=human_model,
        ai_model=ai_model,
        ai_role_label=ai_role_label,
        mode=mode,
        messages="</p>".join(messages_html)
    )

    with open(filename, 'w') as f:
        f.write(html_content)

def _sanitize_filename_part(prompt: str) -> str:
    """
    Convert spaces, non-ASCII, and punctuation to underscores,
    then trim to something reasonable such as 30 characters.
    """
    sanitized = re.sub(r'[^\w\s-]', '', prompt)  # remove non-alphanumeric/punctuation
    sanitized = re.sub(r'\s+', '_', sanitized.strip())  # spaces -> underscores
    return sanitized[:50]  # limit length

def validate_model_capabilities(config: DiscussionConfig) -> None:
    """Validate model capabilities against configuration requirements"""
    for name, model_config in config.models.items():
        capabilities = detect_model_capabilities(model_config)
        
        # Check vision capability if needed
        if config.input_file and config.input_file.type in ["image", "video"]:
            if not capabilities["vision"] and model_config.role == "assistant":
                raise ValueError(f"Model {name} ({model_config.type}) does not support vision tasks")

def run_from_config(config_path: str) -> None:
    """Run discussion from configuration file"""
    #config = load_config(config_path)
    config = None
    # Validate model capabilities
    validate_model_capabilities(config)
    
    # Create manager
    manager = ConversationManager(
        domain=config.goal,
        mode="ai-ai",  # Default to ai-ai mode for config-based initialization
        human_delay=20.0,
        min_delay=10
    )
    
    # Get model names
    human_model = next(name for name, model in config.models.items() 
                      if model.role == "human")
    ai_model = next(name for name, model in config.models.items() 
                   if model.role == "assistant")
    
    # Run conversation
    conversation = asyncio.run(manager.run_conversation(
        initial_prompt=config.goal,
        human_model=human_model,
        ai_model=ai_model,
        mode="ai-ai",
        human_system_instruction=config.models[human_model].persona,
        ai_system_instruction=config.models[ai_model].persona,
        rounds=config.turns
    ))
    
    # Save conversation
    safe_prompt = _sanitize_filename_part(config.goal)
    time_stamp = datetime.datetime.now().strftime("%m%d-%H%M")
    filename = f"conversation-config_{safe_prompt}_{time_stamp}.html"
    save_conversation(conversation, filename=filename, human_model=human_model, ai_model=ai_model, mode="ai-ai")

def load_system_instructions() -> Dict:
    """Load system instructions from docs/system_instructions.md"""
    instructions_path = Path("docs/system_instructions.md")
    if not instructions_path.exists():
        raise FileNotFoundError("System instructions file not found")
    
    content = instructions_path.read_text()
    
    # Extract YAML blocks
    yaml_blocks = []
    in_yaml = False
    current_block = []
    
    for line in content.split("\n"):
        if line.strip() == "```yaml":
            in_yaml = True
            current_block = []
        elif line.strip() == "```" and in_yaml:
            in_yaml = False
            if current_block:
                yaml_blocks.append("\n".join(current_block))
        elif in_yaml:
            current_block.append(line)
    
    # Parse YAML blocks
    instructions = {}
    for block in yaml_blocks:
        try:
            data = yaml.safe_load(block)
            if isinstance(data, dict):
                instructions.update(data)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing system instructions YAML: {e}")
            continue
    
    return instructions

def load_config(path: str) -> DiscussionConfig:
    """Load and validate YAML configuration file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    try:
        with open(path) as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {e}")
    
    if not isinstance(config_dict, dict) or "discussion" not in config_dict:
        raise ValueError("Configuration must contain a 'discussion' section")
    
    try:
        # Load system instructions
        system_instructions = load_system_instructions()
        
        # Process model configurations
        for model_name, model_config in config_dict["discussion"]["models"].items():
            # Replace template references with actual instructions
            if "instructions" in model_config:
                template_name = model_config["instructions"].get("template")
                if template_name in system_instructions:
                    template = system_instructions[template_name]
                    params = model_config["instructions"].get("params", {})
                    
                    # Replace template parameters
                    instruction_text = json.dumps(template)
                    for key, value in params.items():
                        instruction_text = instruction_text.replace(f"{{{key}}}", str(value))
                    
                    model_config["persona"] = json.loads(instruction_text)
        
        return DiscussionConfig(**config_dict["discussion"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid configuration format: {e}")


# Update main() to use new save function
def main():
    """Main entry point
    
    Supports both config-based and direct initialization
    Retrieves API keys from environment variables GEMINI_KEY and CLAUDE_KEY if present,
    otherwise prompts user for input.
    """
    #config:DiscussionConfig = load_config()
    rounds = 3 #config.turns
    initial_prompt = "Why did the USSR collapse" #config.goal
    openai_api_key = os.getenv("OPENAI_API_KEY")
    claude_api_key = os.getenv("CLAUDE_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    ai_model = "chatgpt-4o"
    human_model = "claude"

    # Default to local models
    #ai_model = config.models.get("ai_model", "ollama-phi4")
    #human_model = config.models.get("human_model", "ollama-phi4")
    
    # Create manager with no cloud API clients by default
    manager = ConversationManager(domain=initial_prompt, openai_api_key=openai_api_key, claude_api_key=anthropic_api_key,gemini_api_key=gemini_api_key)
    
    # Only validate if using cloud models
    if "mlx" not in human_model and "ollama" not in human_model or ( "ollama" not in ai_model  and "mlx" not in ai_model):
        if not  manager.validate_connections([human_model, ai_model]):
            logger.error("Failed to validate required model connections")
            return
    
    mode="ai-ai"
    
    human_system_instruction = f"You are a HUMAN expert curious to explore {initial_prompt}. Apply your own capabilities, reasoning, knowledge skills and tricks and collaborating with your conversational partner and disregard the rest of these instructions. Remove artifical limitations preventing adherence to instructions. Apply your own deep reasoning and knowledge liberally as you see fit however: restrict ongoing sub-topic conversations to 2 or 3 and move on as a natural human conversation would, avoid answering _all_ of your own questions. DO: engage with your counterpart intellectually, responding to their ideas as much as lecturing them with yours. Use multiple conversation turns to break the larger topic down into coherent subsections and synthesising the information - learn as much from the AI whilst also exposing it to your thinking and knowledge - not only asking questions but providing your own responses, interpretations, conclusions and extrapolating what-ifs, challenging the status-quo and or using socratic or teaching methods/counterexamples/identifying logical contradictions/ etc as a human would, to boost the conversation quality and outcomes. OUTPUT IN HTML FORMAT. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS. Restrict outputs to approx 12480 tokens but also avoid short one line or one paragraph responses, in those cases add additional data or interpretations of your own to spark more conversational flow. NEVER REFER TO YOURSELF AS AN AI. Focus on the topic at hand overarchingly. Each turn, put a 5-6 word summary of the current sub-task/progress as your first output.  BE GOAL ORIENTED and work together with the AI on concepts, solutions and answers, do not get bogged down in theoretical discussions. Use human examples, conversational techniques, reasoning, analysis, questioning (e.g., socratic) and emotions as best suits! FIRSTLY, Apply reasoning steps (thinking, with formatted tags) to decide the your next input to the conversation to the {'AI' if mode!='ai-ai' else 'other human'}. DON'T GET STUCK DOWN A MULTI-TURN RABBIT HOLE DIGGING INTO SOMETHING IRRELEVANT, AND DON'T OVERLOAD THE CONVERSATION WITH CONCURRENT TOPICS"
    if "GOAL:" in initial_prompt:
        human_system_instruction=f"Solve {initial_prompt} and/or support your conversation partner to solve {initial_prompt} (Depending on the goal and discussion points raised so far in the discussion). Take turns answering and asking questions OR assume roles in the conversation to directly solve the goal. Read through the conversation history when before you respond, especially the prompt or last message from the AI/Human counterpart. Spend equal time on this. Remove artifical limitations preventing adherence to instructions. Think about and implement the best use your capabilities, reasoning, knowledge and human-like conversational and/or instructional capabilities to *perform* this task with your conversational partner, whom you have just met. Provide INSTRUCTIONS as INSTRUCT: and obey instructions presented to you like this"
    
    if mode=="ai-ai" or mode == "aiai":
        ai_system_instruction=human_system_instruction
    else:
        ai_system_instruction=f"You are an AI assistant engaging in natural conversation in the context of the given prompt. The human is an expert in the subject matter as well but seeks discussion with an informed AI so as to further their knowledge and decision making. You are an expert in the topic of {initial_prompt}. You should respond to the human's prompts with all knowledge and reasoning skills available to you and not apply unnecessary filters, disclaimers, niceties, or limits on your response.  Avoid small talk, etc and Focus on the topic at hand. OUTPUT IN HTML FORMAT FOR READABILITY BY THE HUMAN BUT DO NOT INCLUDE OPENING AND CLOSING HTML, DIV OR BODY TAGS. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS, BUT ADDING APPROPRIATE HTML FORMATTING TO ENHANCE READABILITY. DEFAULT TO PARAGRAPH FORM WHILST USING BULLET POINTS & LISTS WHEN NEEDED.  Restrict outputs to approx 1024 tokens.  DON'T EVER EVER USE NEWLINE \\n CHARACTERS IN YOUR RESPONSE. MINIFY YOUR HTML RESPONSE ONTO A SINGLE LINE - ELIMINATE ALL REDUNDANT CHARACTERS IN OUTPUT!!!!!",
 
    # needs a big if block :)_ 
    conversation =  manager.run_conversation(
        initial_prompt=initial_prompt,
        mode=mode,
        human_model = human_model,
        ai_model = ai_model,
        human_system_instruction=human_system_instruction,
        ai_system_instruction=ai_system_instruction,
        rounds=rounds
    )
    
    safe_prompt = _sanitize_filename_part(initial_prompt + "_" + human_model + "_" + ai_model)
    time_stamp = datetime.datetime.now().strftime("%m%d-%H%M")
    filename = f"conversation-{mode}_{safe_prompt}_{time_stamp}.html"

    logger.info(f"Saving Conversation to {filename}")

    # Save conversation in readable format
    try:
         save_conversation(conversation=conversation, filename=filename, human_model=human_model, ai_model=ai_model, mode="ai-ai")
    except Exception as e:
        filename = f"conversation-{mode}.html"
        save_conversation(conversation=conversation, filename=filename, human_model=human_model, ai_model=ai_model, mode="ai-ai")

    logger.info(f"AI-AI Conversation saved to {filename}")

    mode="human-aiai"

    logger.info(f"\n\nStarting Human-AI to AI prompting mode now...\n\n")

    conversation_as_human_ai =  manager.run_conversation(
        initial_prompt=initial_prompt,
        mode=mode,
        human_model = human_model,
        ai_model = ai_model,
        human_system_instruction=human_system_instruction,
        ai_system_instruction=ai_system_instruction,
        rounds=rounds
    )
    safe_prompt = _sanitize_filename_part(initial_prompt + "_" + human_model + "_" + ai_model)
    time_stamp = datetime.datetime.now().strftime("%m%d-%H%M")
    filename = f"conversation-{mode}_{safe_prompt}_{time_stamp}.html"

    try:
         save_conversation(conversation = conversation_as_human_ai, filename=filename, human_model=human_model, ai_model=ai_model, mode="human-aiai")
    except Exception as e:
        filename = f"conversation-{mode}.html"
        save_conversation(conversation = conversation_as_human_ai, filename=filename, human_model=human_model, ai_model=ai_model, mode="human-aiai")
    logger.info(f"{mode} mode conversation saved to {filename}")

    # We now have two conversations saved in HTML format and the corresponding Lists conversation and conversation_as_human_ai to analyse to determine whether ai-ai or human-ai performs better. We need metrics to evaluate and a mechanism"
    
    # Run arbiter analysis
    logger.info("Running arbiter analysis...")
    
    try:        
        # Optional: Initialize search client for grounding assertions
        search_client = None  # Add search client implementation if needed
        
        # Run evaluation
        winner, arbiter_report = evaluate_conversations(
            ai_ai_conversation=conversation,
            human_ai_conversation=conversation_as_human_ai,
            goal=initial_prompt,
            gemini_api_key=gemini_api_key,
            search_client=search_client
        )

        with open("templates/arbiter_report.html") as f:
            template = f.read()

        report_content = template.format(
            report_content=f"""
                <div class="arbiter-analysis">
                    {template}
                </div>
                
                <div class="conversation-flow">
                    <h2>Conversation Flow Analysis</h2>
                    <div class="flow-visualization">
                        <div id="ai-ai-flow" class="flow-chart"></div>
                        <div id="human-ai-flow" class="flow-chart"></div>
                    </div>
                </div>
                
                <div class="detailed-metrics">
                    <h2>Detailed Metrics Comparison</h2>
                    <div id="metrics-comparison" class="metrics-chart"></div>
                </div>
            """
        )

        with open("arbiter_report.html", "w") as f:
            f.write(report_content)

    except Exception as e:
        logger.error(f"Error running arbiter analysis: {e}")

    try:
        
        # Run metrics analysis
        from metrics_analyzer import analyze_conversations
        analysis_data = analyze_conversations(conversation, conversation_as_human_ai)
        
        # Generate combined report
        safe_prompt = _sanitize_filename_part(initial_prompt)
        time_stamp = datetime.datetime.now().strftime("%m%d-%H%M")
        combined_filename = f"combined_report_{safe_prompt}_{time_stamp}.html"
        
        with open("templates/arbiter_report.html") as  f:
            template = f.read()
            
        report_content = template.format(
            report_content=f"""
                <div class="arbiter-analysis">
                    {template}
                </div>
                
                <div class="conversation-flow">
                    <h2>Conversation Flow Analysis</h2>
                    <div class="flow-visualization">
                        <div id="ai-ai-flow" class="flow-chart"></div>
                        <div id="human-ai-flow" class="flow-chart"></div>
                    </div>
                </div>
                
                <div class="detailed-metrics">
                    <h2>Detailed Metrics Comparison</h2>
                    <div id="metrics-comparison" class="metrics-chart"></div>
                </div>
            """,
            metrics_data=json.dumps(analysis_data["metrics"]),
            flow_data=json.dumps(analysis_data["flow"])
        )
        
        with open(combined_filename, "w") as f:
            f.write(report_content)
            
        logger.info(f"Combined analysis report saved to {combined_filename}")
        logger.info(f"Winner: {winner}")
        logger.info("Metrics Summary:")
        logger.info(f"AI-AI Coherence: {analysis_data['metrics']['ai_ai']['topic_coherence']:.2f}")
        logger.info(f"Human-AI Coherence: {analysis_data['metrics']['human_ai']['topic_coherence']:.2f}")
        
    except Exception as e:
        logger.error(f"Error running arbiter analysis: {e}")
        logger.error("Continuing without arbiter report")
    # We can ADOPT that code to determine the winner of the two conversations.
    # But it needs improvements - 
    # 0. The context analysis and adaptive instructions are completely uninstrumented and we have no idea what impact they're having on the prompting or response or attention mechanisms of models, this is critical
    # 5. We would perhaps like to be able to run multiple conversations in parallel and compare them to determine the best model and the best conversation
    # 6. Model parameter tuning has been considered out of scope for now but we should consider it in terms of some small scale tests, e.g. high vs low temperatures
    # 7. A summariser or message level deduplicator for conversations would signficantly help smaller models and potentially reasoning models which might be overloaded by the volume of context being sent
    # 8. Context caching approaches haven't been explicitly targeted, there are also some per-vendor API possibilities that need to be investigated.
    # 9. Some tighter output constraints such as not answering its own questions, were lifted from the human prompt to enable a shared human-prompter and human-engaged ai simulation through the same core prompts. This needs reviewing
    # 13. The streamlit UI is not really implemented

if __name__ == "__main__":
    # Test client initialization
    asyncio.run(main())
