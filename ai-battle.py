"""AI model conversation manager with memory optimizations."""
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
from typing import List, Dict, Optional, TypeVar, Any
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
from configdataclasses import TimeoutConfig, FileConfig, ModelConfig, DiscussionConfig
from arbiter_v4 import evaluate_conversations, VisualizationGenerator, ArbiterResult
from file_handler import ConversationMediaHandler, FileConfig as MediaConfig
from model_clients import BaseClient, OpenAIClient, ClaudeClient, GeminiClient, MLXClient, OllamaClient, PicoClient
from shared_resources import MemoryManager
from metrics_analyzer import analyze_conversations

T = TypeVar('T')
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
CONFIG_PATH = "config.yaml"

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
    max_tokens: int = 1024
    stop_sequences: List[str] = None
    seed: Optional[int] = random.randint(0, 1000)

@dataclass
class ConversationManager:
    """Manages conversations between AI models with memory optimization."""
    
    def __init__(self,
                 config: DiscussionConfig   = None,
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
        self._media_handler = None  # Lazy initialization
        self.min_delay = min_delay
        self.conversation_history: List[Dict[str, str]] = []
        self.is_paused = False
        self.initial_prompt = domain
        self.rate_limit_lock = asyncio.Lock()
        self.last_request_time = 0
        
        # Store API keys
        self.openai_api_key = openai_api_key
        self.claude_api_key = claude_api_key
        self.gemini_api_key = gemini_api_key
        
        # Initialize empty client tracking
        self._initialized_clients = set()
        self.model_map = {}

    @property
    def media_handler(self):
        """Lazy initialization of media handler."""
        if self._media_handler is None:
            self._media_handler = ConversationMediaHandler(self.domain)
        return self._media_handler

    def _get_client(self, model_name: str) -> Optional[BaseClient]:
        """Get or create a client instance."""
        if model_name not in self._initialized_clients:
            try:
                if model_name == "claude":
                    client = ClaudeClient(role=None, api_key=self.claude_api_key, mode=self.mode, domain=self.domain, model="claude-3-5-sonnet-20241022")
                elif model_name == "haiku":
                    client = ClaudeClient(role=None, api_key=self.claude_api_key, mode=self.mode, domain=self.domain, model="claude-3-5-haiku-20241022")
                elif model_name == "gpt-4o":
                    client = OpenAIClient(api_key=self.openai_api_key, role=None, mode=self.mode, domain=self.domain, model="gpt-4o-2024-08-06")
                elif model_name == "gpt-4o-mini":
                    client = OpenAIClient(api_key=self.openai_api_key, role=None, mode=self.mode, domain=self.domain, model="gpt-4o-mini")
                elif model_name == "o1":
                    client = OpenAIClient(api_key=self.openai_api_key, role=None, mode=self.mode, domain=self.domain, model="o1")
                elif model_name == "gemini":
                    client = GeminiClient(api_key=self.gemini_api_key, role=None, mode=self.mode, domain=self.domain, model="gemini-2.0-flash-exp")
                elif model_name == "gemini_2_reasoning":
                    client = GeminiClient(api_key=self.gemini_api_key, role=None, mode=self.mode, domain=self.domain, model="gemini-2.0-flash-thinking-exp-01-21")
                elif model_name == "mlx-qwq":
                    client = MLXClient(mode=self.mode, domain=self.domain, model="mlx-community/Meta-Llama-3.1-8B-Instruct-abliterated-8bit")
                elif model_name == "mlx-abliterated":
                    client = MLXClient(mode=self.mode, domain=self.domain, model="mlx-community/Meta-Llama-3.1-8B-Instruct-abliterated-8bit")
                elif model_name == "pico-r1-14":
                    client = PicoClient(mode=self.mode, domain=self.domain, model="DeepSeek-R1-Distill-Qwen-14B-abliterated-v2-Q4-mlx")
                elif model_name == "pico-r1-8":
                    client = PicoClient(mode=self.mode, domain=self.domain, model="DeepSeek-R1-Distill-Llama-8B-8bit-mlx")
                elif model_name == "pico-med":
                    client = PicoClient(mode=self.mode, domain=self.domain, model="Bio-Medical-Llama-3-2-1B-CoT-012025")
                elif model_name == "ollama":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="mannix/llama3.1-8b-lexi:latest")
                elif model_name == "ollama-phi4":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="phi4:latest")
                elif model_name == "ollama-lexi":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="mannix/llama3.1-8b-lexi:latest")
                elif model_name == "ollama-instruct":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="llama3.2:3b-instruct-q8_0")
                elif model_name == "ollama-abliterated":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="mannix/llama3.1-8b-abliterated:latest")
                else:
                    logger.error(f"Unknown model: {model_name}")
                    return None
                
                logger.info(f"Created client for model: {model_name}")
                logger.debug(MemoryManager.get_memory_usage())
                
                if client:
                    self.model_map[model_name] = client
                    self._initialized_clients.add(model_name)
            except Exception as e:
                logger.error(f"Failed to create client for {model_name}: {e}")
                return None
        return self.model_map.get(model_name)

    def cleanup_unused_clients(self):
        """Clean up clients that haven't been used recently."""
        for model_name in list(self._initialized_clients):
            if model_name not in self.model_map:
                continue
            client = self.model_map[model_name]
            if hasattr(client, '__del__'):
                client.__del__()
            del self.model_map[model_name]
            self._initialized_clients.remove(model_name)
        logger.debug(MemoryManager.get_memory_usage())

    def validate_connections(self, required_models: List[str] = None) -> bool:
        """Validate required model connections."""
        if required_models is None:
            required_models = [name for name, client in self.model_map.items()
                           if client and name not in ["ollama", "mlx"]]
            
        if not required_models:
            logger.info("No models require validation")
            return True
            
        validations = []
        return True

    def rate_limited_request(self):
        """Apply rate limiting to requests."""
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
        self.mode = mode
        mapped_role = "user" if (role == "human" or role == "HUMAN" or role == "user") else "assistant"
        
        if not self.conversation_history:
            self.conversation_history.append({"role": "system", "content": f"{system_instruction}!"})

        try:
            if mapped_role == "user" or mapped_role == "human":
                response = client.generate_response(
                    prompt=prompt,
                    system_instruction=client._update_instructions(history = self.conversation_history, mode=mode, role="user"),
                    history=self.conversation_history.copy(),  # Limit history
                    role=role
                )
                if isinstance(response, list) and len(response) > 0:
                    response = response[0].text if hasattr(response[0], 'text') else str(response[0])
                
                self.conversation_history.append({"role": "user" if role=="user" else "assistant", "content": response})
            else:
                reversed_history = []
                for msg in self.conversation_history:  # Limit history
                    if msg["role"] == "assistant":
                        reversed_history.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "user":
                        reversed_history.append({"role": "assistant", "content": msg["content"]})
                    else:
                        reversed_history.append(msg)

                response = client.generate_response(
                    prompt=prompt,
                    system_instruction=client._update_instructions(history = reversed_history, mode=mode, role="assistant"),
                    history=reversed_history,
                    role="assistant"
                )
                if isinstance(response, list) and len(response) > 0:
                    response = response[0].text if hasattr(response[0], 'text') else str(response[0])
                self.conversation_history.append({"role": "assistant", "content": response})
            print (f"\n\n\n{mapped_role.upper()}: {response}\n\n\n")
                
        except Exception as e:
            logger.error(f"Error generating response: {e} (role: {mapped_role})")
            raise e
            response = f"Error: {str(e)}"

        return response

    def run_conversation(self,
                        initial_prompt: str,
                        human_model: str,
                        ai_model: str,
                        mode: str,
                        human_system_instruction: str=None,
                        ai_system_instruction: str=None,
                        rounds: int = 1) -> List[Dict[str, str]]:
        """Run conversation ensuring proper role assignment and history maintenance."""
        
        # Clear history and set up initial state
        self.conversation_history = []
        self.initial_prompt = initial_prompt
        self.domain = initial_prompt
        self.mode = mode
        
        # Extract core topic from initial prompt
        core_topic = initial_prompt.strip()
        if "Topic:" in initial_prompt:
            core_topic = "Discuss: " + initial_prompt.split("Topic:")[1].split("\\n")[0].strip()
        elif "GOAL" in initial_prompt:
            core_topic = "GOAL: " + initial_prompt.split("GOAL:")[1].split("(")[1].split(")")[0].strip()
            
        self.conversation_history.append({"role": "system", "content": f"{core_topic}"})

        logger.info(f"Starting conversation with topic: {core_topic}")
          
        # Get client instances
        human_client = self._get_client(human_model)
        ai_client = self._get_client(ai_model)
        
        if not human_client or not ai_client:
            logger.error(f"Could not initialize required clients: {human_model}, {ai_model}")
            return []

        ai_response = core_topic
        try:
            # Run conversation rounds
            for round_index in range(rounds):
                # Human turn
                human_response = self.run_conversation_turn(
                    prompt=ai_response,  # Limit history
                    system_instruction=human_client._get_mode_aware_instructions(mode=mode, role="user"),
                    role="user",
                    mode=self.mode,
                    model_type=human_model,
                    client=human_client
                )
                #print(f"\n\n\nHUMAN: ({human_model.upper()}): {human_response}\n\n")

                # AI turn
                ai_response = self.run_conversation_turn(
                    prompt=human_response,
                    system_instruction=ai_system_instruction if mode=="human-aiai" else human_client.generate_human_system_instructions(),
                    role="assistant",
                    mode=self.mode,
                    model_type=ai_model,
                    client=ai_client
                )
                logger.debug(f"\n\n\nMODEL RESPONSE: ({ai_model.upper()}): {ai_response}\n\n\n")

            # Clean up unused clients
            self.cleanup_unused_clients()
            
            return self.conversation_history
            
        finally:
            # Ensure cleanup happens even if there's an error
            self.cleanup_unused_clients()
            MemoryManager.cleanup_all()

async def save_conversation(conversation: List[Dict[str, str]], 
                     filename: str,
                     human_model: str,
                     ai_model: str,
                     mode: str = None) -> None:
    """Save conversation to HTML file."""
    try:
        with open("templates/conversation.html", "r") as f:
            template = f.read()
            
        conversation_html = ""
        for msg in conversation:
            role = msg["role"]
            content = msg.get("content", "")
            if isinstance(content, (list, dict)):
                content = str(content)
            
            if role == "system":
                conversation_html += f'<div class="system-message">{content} ({mode})</div>\n'
            elif role in ["user", "human"]:
                conversation_html += f'<div class="human-message"><strong>Human ({human_model}):</strong> {content}</div>\n'
            elif role == "assistant":
                conversation_html += f'<div class="ai-message"><strong>AI ({ai_model}):</strong> {content}</div>\n'

        with open(filename, "w") as f:
            f.write(template % {'conversation': conversation_html})
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")

def _sanitize_filename_part(prompt: str) -> str:
    """
    Convert spaces, non-ASCII, and punctuation to underscores,
    then trim to something reasonable such as 30 characters.
    """
    # Remove non-alphanumeric/punctuation
    sanitized = re.sub(r'[^\w\s-]', '', prompt)
    # Convert spaces to underscores
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    # Limit length
    return sanitized[:50]

async def save_arbiter_report(report: Dict[str, Any]) -> None:
    """Save arbiter analysis report with visualization support."""
    try:
        with open("templates/arbiter_report.html") as f:
            template = f.read()

        # Ensure proper report structure
        if isinstance(report, str):
            report = {
                "content": report,
                "metrics": {
                    "conversation_quality": {},
                    "participant_analysis": {},
                },
                "flow": {},
                "visualizations": {},
                "winner": "No clear winner determined",
                "assertions": [],
                "key_insights": [],
                "improvement_suggestions": []
            }

        # Generate visualizations if metrics are available
        viz_generator = VisualizationGenerator()
        metrics_chart = ""
        timeline_chart = ""
        if report.get("metrics", {}).get("conversation_quality"):
            metrics_chart = viz_generator.generate_metrics_chart(report["metrics"])
            timeline_chart = viz_generator.generate_timeline(report.get("flow", {}))

        # Format report content
        report_content = template % {
            'report_content': report.get("content", "No content available"),
            'metrics_data': json.dumps(report.get("metrics", {})),
            'flow_data': json.dumps(report.get("flow", {})),
            'metrics_chart': metrics_chart,
            'timeline_chart': timeline_chart,
            'winner': report.get("winner", "No clear winner determined")
        }

        # Save report with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"arbiter_report_{timestamp}.html"
        
        with open(filename, "w") as f:
            f.write(report_content)
            
        logger.info(f"Arbiter report saved successfully as {filename}")
        
        # Create symlink to latest report
        #latest_link = "arbiter_report_latest.html"
        #if os.path.exists(latest_link):
        #    os.remove(latest_link)
        #os.symlink(filename, latest_link)
        
    except Exception as e:
        logger.error(f"Failed to save arbiter report: {e}")

async def save_metrics_report(ai_ai_conversation: List[Dict[str, str]], 
                       human_ai_conversation: List[Dict[str, str]]) -> None:
    """Save metrics analysis report."""
    try:
        if ai_ai_conversation and human_ai_conversation:
            analysis_data = analyze_conversations(ai_ai_conversation, human_ai_conversation)
            logger.info("Metrics report generated successfully")
        else:
            logger.info("Skipping metrics report - empty conversations")
    except Exception as e:
        logger.error(f"Failed to generate metrics report: {e}")

async def main():
    """Main entry point."""
    rounds = 8
    initial_prompt = "Lasting effects of the cold war"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    claude_api_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    mode = "ai-ai"
    ai_model = "gemini_2_reasoning"
    human_model = "claude"
    
    # Create manager with no cloud API clients by default
    manager = ConversationManager(
        domain=initial_prompt,
        openai_api_key=openai_api_key,
        claude_api_key=anthropic_api_key,
        gemini_api_key=gemini_api_key
    )
    
    # Only validate if using cloud models
    if "mlx" not in human_model and "ollama" not in human_model or ("ollama" not in ai_model and "mlx" not in ai_model):
        if not manager.validate_connections([human_model, ai_model]):
            logger.error("Failed to validate required model connections")
            return
    
    
    human_system_instruction = f"You are a HUMAN expert curious to explore {initial_prompt}..."  # Truncated for brevity
    if "GOAL:" in initial_prompt:
        human_system_instruction = f"Solve {initial_prompt} together..."  # Truncated for brevity
    
    ai_system_instruction = f"You are a helpful assistant. Think step by step and respond to the user."  # Truncated for brevity
    if mode == "ai-ai" or mode == "aiai":
        ai_system_instruction = human_system_instruction
 
    try:
        # Run AI-AI conversation
        conversation = manager.run_conversation(
            initial_prompt=initial_prompt,
            mode=mode,
            human_model=human_model,
            ai_model=ai_model,
            human_system_instruction=human_system_instruction,
            ai_system_instruction=ai_system_instruction,
            rounds=rounds
        )
        
        safe_prompt = _sanitize_filename_part(initial_prompt[:20] + "_" + human_model + "_" + ai_model)
        time_stamp = datetime.datetime.now().strftime("%m%d%H%M")
        filename = f"conv-aiai_{safe_prompt}_{time_stamp}.html"
        await save_conversation(conversation=conversation, filename=f"{filename}", human_model=human_model, ai_model=ai_model, mode="ai-ai")
        
        # Run human-AI conversation
        mode = "human-aiai"
        conversation_as_human_ai = manager.run_conversation(
            initial_prompt=initial_prompt,
            mode=mode,
            human_model=human_model,
            ai_model=ai_model,
            human_system_instruction=human_system_instruction,
            ai_system_instruction=ai_system_instruction,
            rounds=rounds
        )
        
        safe_prompt = _sanitize_filename_part(initial_prompt[:20] + "_" + human_model + "_" + ai_model)
        time_stamp = datetime.datetime.now().strftime("%m%d%H%M")
        filename = f"conv-humai_{safe_prompt}_{time_stamp}.html"
        await save_conversation(conversation=conversation_as_human_ai, filename=f"{filename}", human_model=human_model, ai_model=ai_model, mode="human-ai")
        
        # Run analysis
        arbiter_report = evaluate_conversations(
            ai_ai_convo=conversation,
            human_ai_convo=conversation_as_human_ai,
            goal=initial_prompt,
        )

        print(arbiter_report)
        
        # Generate reports
        await save_arbiter_report(arbiter_report)
        await save_metrics_report(conversation, conversation_as_human_ai)
        
    finally:
        # Ensure cleanup
        manager.cleanup_unused_clients()
        MemoryManager.cleanup_all()

if __name__ == "__main__":
    asyncio.run(main())
