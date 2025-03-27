"""AI model conversation manager with memory optimizations."""
import json
import os
import datetime
import base64
import sys
import time
import random
import logging
import re
from typing import List, Dict, Optional, TypeVar, Any, Union
from dataclasses import dataclass
import io
import asyncio
# Local imports
from configuration import load_config, DiscussionConfig, detect_model_capabilities
from configdataclasses import FileConfig, DiscussionConfig
from arbiter_v4 import evaluate_conversations, VisualizationGenerator
from file_handler import ConversationMediaHandler
from model_clients import BaseClient, OpenAIClient, ClaudeClient, GeminiClient, MLXClient, OllamaClient, PicoClient
from shared_resources import MemoryManager
from metrics_analyzer import analyze_conversations

T = TypeVar('T')
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
CONFIG_PATH = "config.yaml"
TOKENS_PER_TURN = 1280
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_battle.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Model templates for accessing different model versions with reasoning levels
OPENAI_MODELS = {
    # Base models (text-only with reasoning support)
    "o1": {"model": "o1", "reasoning_level": "auto", "multimodal": False},
    "o3": {"model": "o3", "reasoning_level": "auto", "multimodal": False},
    
    # O1 with reasoning levels (text-only)
    "o1-reasoning-high": {"model": "o1", "reasoning_level": "high", "multimodal": False},
    "o1-reasoning-medium": {"model": "o1", "reasoning_level": "medium", "multimodal": False},
    "o1-reasoning-low": {"model": "o1", "reasoning_level": "low", "multimodal": False},
    
    # O3 with reasoning levels (text-only)
    "o3-reasoning-high": {"model": "o3", "reasoning_level": "high", "multimodal": False},
    "o3-reasoning-medium": {"model": "o3", "reasoning_level": "medium", "multimodal": False},
    "o3-reasoning-low": {"model": "o3", "reasoning_level": "low", "multimodal": False},
    
    # Multimodal models without reasoning parameter
    "gpt-4o": {"model": "gpt-4o", "reasoning_level": None, "multimodal": True},
    "gpt-4o-mini": {"model": "gpt-4o-mini", "reasoning_level": None, "multimodal": True},
}

CLAUDE_MODELS = {
    # Base models (newest versions)
    "claude": {"model": "claude-3-5-sonnet", "reasoning_level": None, "extended_thinking": False},
    "sonnet": {"model": "claude-3-5-sonnet", "reasoning_level": None, "extended_thinking": False},
    "haiku": {"model": "claude-3-5-haiku", "reasoning_level": None, "extended_thinking": False},
    
    # Specific versions
    "claude-3-5-sonnet": {"model": "claude-3-5-sonnet", "reasoning_level": None, "extended_thinking": False},
    "claude-3-5-haiku": {"model": "claude-3-5-haiku", "reasoning_level": None, "extended_thinking": False},
    "claude-3-7": {"model": "claude-3-7-sonnet", "reasoning_level": "auto", "extended_thinking": False},
    "claude-3-7-sonnet": {"model": "claude-3-7-sonnet", "reasoning_level": "auto", "extended_thinking": False},
    
    # Claude 3.7 with reasoning levels
    "claude-3-7-reasoning": {"model": "claude-3-7-sonnet", "reasoning_level": "high", "extended_thinking": False},
    "claude-3-7-reasoning-high": {"model": "claude-3-7-sonnet", "reasoning_level": "high", "extended_thinking": False},
    "claude-3-7-reasoning-medium": {"model": "claude-3-7-sonnet", "reasoning_level": "medium", "extended_thinking": False},
    "claude-3-7-reasoning-low": {"model": "claude-3-7-sonnet", "reasoning_level": "low", "extended_thinking": False},
    "claude-3-7-reasoning-none": {"model": "claude-3-7-sonnet", "reasoning_level": "none", "extended_thinking": False},
    
    # Claude 3.7 with extended thinking
    "claude-3-7-extended": {"model": "claude-3-7-sonnet", "reasoning_level": "high", "extended_thinking": True, "budget_tokens": 8000},
    "claude-3-7-extended-deep": {"model": "claude-3-7-sonnet", "reasoning_level": "high", "extended_thinking": True, "budget_tokens": 16000},
}

@dataclass
class ModelConfig:
    """Configuration for AI model parameters"""
    temperature: float = 0.8
    max_tokens: int = 1280
    stop_sequences: List[str] = None
    seed: Optional[int] = random.randint(0, 1000)

@dataclass
class ConversationManager:
    """Manages conversations between AI models with memory optimization."""

    def __init__(self,
                 config: DiscussionConfig   = None,
                 domain: str = "General knowledge",
                 human_delay: float = 4.0,
                 mode: str = None,
                 min_delay: float = 2,
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
            self._media_handler = ConversationMediaHandler(output_dir="processed_files")
        return self._media_handler

    def _get_client(self, model_name: str) -> Optional[BaseClient]:
        """
        Get an existing client instance or create a new one for the specified model.

        This method manages client instances, creating them on demand and caching them
        for reuse. It supports various model types including Claude, GPT, Gemini, MLX,
        Ollama, and Pico models.

        Args:
            model_name: Name of the model to get or create a client for

        Returns:
            Optional[BaseClient]: Client instance if successful, None if the model
                                 is unknown or client creation fails
        """
        if model_name not in self._initialized_clients:
            try:
                # Handle Claude models using templates
                if model_name in CLAUDE_MODELS:
                    model_config = CLAUDE_MODELS[model_name]
                    client = ClaudeClient(
                        role=None, 
                        api_key=self.claude_api_key, 
                        mode=self.mode, 
                        domain=self.domain, 
                        model=model_config["model"]
                    )
                    
                    # Set reasoning level if specified
                    if model_config["reasoning_level"] is not None:
                        client.reasoning_level = model_config["reasoning_level"]
                        logger.debug(f"Set reasoning level to '{model_config['reasoning_level']}' for {model_name}")
                    
                    # Set extended thinking if enabled
                    if model_config.get("extended_thinking", False):
                        budget_tokens = model_config.get("budget_tokens", None)
                        client.set_extended_thinking(True, budget_tokens)
                        logger.debug(f"Enabled extended thinking with budget_tokens={budget_tokens} for {model_name}")
                
                # Handle OpenAI models using templates
                elif model_name in OPENAI_MODELS:
                    model_config = OPENAI_MODELS[model_name]
                    client = OpenAIClient(
                        api_key=self.openai_api_key, 
                        role=None, 
                        mode=self.mode, 
                        domain=self.domain, 
                        model=model_config["model"]
                    )
                    
                    # Set reasoning level if specified
                    if model_config["reasoning_level"] is not None:
                        client.reasoning_level = model_config["reasoning_level"]
                        logger.debug(f"Set reasoning level to '{model_config['reasoning_level']}' for {model_name}")
                
                # Handle other model types
                elif model_name == "gpt-4o":
                    client = OpenAIClient(api_key=self.openai_api_key, role=None, mode=self.mode, domain=self.domain, model="chatgpt-latest")
                elif model_name == "gpt-4o-mini":
                    client = OpenAIClient(api_key=self.openai_api_key, role=None, mode=self.mode, domain=self.domain, model="gpt-4o-mini")
                elif model_name == "gemini":
                    client = GeminiClient(api_key=self.gemini_api_key, role=None, mode=self.mode, domain=self.domain, model="gemini-2.0-flash-exp")
                elif model_name == "gemini-2-reasoning":
                    client = GeminiClient(api_key=self.gemini_api_key, role=None, mode=self.mode, domain=self.domain, model="gemini-2.0-flash-thinking-exp-01-21")
                elif model_name == "gemini-2.0-flash-exp":
                    client = GeminiClient(api_key=self.gemini_api_key, role=None, mode=self.mode, domain=self.domain, model="gemini-2.0-flash-exp")
                elif model_name == "gemini-2-pro":
                    client = GeminiClient(api_key=self.gemini_api_key, role=None, mode=self.mode, domain=self.domain, model="gemini-2.0-pro-exp-02-05")
                elif model_name == "gemini-2-flash-lite":
                    client = GeminiClient(api_key=self.gemini_api_key, role=None, mode=self.mode, domain=self.domain, model="gemini-2.0-flash-lite-preview-02-05")
                elif model_name == "mlx-qwq":
                    client = MLXClient(mode=self.mode, domain=self.domain, model="mlx-community/Meta-Llama-3.1-8B-Instruct-abliterated-8bit")
                elif model_name == "mlx-abliterated":
                    client = MLXClient(mode=self.mode, domain=self.domain, model="mlx-community/Meta-Llama-3.1-8B-Instruct-abliterated-8bit")
                elif model_name == "pico-r1-14":
                    client = PicoClient(mode=self.mode, domain=self.domain, model="DeepSeek-R1-Distill-Qwen-14B-abliterated-v2-Q4-mlx")
                elif model_name == "pico-r1-8":
                    client = PicoClient(mode=self.mode, domain=self.domain, model="DeepSeek-R1-Distill-Llama-8B-8bit-mlx")
                elif model_name == "pico-phi4":
                    client = PicoClient(mode=self.mode, domain=self.domain, model="phi-4-abliterated-3bit")
                elif model_name == "ollama":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="mannix/llama3.1-8b-lexi:latest")
                elif model_name == "ollama-phi4":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="phi4:latest")
                elif model_name == "ollama-lexi":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="mannix/llama3.1-8b-lexi:latest")
                elif model_name == "ollama-instruct":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="llama3.2:3b-instruct-q5_K_S")
                elif model_name == "ollama-qwen32-r1":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="hf.co/mili-tan/DeepSeek-R1-Distill-Qwen-32B-abliterated-Q2_K-GGUF:latest")
                elif model_name == "ollama-gemma3-1b":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="gemma3:1b-it-q8_0")
                elif model_name == "ollama-gemma3-4b":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="gemma3:4b-it-q8_0")
                elif model_name == "ollama-gemma3-12b":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="gemma3:12b-it-q4_K_M")
                elif model_name == "ollama-gemma3-27b":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="gemma3:27b-it-fp16")
                elif model_name == "ollama-llama3.2-11b":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="llama3.2-vision:11b-instruct-q4_K_M")
                elif model_name == "ollama-abliterated":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="mannix/llama3.1-8b-abliterated:latest")
                elif model_name == "ollama-zephyr":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="zephyr:latest")
                elif model_name == "ollama-r1-deepseek":
                    client = OllamaClient(mode=self.mode, domain=self.domain, model="tom_himanen/deepseek-r1-roo-cline-tools:8b")
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
        """
        Clean up clients that haven't been used recently to free up resources.

        This method removes client instances from the model map and initialized
        clients set, calling their __del__ method if available to ensure proper
        cleanup of resources. It helps manage memory usage by releasing resources
        associated with unused model clients.
        """
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
        """
        Validate that required model connections are available and working.

        This method checks if the specified models are available and properly
        initialized. If no specific models are provided, it validates all models
        in the model map except for local models like "ollama" and "mlx".

        Args:
            required_models: List of model names to validate. If None, validates
                           all models in the model map except "ollama" and "mlx".

        Returns:
            bool: True if all required connections are valid, False otherwise.
        """
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
        """
        Apply rate limiting to requests to avoid overwhelming API services.

        This method ensures that consecutive requests are separated by at least
        the minimum delay specified in self.min_delay. If a request is made
        before the minimum delay has elapsed since the last request, this method
        will sleep for the remaining time to enforce the rate limit. This helps
        prevent rate limit errors from API providers.
        """
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
                            file_data: Dict[str, Any] = None,
                            system_instruction: str=None) -> str:
        """Single conversation turn with specified model and role."""
        """
        Execute a single conversation turn with the specified model and role.

        This method handles the complexity of generating appropriate responses
        based on the conversation mode, role, and history. It supports different
        prompting strategies including meta-prompting and no-meta-prompting modes.

        Args:
            prompt: The input prompt for this turn
            model_type: Type of model to use
            client: Client instance for the model
            mode: Conversation mode (e.g., "human-aiai", "no-meta-prompting")
            role: Role for this turn ("user" or "assistant")
            file_data: Optional file data to include with the request
            system_instruction: Optional system instruction to override defaults

        Returns:
            str: Generated response text
        """
        self.mode = mode
        mapped_role = "user" if (role == "human" or role == "HUMAN" or role == "user") else "assistant"
        prompt_level = "no-meta-prompting" if mode == "no-meta-prompting" or mode =="default" else mapped_role
        if not self.conversation_history:
            self.conversation_history.append({"role": "system", "content": f"{system_instruction}!"})

        try:
            if prompt_level == "no-meta-prompting":
                response = client.generate_response(
                    prompt=prompt,
                    system_instruction=f"You are a helpful assistant. Think step by step and respond to the user. RESTRICT OUTPUTS TO APPROX {TOKENS_PER_TURN} tokens",
                    history=self.conversation_history.copy(),  # Limit history
                    role="assistant", #even if its the user role, it should get no instructions
                    file_data=file_data
                )
                if isinstance(response, list) and len(response) > 0:
                    response = response[0].text if hasattr(response[0], 'text') else str(response[0])
                self.conversation_history.append({"role": role, "content": response})
            elif (mapped_role == "user" or mapped_role == "human" or mode == "human-aiai"):
                reversed_history = []
                for msg in self.conversation_history:  # Limit history
                    if msg["role"] == "assistant":
                        reversed_history.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "user":
                        reversed_history.append({"role": "assistant", "content": msg["content"]})
                    else:
                        reversed_history.append(msg)
                if mode == "human-aiai" and role == "assistant":
                    reversed_history = self.conversation_history.copy()
                response = client.generate_response(
                    prompt=prompt,
                    system_instruction=client.adaptive_manager.generate_instructions(history = reversed_history, mode=mode, role=role,domain=self.domain),
                    history=reversed_history,  # Limit history
                    role=role,
                    file_data=file_data
                )
                if isinstance(response, list) and len(response) > 0:
                    response = response[0].text if hasattr(response[0], 'text') else str(response[0])

                self.conversation_history.append({"role": role, "content": response})
            else:

                response = client.generate_response(
                    prompt=prompt,
                    system_instruction=client.adaptive_manager.generate_instructions(history = self.conversation_history, mode=mode, role="assistant",domain=self.domain),
                    history=self.conversation_history.copy(),
                    role="assistant",
                    file_data=file_data
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

    def run_conversation_with_file(self,
                                  initial_prompt: str,
                                  human_model: str,
                                  ai_model: str,
                                  mode: str,
                                  file_config: Union[FileConfig, Dict[str, Any], 'MultiFileConfig'],
                                  human_system_instruction: str = None,
                                  ai_system_instruction: str = None,
                                  rounds: int = 1) -> List[Dict[str, str]]:
        """Run conversation with file input."""
        # Clear history and set up initial state
        self.conversation_history = []
        self.initial_prompt = initial_prompt
        self.domain = initial_prompt
        self.mode = mode

        # Process file if provided
        file_data = None
        
        # Handle MultiFileConfig object
        if hasattr(file_config, 'files') and isinstance(file_config.files, list):
            # Multiple files case using MultiFileConfig object
            files_list = file_config.files
            if not files_list:
                logger.warning("No files found in MultiFileConfig")
                return []
                
            # Process all files and create a list of file data
            file_data_list = []
            for file_config_item in files_list:
                try:
                    # Process file
                    file_metadata = self.media_handler.process_file(file_config_item.path)
                    
                    # Create file data dictionary
                    single_file_data = {
                        "type": file_metadata.type,
                        "path": file_config_item.path,
                        "mime_type": file_metadata.mime_type,
                        "dimensions": file_metadata.dimensions
                    }
                    
                    # Add type-specific data
                    if file_metadata.type == "image":
                        with open(file_config_item.path, 'rb') as f:
                            single_file_data["base64"] = base64.b64encode(f.read()).decode('utf-8')
                            single_file_data["type"] = "image"
                            single_file_data["mime_type"] = file_metadata.mime_type
                            single_file_data["path"] = file_config_item.path
                            
                    elif file_metadata.type in ["text", "code"]:
                        single_file_data["text_content"] = file_metadata.text_content
                        
                    elif file_metadata.type == "video":
                        # Handle video processing (same as single file case)
                        single_file_data["duration"] = file_metadata.duration
                        # Use the entire processed video file
                        if file_metadata.processed_video and "processed_video_path" in file_metadata.processed_video:
                            processed_video_path = file_metadata.processed_video["processed_video_path"]
                            # Set the path to the processed video file, not the original
                            single_file_data["path"] = processed_video_path
                            # Set the mime type to video/mp4 for better compatibility
                            single_file_data["mime_type"] = file_metadata.processed_video.get("mime_type", "video/mp4")
                    
                    # Add to list
                    file_data_list.append(single_file_data)
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_config_item.path}: {e}")
                    # Continue with other files
            
            # Pass the entire list of file data to the model client
            if file_data_list:
                file_data = file_data_list
        # Handle dictionary format for multiple files
        elif isinstance(file_config, dict) and "files" in file_config:
            # Multiple files case using dictionary
            files_list = file_config.get("files", [])
            if not files_list:
                logger.warning("No files found in file_config dictionary")
                return []
                
            # Process all files and create a list of file data
            file_data_list = []
            for file_config_item in files_list:
                try:
                    # Process file
                    file_metadata = self.media_handler.process_file(file_config_item.path)
                    
                    # Create file data dictionary
                    single_file_data = {
                        "type": file_metadata.type,
                        "path": file_config_item.path,
                        "mime_type": file_metadata.mime_type,
                        "dimensions": file_metadata.dimensions
                    }
                    
                    # Add type-specific data
                    if file_metadata.type == "image":
                        with open(file_config_item.path, 'rb') as f:
                            single_file_data["base64"] = base64.b64encode(f.read()).decode('utf-8')
                            single_file_data["type"] = "image"
                            single_file_data["mime_type"] = file_metadata.mime_type
                            single_file_data["path"] = file_config_item.path
                            
                    elif file_metadata.type in ["text", "code"]:
                        single_file_data["text_content"] = file_metadata.text_content
                        
                    elif file_metadata.type == "video":
                        # Handle video processing (same as single file case)
                        single_file_data["duration"] = file_metadata.duration
                        # Use the entire processed video file
                        if file_metadata.processed_video and "processed_video_path" in file_metadata.processed_video:
                            processed_video_path = file_metadata.processed_video["processed_video_path"]
                            # Set the path to the processed video file, not the original
                            single_file_data["path"] = processed_video_path
                            # Set the mime type to video/mp4 for better compatibility
                            single_file_data["mime_type"] = file_metadata.processed_video.get("mime_type", "video/mp4")
                            # Process video chunks (same as single file case)
                            # ... (video processing code)
                            
                    # Add to list
                    file_data_list.append(single_file_data)
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_config_item.path}: {e}")
                    # Continue with other files
            
            # Pass the entire list of file data to the model client
            if file_data_list:
                file_data = file_data_list
                    
        # Handle single FileConfig object
        elif file_config:
            try:
                # Process file
                file_metadata = self.media_handler.process_file(file_config.path)

                # Create file data dictionary
                file_data = {
                    "type": file_metadata.type,
                    "path": file_config.path,
                    "mime_type": file_metadata.mime_type,
                    "dimensions": file_metadata.dimensions
                }

                # Add type-specific data
                if file_metadata.type == "image":
                    with open(file_config.path, 'rb') as f:
                        file_data["base64"] = base64.b64encode(f.read()).decode('utf-8')
                        file_data["type"] = "image"
                        file_data["mime_type"] = file_metadata.mime_type
                        file_data["path"] = file_config.path
                elif file_metadata.type in ["text", "code"]:
                    file_data["text_content"] = file_metadata.text_content
                elif file_metadata.type == "video":
                    # For video, we need to extract frames
                    file_data["duration"] = file_metadata.duration
                    # Use the entire processed video file
                    if file_metadata.processed_video and "processed_video_path" in file_metadata.processed_video:
                        processed_video_path = file_metadata.processed_video["processed_video_path"]
                        # Set the path to the processed video file, not the original
                        file_data["path"] = processed_video_path
                        # Set the mime type to video/mp4 for better compatibility
                        file_data["mime_type"] = file_metadata.processed_video.get("mime_type", "video/mp4")
                        chunk_size = 1024 * 1024  # 1MB chunks
                        try:
                            with open(processed_video_path, 'rb') as f:
                                video_content = f.read()
                                # Calculate number of chunks
                                total_size = len(video_content)
                                num_chunks = (total_size + chunk_size - 1) // chunk_size

                                # Create chunks
                                chunks = []
                                for i in range(num_chunks):
                                    start = i * chunk_size
                                    end = min(start + chunk_size, total_size)
                                    chunk = video_content[start:end]
                                    chunks.append(base64.b64encode(chunk).decode('utf-8'))

                                file_data["video_chunks"] = chunks
                                file_data["num_chunks"] = num_chunks
                                file_data["video_path"] = processed_video_path
                                file_data["fps"] = file_metadata.processed_video.get("fps", 2)
                                file_data["resolution"] = file_metadata.processed_video.get("resolution", (0, 0))
                                logger.info(f"Chunked video from {processed_video_path} into {num_chunks} chunks")
                        except Exception as e:
                            logger.error(f"Error reading processed video from {processed_video_path}: {e}")

                            # Fallback to thumbnail if available
                            if file_metadata.thumbnail_path:
                                try:
                                    with open(file_metadata.thumbnail_path, 'rb') as f:
                                        file_data["key_frames"] = [{
                                            "timestamp": 0,
                                            "base64": base64.b64encode(f.read()).decode('utf-8')
                                        }]
                                        logger.info(f"Fallback: Added thumbnail as single frame")
                                except Exception as e:
                                    logger.error(f"Error reading thumbnail from {file_metadata.thumbnail_path}: {e}")

                # Add file context to prompt
                file_context = f"Analyzing {file_metadata.type} file: {file_config.path}"
                if file_metadata.dimensions:
                    file_context += f" ({file_metadata.dimensions[0]}x{file_metadata.dimensions[1]})"
                if file_metadata.type == "video" and "video_chunks" in file_data:
                    file_context += f" - FULL VIDEO CONTENT INCLUDED (in {file_data['num_chunks']} chunks)"
                    if "fps" in file_data:
                        file_context += f" at {file_data['fps']} fps"
                initial_prompt = f"{file_context}\n\n{initial_prompt}"

            except Exception as e:
                logger.error(f"Error processing file: {e}")
                return []

        # Extract core topic from initial prompt
        core_topic = initial_prompt.strip()
        try:
            if "Topic:" in initial_prompt:
                core_topic = "Discuss: " + initial_prompt.split("Topic:")[1].split("\\n")[0].strip()
            elif "GOAL:" in initial_prompt:
                # Try to extract goal with more robust parsing
                goal_parts = initial_prompt.split("GOAL:")[1].strip()
                if "(" in goal_parts and ")" in goal_parts:
                    # Extract content between parentheses if present
                    try:
                        core_topic = "GOAL: " + goal_parts.split("(")[1].split(")")[0].strip()
                    except IndexError:
                        # If extraction fails, use the whole goal part
                        core_topic = "GOAL: " + goal_parts
                else:
                    # Just use what comes after "GOAL:"
                    core_topic = "GOAL: " + goal_parts.split("\n")[0].strip()
        except (IndexError, Exception) as e:
            # If parsing fails, use the full prompt
            logger.warning(f"Failed to extract core topic from prompt: {e}")
            core_topic = initial_prompt.strip()

        self.conversation_history.append({"role": "system", "content": f"{core_topic}"})

        # Continue with standard conversation flow, but pass file_data to the first turn
        return self._run_conversation_with_file_data(core_topic, human_model, ai_model, mode, file_data, human_system_instruction, ai_system_instruction, rounds)

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
        try:
            if "Topic:" in initial_prompt:
                core_topic = "Discuss: " + initial_prompt.split("Topic:")[1].split("\\n")[0].strip()
            elif "GOAL:" in initial_prompt:
                # Try to extract goal with more robust parsing
                goal_parts = initial_prompt.split("GOAL:")[1].strip()
                if "(" in goal_parts and ")" in goal_parts:
                    # Extract content between parentheses if present
                    try:
                        core_topic = "GOAL: " + goal_parts.split("(")[1].split(")")[0].strip()
                    except IndexError:
                        # If extraction fails, use the whole goal part
                        core_topic = "GOAL: " + goal_parts
                else:
                    # Just use what comes after "GOAL:"
                    core_topic = "GOAL: " + goal_parts.split("\n")[0].strip()
        except (IndexError, Exception) as e:
            # If parsing fails, use the full prompt
            logger.warning(f"Failed to extract core topic from prompt: {e}")
            core_topic = initial_prompt.strip()

        self.conversation_history.append({"role": "system", "content": f"{core_topic}"})

        logger.info(f"Starting conversation with topic: {core_topic}")

        # Get client instances
        human_client = self._get_client(human_model)
        ai_client = self._get_client(ai_model)

        if not human_client or not ai_client:
            logger.error(f"Could not initialize required clients: {human_model}, {ai_model}")
            return []

        return self._run_conversation_with_file_data(core_topic, human_model, ai_model, mode, None, human_system_instruction, ai_system_instruction, rounds)

    def _run_conversation_with_file_data(self,
                                       core_topic: str,
                                       human_model: str,
                                       ai_model: str,
                                       mode: str,
                                       file_data: Dict[str, Any] = None,
                                       human_system_instruction: str = None,
                                       ai_system_instruction: str = None,
                                       rounds: int = 5) -> List[Dict[str, str]]:
        """Internal method to run conversation with optional file data."""
        logger.info(f"Starting conversation with topic: {core_topic}")
        self.mode="human-ai"
        # Get client instances
        human_client = self._get_client(human_model)
        ai_client = self._get_client(ai_model)

        if not human_client or not ai_client:
            logger.error(f"Could not initialize required clients: {human_model}, {ai_model}")
            return []

        # Check if models support vision if file is image/video
        if file_data and file_data["type"] in ["image", "video"]:
            human_capabilities = detect_model_capabilities(human_model)
            ai_capabilities = detect_model_capabilities(ai_model)

            if not human_capabilities.get("vision", False) or not ai_capabilities.get("vision", False):
                logger.warning("One or both models do not support vision capabilities")
                # We'll continue but log a warning

                # If AI model doesn't support vision, we'll convert image to text description
                if not ai_capabilities.get("vision", False) and file_data["type"] == "image":
                    # Add a note that this is an image description
                    dimensions = file_data.get("dimensions", (0, 0))
                    file_data = {
                        "type": "text",
                        "text_content": f"[This is an image with dimensions {dimensions[0]}x{dimensions[1]}]",
                        "path": file_data.get("path", "")
                    }

        ai_response = core_topic
        try:
            # Run conversation rounds
            for round_index in range(rounds):
                # Human turn
                human_response = self.run_conversation_turn(
                    prompt=ai_response,  # Limit history
                    system_instruction=f"{core_topic}. Think step by step. RESTRICT OUTPUTS TO APPROX {TOKENS_PER_TURN} tokens" if mode == "no-meta-prompting" else human_client.adaptive_manager.generate_instructions(mode=mode, role="user",history=self.conversation_history,domain=self.domain),
                    role="user",
                    mode=self.mode,
                    model_type=human_model,
                    file_data=file_data,  # Only pass file data on first turn
                    client=human_client
                )
                #print(f"\n\n\nHUMAN: ({human_model.upper()}): {human_response}\n\n")

                # AI turn
                ai_response = self.run_conversation_turn(
                    prompt=human_response,
                    system_instruction=f"{core_topic}. You are a helpful AI. Think step by step. RESTRICT OUTPUTS TO APPROX {TOKENS_PER_TURN} tokens" if mode == "no-meta-prompting" else human_client.adaptive_manager.generate_instructions(mode=mode, role="assistant",history=self.conversation_history,domain=self.domain) if mode=="human-aiai" else ai_system_instruction,
                    role="assistant",
                    mode=self.mode,
                    model_type=ai_model,
                    file_data=file_data,
                    client=ai_client
                )
                logger.debug(f"\n\n\nMODEL RESPONSE: ({ai_model.upper()}): {ai_response}\n\n\n")

            # Clean up unused clients
            #self.cleanup_unused_clients()

            return self.conversation_history

        finally:
            # Ensure cleanup happens even if there's an error
            self.cleanup_unused_clients()
            MemoryManager.cleanup_all()

    @classmethod
    def from_config(cls, config_path: str) -> 'ConversationManager':
        """Create ConversationManager instance from configuration file."""
        config = load_config(config_path)

        # Initialize manager with config
        manager = cls(
            config=config,
            domain=config.goal,
            mode="human-ai"  # Default mode
        )

        # Set up models based on configuration
        for model_id, model_config in config.models.items():
            # Detect model capabilities
            capabilities = detect_model_capabilities(model_config.type)

            # Initialize appropriate client
            client = manager._get_client(model_config.type)
            if client:
                # Store client in model map with configured role
                client.role = model_config.role
                manager.model_map[model_id] = client
                manager._initialized_clients.add(model_id)

        return manager

async def save_conversation(conversation: List[Dict[str, str]],
                     filename: str,
                     human_model: str,
                     ai_model: str,
                     file_data: Dict[str, Any] = None,
                     mode: str = None) -> None:
    """Save an AI conversation to an HTML file with proper encoding.

    Args:
        conversation (List[Dict[str, str]]): List of conversation messages with 'role' and 'content'
        filename (str): Output HTML file path
        human_model (str): Name of the human/user model
        ai_model (str): Name of the AI model
        file_data (Dict[str, Any], optional): Any associated file content (images, video, text)
        mode (str, optional): Conversation mode ('human-ai' or 'ai-ai')

    Raises:
        Exception: If saving fails or template is missing
    """
    try:
        with open("templates/conversation.html", "r") as f:
            template = f.read()

        conversation_html = ""

        # Add file content if present
        if file_data:
            # Handle multiple files (list of file data)
            if isinstance(file_data, list):
                for idx, file_item in enumerate(file_data):
                    if isinstance(file_item, dict) and "type" in file_item:
                        if file_item["type"] == "image" and "base64" in file_item:
                            # Add image to the conversation
                            mime_type = file_item.get("mime_type", "image/jpeg")
                            conversation_html += f'<div class="file-content"><h3>File {idx+1}: {file_item.get("path", "Image")}</h3>'
                            conversation_html += f'<img src="data:{mime_type};base64,{file_item["base64"]}" alt="Input image" style="max-width: 100%; max-height: 500px;"/></div>\n'
                        elif file_item["type"] == "video" and "key_frames" in file_item and file_item["key_frames"]:
                            # Add first frame of video
                            frame = file_item["key_frames"][0]
                            conversation_html += f'<div class="file-content"><h3>File {idx+1}: {file_item.get("path", "Video")} (First Frame)</h3>'
                            conversation_html += f'<img src="data:image/jpeg;base64,{frame["base64"]}" alt="Video frame" style="max-width: 100%; max-height: 500px;"/></div>\n'
                        elif file_item["type"] in ["text", "code"] and "text_content" in file_item:
                            # Add text content
                            conversation_html += f'<div class="file-content"><h3>File {idx+1}: {file_item.get("path", "Text")}</h3><pre>{file_item["text_content"]}</pre></div>\n'
            # Handle single file (original implementation)
            elif isinstance(file_data, dict) and "type" in file_data:
                if file_data["type"] == "image" and "base64" in file_data:
                    # Add image to the conversation
                    mime_type = file_data.get("mime_type", "image/jpeg")
                    conversation_html += f'<div class="file-content"><h3>File: {file_data.get("path", "Image")}</h3>'
                    conversation_html += f'<img src="data:{mime_type};base64,{file_data["base64"]}" alt="Input image" style="max-width: 100%; max-height: 500px;"/></div>\n'
                elif file_data["type"] == "video" and "key_frames" in file_data and file_data["key_frames"]:
                    # Add first frame of video
                    frame = file_data["key_frames"][0]
                    conversation_html += f'<div class="file-content"><h3>File: {file_data.get("path", "Video")} (First Frame)</h3>'
                    conversation_html += f'<img src="data:image/jpeg;base64,{frame["base64"]}" alt="Video frame" style="max-width: 100%; max-height: 500px;"/></div>\n'
                elif file_data["type"] in ["text", "code"] and "text_content" in file_data:
                    # Add text content
                    conversation_html += f'<div class="file-content"><h3>File: {file_data.get("path", "Text")}</h3><pre>{file_data["text_content"]}</pre></div>\n'

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

            # Check if message contains file content (for multimodal messages)
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        # Extract image data
                        image_data = item.get("image_url", {}).get("url", "")
                        if image_data.startswith("data:"):
                            conversation_html += f'<div class="message-image"><img src="{image_data}" alt="Image in message" style="max-width: 100%; max-height: 300px;"/></div>\n'

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
    rounds = 5
    initial_prompt = """"
    GOAL: Write a short story about a detective solving a mystery.
"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    claude_api_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    mode = "ai-ai"
    ai_model = "gemini-2-pro"
    human_model = "haiku"

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


    human_system_instruction = f"You are a HUMAN expert curious to explore {initial_prompt}... Restrict output to {TOKENS_PER_TURN} tokens"  # Truncated for brevity
    if "GOAL:" in initial_prompt:
        human_system_instruction = f"Solve {initial_prompt} together..."  # Truncated for brevity

    ai_system_instruction = f"You are a helpful assistant. Think step by step and respond to the user. Restrict your output to {TOKENS_PER_TURN} tokens"  # Truncated for brevity
    if mode == "ai-ai" or mode == "aiai":
        ai_system_instruction = human_system_instruction

    try:
        # Run default conversation
        mode="ai-ai"
        # Run AI-AI conversation
        conversation = manager.run_conversation(
            initial_prompt=initial_prompt,
            mode=mode,
            human_model=human_model,
            ai_model=ai_model,
            human_system_instruction=human_model,
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
            ai_system_instruction=human_system_instruction,
            rounds=rounds
        )

        safe_prompt = _sanitize_filename_part(initial_prompt[:20] + "_" + human_model + "_" + ai_model)
        time_stamp = datetime.datetime.now().strftime("%m%d%H%M")
        filename = f"conv-humai_{safe_prompt}_{time_stamp}.html"
        await save_conversation(conversation=conversation_as_human_ai, filename=f"{filename}", human_model=human_model, ai_model=ai_model, mode="human-ai")

        mode = "no-meta-prompting"
        conv_default = manager.run_conversation(
            initial_prompt=initial_prompt,
            mode=mode,
            human_model=human_model,
            ai_model=ai_model,
            human_system_instruction=ai_system_instruction,
            ai_system_instruction=ai_system_instruction,
            rounds=rounds
        )

        safe_prompt = _sanitize_filename_part(initial_prompt[:16] + "_" + human_model + "_" + ai_model)
        time_stamp = datetime.datetime.now().strftime("%m%d%H%M")
        filename = f"conv-defaults_{safe_prompt}_{time_stamp}.html"
        await save_conversation(conversation=conv_default, filename=f"{filename}", human_model=human_model, ai_model=ai_model, mode="human-ai")

        # Run analysis
        arbiter_report = evaluate_conversations(
            ai_ai_convo=conversation,
            human_ai_convo=conversation_as_human_ai,
            default_convo=conv_default,
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
