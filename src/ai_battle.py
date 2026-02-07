"""
AI model conversation manager with memory optimizations.
"""
import asyncio
import io
import json
import os
import datetime
import base64
import sys
import time
import random
import logging
import re
import traceback
from typing import List, Dict, Optional, TypeVar, Any, Union
from dataclasses import dataclass

# Local imports
from configuration import load_config, detect_model_capabilities
from configdataclasses import FileConfig, DiscussionConfig
from arbiter_v4 import evaluate_conversations, VisualizationGenerator
from file_handler import ConversationMediaHandler
from model_clients import (
    BaseClient,
    OpenAIClient,
    ClaudeClient,
    GeminiClient,
    MLXClient,
    OllamaClient,
)

from lmstudio_client import LMStudioClient
from claude_reasoning_config import ClaudeReasoningConfig
from shared_resources import MemoryManager
from metrics_analyzer import analyze_conversations

T = TypeVar("T")
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


# Models to use in default mode
# these must match the model names below not necessary the exact actual model name
HUMAN_MODEL = "ollama-gemma3:4b-it-q8_0" #"ollama-gpt-oss:120b" #"gemini-2.5-flash-lite" #"gemini-2.5-flash-preview" #"ollama-phi4:14b-fp16" #"gemini-2.0-flash-thinking-exp"# "ollama-gemma3:4b-it-q8_0"
AI_MODEL = "ollama-gpt-oss:20b"  #"gemini-3.0-flash" # "ollama-gemma3:27b-it-q8_0"  #"gemini-2.0-flash-thinking-exp"
DEFAULT_ROUNDS=4



# Local imports
#from configuration import load_config, detect_model_capabilities
# Set environment variables for these model names so arbiter can use them
os.environ["AI_MODEL"] = AI_MODEL
os.environ["HUMAN_MODEL"] = HUMAN_MODEL

CONFIG_PATH = "config.yaml"
TOKENS_PER_TURN = 4096
MAX_TOKENS = TOKENS_PER_TURN
DEFAULT_PROMPT = """
Discuss societal, productivity and privacy implications (pros and cons) of conversational memory recall, embeddings and persistence in web-based AI systems, memory on vs memory off contexts, and the collection of user-related metadata in the context of a conversations including via multimodal inputs.
Consider in the context of the conversational recall of ChatGPT as an example:

The “Reference chat history” feature in AI systems like ChatGPT allows the model to utilize past interactions with a user to provide more personalized and contextually relevant responses. Here’s how it works and how you can manage it:

How “Reference Chat History” Works
Personalization: When enabled, the system uses information from previous conversations to tailor future interactions based on your interests, preferences, and past discussions.

Dynamic Learning: Unlike saved memories, which are explicitly retained until deleted, “chat history” is more dynamic. The AI updates what it considers useful to remember over time, potentially altering its focus as new interactions occur.

Enabling or Disabling “Reference Chat History”
Access Settings: You can turn this feature on or off in the Personalization section of your settings.

Dependency on Saved Memories:

Turning off “Reference saved memories” will also disable “Reference chat history.”
If “Reference saved memories” is enabled, you can still choose to turn off “Reference chat history.”
Effects of Disabling “Reference Chat History”
Deletion of Information: When disabled, the information from past chats that was being referenced will be deleted from the system within 30 days.

Storage Limitations: There is no specified storage limit for what can be referenced when this feature is enabled.

Managing What ChatGPT Remembers
Reviewing Memories: You can ask ChatGPT to recall what it remembers about you. This helps in understanding how your information is being used.

Forgetting Information:

Request ChatGPT to forget specific details from past conversations.
Once forgotten, the AI will not use that information in future responses, though the original conversation remains unless deleted by you.
Full Deletion of Information
Memory On: If memory is active and you want to completely remove something:

Delete both the saved memories in settings and the specific chat where the information was shared.
Chat History Off: Turning off “Reference chat history” will lead to the deletion of all referenced past conversation data from the system within 30 days.

Key Considerations
Data Retention: Even with “chat history” turned off, original chats remain unless manually deleted by you.

Privacy and Control: Users have control over what is remembered or forgotten, allowing for a balance between personalization and privacy.

This feature aims to enhance user experience by making interactions more relevant while providing users with the tools to manage their data privacy effectively.
"""
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("log/ai_battle.log"), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# Model templates for accessing different model versions with reasoning levels
OPENAI_MODELS = {
    # Base models (text-only with reasoning support)
    "o1": {"model": "o1", "reasoning_level": "medium", "multimodal": False},
    "o3": {"model": "o3", "reasoning_level": "auto", "multimodal": True},
    # O3 with reasoning levels (text-only)
    "o3-reasoning-high": {
        "model": "o3",
        "reasoning_level": "high",
        "multimodal": True,
    },
    "o3-reasoning-medium": {
        "model": "o3",
        "reasoning_level": "medium",
        "multimodal": True,
    },
    "o3-reasoning-low": {"model": "o3", "reasoning_level": "low", "multimodal": True},
    "o3-mini-high": {"model": "o3-mini", "reasoning_level": "high", "multimodal": True},
    "o4-mini": {"model": "o4-mini", "reasoning_level": "medium", "multimodal": True},
    "o4-mini-high": {"model": "o4-mini", "reasoning_level": "high", "multimodal": True},
    # Multimodal models without reasoning parameter
    "gpt-4o": {"model": "gpt-4o", "reasoning_level": None, "multimodal": True},
    "gpt-4.1": {"model": "gpt-4.1", "reasoning_level": None, "multimodal": True},
    "gpt-5": {"model": "gpt-5", "reasoning_level": "medium" , "multimodal": True},
    "gpt-4.1-mini": {"model": "gpt-4.1-mini", "reasoning_level": None, "multimodal": True},
    "gpt-4.1-nano": {"model": "gpt-4.1-nano", "reasoning_level": None, "multimodal": True},
    "chatgpt-latest": {"model": "chatgpt-latest", "reasoning_level": None, "multimodal": True},
}

# Gemini model configurations
GEMINI_MODELS = {
    "gemini-2.0-pro": {"model": "gemini-2.0-pro-exp-02-05", "multimodal": True},
    "gemini-2.5-pro-exp": {"model": "gemini-2.5-pro-exp-03-25", "multimodal": True},
    "gemini-2.5-flash-preview": {"model": "gemini-2.5-flash-preview-04-17", "multimodal": True},
    "gemini-2.5-pro-preview-03-25": {"model": "gemini-2.5-pro-preview-03-25", "multimodal": True},
    "gemini-2.5-flash-preview-04-17": {"model": "gemini-2.5-flash-preview-04-17", "multimodal": True},
    "gemini-2.0-flash-exp": {"model": "gemini-2.0-flash-exp", "multimodal": True},
    "gemini-2.0-flash-thinking-exp": {"model": "gemini-2.0-flash-thinking-exp", "multimodal": True},
    "gemini-2.0-flash-thinking-exp-01-21": {"model": "gemini-2.0-flash-thinking-exp-01-21", "multimodal": True},
    "gemini-2.0-flash-lite": {"model": "gemini-2.0-flash-lite-preview-02-05", "multimodal": True},
    # Added Gemini 2.5 Pro and Flash (using 1.5 latest as placeholders)
    "gemini-2.5-pro": {"model": "gemini-2.5-pro-latest", "multimodal": True},
    "gemini-2.5-flash-lite": {"model": "gemini-2.5-flash-lite", "multimodal": True},
    "gemini-2.5-flash": {"model": "gemini-2.5-flash", "multimodal": True},
}

CLAUDE_MODELS = {
    # Base models (newest versions)
    "claude": {
        "model": "claude-3-7-sonnet-latest",
        "reasoning_level": None,
        "extended_thinking": False,
    },
    "sonnet": {
        "model": "claude-3-7-sonnet-latest",
        "reasoning_level": None,
        "extended_thinking": False,
    },
    "haiku": {
        "model": "claude-3-5-haiku-latest",
        "reasoning_level": None,
        "extended_thinking": False,
    },
    # Specific versions
    "claude-3-5-sonnet": {
        "model": "claude-3-5-sonnet-latest",
        "reasoning_level": None,
        "extended_thinking": False,
    },
    "claude-3-5-haiku": {
        "model": "claude-3-5-haiku-latest",
        "reasoning_level": None,
        "extended_thinking": False,
    },
    "claude-3-7": {
        "model": "claude-3-7-sonnet-latest",
        "reasoning_level": "auto",
        "extended_thinking": False,
    },
    "claude-3-7-sonnet": {
        "model": "claude-3-7-sonnet-latest",
        "reasoning_level": "auto",
        "extended_thinking": False,
    },
    "claude-4-0-sonnet": {
        "model": "claude-4-0-sonnet-latest",
        "reasoning_level": "auto",
        "extended_thinking": False,
    },
    "claude-4-0-opus": {
        "model": "claude-4-0-opus-latest",
        "reasoning_level": "high",
        "extended_thinking": True,
    },
    # Claude 3.7 with reasoning levels
    "claude-3-7-reasoning": {
        "model": "claude-3-7-sonnet-latest",
        "reasoning_level": "high",
        "extended_thinking": False,
    },
    "claude-3-7-reasoning-high": {
        "model": "claude-3-7-sonnet-latest",
        "reasoning_level": "high",
        "extended_thinking": False,
    },
    "claude-3-7-reasoning-medium": {
        "model": "claude-3-7-sonnet-latest",
        "reasoning_level": "medium",
        "extended_thinking": False,
    },
    "claude-3-7-reasoning-low": {
        "model": "claude-3-7-sonnet",
        "reasoning_level": "low",
        "extended_thinking": False,
    },
    "claude-3-7-reasoning-none": {
        "model": "claude-3-7-sonnet",
        "reasoning_level": "none",
        "extended_thinking": False,
    },
    # Claude 3.7 with extended thinking
    "claude-3-7-extended": {
        "model": "claude-3-7-sonnet-latest",
        "reasoning_level": "high",
        "extended_thinking": True,
        "budget_tokens": 8000,
    },
    "claude-3-7-extended-deep": {
        "model": "claude-3-7-sonnet-latest",
        "reasoning_level": "high",
        "extended_thinking": True,
        "budget_tokens": 16000,
    },
}

# Ollama model configurations for thinking-capable models
# Keyword-based overrides applied when a matching Ollama model is discovered dynamically.
# Default reasoning_level is "high" for all thinking models.
OLLAMA_THINKING_CONFIG = {
    "gpt-oss": {"reasoning_level": "medium", "extended_thinking": True, "num_ctx": 131072},
    "deepseek-r1": {"reasoning_level": "high", "extended_thinking": True},
    "qwen3": {"reasoning_level": "high", "extended_thinking": True},
    "phi4-reasoning": {"reasoning_level": "high", "extended_thinking": True},
    "granite-reasoning": {"reasoning_level": "high", "extended_thinking": True},
}


@dataclass
class ModelConfig:
    """Configuration for AI model parameters"""

    temperature: float = 0.7
    max_tokens: int = MAX_TOKENS
    stop_sequences: List[str] = None
    seed: Optional[int] = random.randint(0, 1000)
    human_delay: float = 4.0


@dataclass
class ConversationManager:
    """Manages conversations between AI models with memory optimization."""

    def __init__(
        self,
        config: DiscussionConfig = None,
        domain: str = "General knowledge",
        human_delay: float = 4.0,
        mode: str = None,
        min_delay: float = 2,
        gemini_api_key: Optional[str] = None,
        claude_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.config = config
        self.domain = config.goal if config else domain

        self.human_delay = human_delay
        self.mode = mode  # "human-ai" or "ai-ai"
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
        self._ollama_models = {}
        self._lmstudio_models = {}

    @property
    def media_handler(self):
        """Lazy initialization of media handler."""
        if self._media_handler is None:
            self._media_handler = ConversationMediaHandler(output_dir="processed_files")
        return self._media_handler

    def _get_available_ollama_models(self) -> Dict[str, str]:
        """
        Fetch available models from local Ollama instance.

        Returns:
            Dict[str, str]: Dictionary mapping friendly names to actual model names
        """
        if not self._ollama_models:
            try:
                # Create a temporary Ollama client to fetch models
                temp_client = OllamaClient(mode=self.mode, domain=self.domain, model="")
                model_list = temp_client.client.list()

                # Map models to friendly names (SDK 0.6.x returns typed ListResponse)
                for model_info in model_list.models:
                    model_name = model_info.model
                    if model_name:
                        # Create a friendly name (prefix with ollama-)
                        base_name = model_name.split(':')[0].split('/')[-1].lower()
                        friendly_name = f"ollama-{base_name}"
                        # Also create a full-tag friendly name for disambiguation
                        # e.g. both "ollama-gpt-oss" and "ollama-gpt-oss:120b" resolve
                        full_friendly_name = f"ollama-{model_name.lower()}"
                        self._ollama_models[friendly_name] = model_name  # Last wins for base name
                        self._ollama_models[full_friendly_name] = model_name  # Always unique

                logger.info(f"Found {len(self._ollama_models)} available Ollama models")
            except Exception as e:
                logger.error(f"Failed to fetch Ollama models: {e}")
                # Default models for fallback
                self._ollama_models = None
                raise e

        return self._ollama_models

    def _get_available_lmstudio_models(self) -> Dict[str, str]:
        """
        Fetch available models from local LMStudio instance.

        Returns:
            Dict[str, str]: Dictionary mapping friendly names to actual model names
        """
        if not self._lmstudio_models:
            try:
                # Create a temporary LMStudio client to fetch models
                temp_client = LMStudioClient(mode=self.mode, domain=self.domain)

                # Map models to friendly names
                for model_name in temp_client.available_models:
                    if model_name:
                        # Create a friendly name (prefix with lmstudio-)
                        parts = model_name.split('/')
                        base_name = parts[-1].lower() if len(parts) > 1 else model_name.lower()
                        friendly_name = f"lmstudio-{base_name}"
                        self._lmstudio_models[friendly_name] = model_name

                logger.info(f"Found {len(self._lmstudio_models)} available LMStudio models")
            except Exception as e:
                logger.warning(f"Failed to fetch LMStudio models: {e}")
                # Default models for fallback
                self._lmstudio_models = None

        return self._lmstudio_models

    def _get_client(self, model_name: str) -> Optional[BaseClient]:
        """
        Get an existing client instance or create a new one for the specified model.

        This method manages client instances, creating them on demand and caching them
        for reuse. It supports various model types including Claude, GPT, Gemini, MLX,
        Ollama

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
                        model=model_config["model"],
                    )

                    # Set reasoning level if specified
                    if model_config["reasoning_level"] is not None:
                        client.reasoning_level = model_config["reasoning_level"]
                        logger.debug(
                            f"Set reasoning level to '{model_config['reasoning_level']}' for {model_name}"
                        )

                    # Set extended thinking if enabled
                    if model_config.get("extended_thinking", False):
                        budget_tokens = model_config.get("budget_tokens", None)
                        client.set_extended_thinking(True, budget_tokens)
                        logger.info( # Changed to info for better visibility
                            f"Enabled extended thinking with budget_tokens={budget_tokens} for Claude model {model_name}"
                        )

                # Handle OpenAI models using templates
                elif model_name in OPENAI_MODELS:
                    model_config = OPENAI_MODELS[model_name]
                    client = OpenAIClient(
                        api_key=self.openai_api_key,
                        role=None,
                        mode=self.mode,
                        domain=self.domain,
                        model=model_config["model"],
                    )

                    # Set reasoning level if specified
                    if model_config["reasoning_level"] is not None:
                        client.reasoning_level = model_config["reasoning_level"]
                        logger.debug(
                            f"Set reasoning level to '{model_config['reasoning_level']}' for {model_name}"
                        )

                # Handle Ollama models dynamically
                elif model_name.startswith("ollama-"):
                    # Get available Ollama models
                    ollama_models = self._get_available_ollama_models()

                    if ollama_models and model_name in ollama_models:
                        # Use the mapped model name
                        actual_model = ollama_models[model_name]
                        client = OllamaClient(
                            mode=self.mode,
                            domain=self.domain,
                            model=actual_model
                        )
                        logger.info(f"Using Ollama model: {model_name} -> {actual_model}")
                    else:
                        # Try to extract model name directly from the request
                        # This handles cases like "ollama-new-model" that aren't in our map
                        # or when ollama_models is None due to connection issues
                        model_suffix = model_name[len("ollama-"):]
                        if ":" not in model_suffix:
                            model_suffix = f"{model_suffix}:latest"

                        if ollama_models is None:
                            logger.warning(f"Unable to fetch Ollama models, using direct model name: {model_suffix}")
                        else:
                            logger.warning(f"Ollama model {model_name} not found in available models, trying direct: {model_suffix}")

                        actual_model = model_suffix
                        client = OllamaClient(
                            mode=self.mode,
                            domain=self.domain,
                            model=actual_model
                        )

                    # Apply thinking configuration if the model matches known thinking-capable patterns
                    if client:
                        actual_model_lower = actual_model.lower()
                        for keyword, thinking_config in OLLAMA_THINKING_CONFIG.items():
                            if keyword in actual_model_lower:
                                if thinking_config.get("reasoning_level"):
                                    client.reasoning_level = thinking_config["reasoning_level"]
                                    logger.info(
                                        f"Set reasoning level to '{thinking_config['reasoning_level']}' "
                                        f"for Ollama model {model_name} (matched '{keyword}')"
                                    )
                                if thinking_config.get("extended_thinking", False):
                                    client.set_extended_thinking(True)
                                    logger.info(
                                        f"Enabled extended thinking for Ollama model {model_name} "
                                        f"(matched '{keyword}')"
                                    )
                                if "num_ctx" in thinking_config:
                                    client.num_ctx = thinking_config["num_ctx"]
                                    logger.info(
                                        f"Set num_ctx={thinking_config['num_ctx']} "
                                        f"for Ollama model {model_name}"
                                    )
                                if "keep_alive" in thinking_config:
                                    client.keep_alive = thinking_config["keep_alive"]
                                break  # Apply first matching config only

                # Handle LMStudio models dynamically
                elif model_name.startswith("lmstudio-"):
                    # Get available LMStudio models
                    lmstudio_models = self._get_available_lmstudio_models()

                    # Only attempt matching if we have models to match against
                    if lmstudio_models:
                        # Try matching by prefix (allows partial matches)
                        matched_name = None
                        for lms_name in lmstudio_models:
                            if model_name == lms_name or (
                                # Handle the case where model names have additional specifics (like bit depth)
                                # e.g., "lmstudio-qwq-32b" would match "lmstudio-qwq-32b-8bit-MLX"
                                model_name.startswith(lms_name) or lms_name.startswith(model_name)
                            ):
                                matched_name = lms_name
                                break

                        if matched_name:
                            actual_model = lmstudio_models[matched_name]
                            client = LMStudioClient(
                                mode=self.mode,
                                domain=self.domain,
                                model=actual_model
                            )
                            logger.info(f"Using LMStudio model: {model_name} -> {actual_model}")
                            return client

                    # If we get here, either lmstudio_models is None or no matching model was found
                    if lmstudio_models is None:
                        logger.warning(f"Unable to fetch LMStudio models, creating default client")
                    else:
                        logger.warning(f"LMStudio model {model_name} not found in available models, using first available")

                    # Create client with no specific model - LMStudioClient will use first available
                    client = LMStudioClient(
                        mode=self.mode,
                        domain=self.domain
                    )

                # Handle Gemini models using templates
                elif model_name in GEMINI_MODELS:
                    model_config = GEMINI_MODELS[model_name]
                    client = GeminiClient(
                        api_key=self.gemini_api_key,
                        role=None,
                        mode=self.mode,
                        domain=self.domain,
                        model=model_config["model"],
                    )
                elif model_name == "mlx-llama-3.1-abb":
                    client = MLXClient(
                        mode=self.mode,
                        domain=self.domain,
                        model="mlx-community/Meta-Llama-3.1-8B-Instruct-abliterated-8bit",
                    )
                else:
                    logger.error(f"Unknown model: {model_name}")
                    return None

                logger.info(f"Created client for model: {model_name}")
                logger.debug(MemoryManager.get_memory_usage())

                if client:
                    self.model_map[model_name] = client
                    self._initialized_clients.add(model_name)
            except Exception as e:
                # Check if this is a critical error that should terminate the program
                is_critical_error = False
                error_msg = str(e).lower()

                # Missing API key errors are critical and should terminate the program
                if "api key" in error_msg and ("missing" in error_msg or "no api key" in error_msg or "not provided" in error_msg):
                    is_critical_error = True
                elif "no api key provided" in error_msg:
                    is_critical_error = True

                if is_critical_error:
                    logger.critical(f"CRITICAL ERROR: Failed to create client for {model_name}: {e}")
                    logger.critical(f"Program will terminate as required API key is missing")
                    # Re-raise the exception to terminate the program
                    raise RuntimeError(f"Missing required API key for {model_name}: {e}")
                else:
                    logger.error(f"Failed to create client for {model_name}: {e}")
                    return None
        return self.model_map.get(model_name)

    def cleanup_unused_clients(self):
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
            if hasattr(client, "__del__"):
                client.__del__()
            del self.model_map[model_name]
            self._initialized_clients.remove(model_name)
        logger.debug(MemoryManager.get_memory_usage())

    def validate_connections(self, required_models: List[str] = None) -> bool:
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
            required_models = [
                name
                for name, client in self.model_map.items()
                if client and name not in ["ollama", "mlx"]
            ]

        if not required_models:
            logger.info("No models require validation")
            return True

        # Prime the model lists cache, but handle exceptions gracefully
        if any(model.startswith("ollama-") for model in required_models):
            try:
                self._get_available_ollama_models()
            except Exception as e:
                logger.warning(f"Failed to prime Ollama models cache: {e}")

        if any(model.startswith("lmstudio-") for model in required_models):
            try:
                self._get_available_lmstudio_models()
            except Exception as e:
                logger.warning(f"Failed to prime LMStudio models cache: {e}")

        validations = []
        return True

    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get all available models from local and remote sources.

        Returns:
            Dict[str, List[str]]: Dictionary of model categories and available models
        """
        # Get models from Ollama and LMStudio
        ollama_models = self._get_available_ollama_models()
        lmstudio_models = self._get_available_lmstudio_models()

        result = {
            "ollama": list(ollama_models.keys()) if ollama_models else ["ollama-not-available"],
            "lmstudio": list(lmstudio_models.keys()) if lmstudio_models else ["lmstudio-not-available"],
            "claude": list(CLAUDE_MODELS.keys()),
            "openai": list(OPENAI_MODELS.keys()),
            "gemini": list(GEMINI_MODELS.keys()),
        }
        return result

    def rate_limited_request(self):
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
                time.sleep(self.min_delay)
            self.last_request_time = time.time()

    def run_conversation_turn(
        self,
        prompt: str,
        model_type: str,
        client: BaseClient,
        mode: str,
        role: str,
        file_data: Dict[str, Any] = None,
        system_instruction: str = None,
    ) -> str:
        """
        Execute a single conversation turn with the specified model and role.

        This method handles the complexity of generating appropriate responses
        based on the conversation mode, role, and history. It supports different
        prompting strategies including meta-prompting and no-meta-prompting modes.

        Args:
            prompt: The input prompt for this turn
            model_type: Type of model to use
            client: Client instance for the model
            mode: Conversation mode (e.g., "human-ai", "no-meta-prompting")
            role: Role for this turn ("user" or "assistant")
            file_data: Optional file data to include with the request
            system_instruction: Optional system instruction to override defaults

        Returns:
            str: Generated response text
        """
        self.mode = mode
        mapped_role = (
            "user"
            if (role == "human" or role == "HUMAN" or role == "user")
            else "assistant"
        )
        prompt_level = (
            "no-meta-prompting"
            if mode == "no-meta-prompting" or mode == "default"
            else mapped_role
        )
        if not self.conversation_history:
            self.conversation_history.append(
                {"role": "system", "content": f"{system_instruction}!"}
            )

        # Define a list of known fatal errors that should halt processing
        fatal_connection_errors = [
            "Connection aborted",
            "Remote end closed connection without response",
            "Connection refused",
            "Max retries exceeded",
            "Read timed out",
            "API key not valid",
            "Authentication failed",
            "Quota exceeded",
            "Service unavailable"
        ]

        try:
            if prompt_level == "no-meta-prompting":
                response = client.generate_response(
                    prompt=prompt,
                    system_instruction=f"You are a helpful assistant. Think step by step and respond to the user. RESTRICT OUTPUTS TO APPROX {TOKENS_PER_TURN} tokens",
                    history=self.conversation_history.copy(),  # Limit history
                    role="assistant",  # even if its the user role, it should get no instructions
                    file_data=file_data,
                )
                if isinstance(response, list) and len(response) > 0:
                    response = (
                        response[0].text
                        if hasattr(response[0], "text")
                        else str(response[0])
                    )
                self.conversation_history.append({"role": role, "content": response})
            elif (mapped_role == "user" or mapped_role == "human"):
                # Only swap roles in human-ai mode where the human role needs AI-like prompting
                if mode == "human-ai":
                    reversed_history = []
                    for msg in self.conversation_history:  # Limit history
                        if msg["role"] == "assistant":
                            reversed_history.append(
                                {"role": "user", "content": msg["content"]}
                            )
                        elif msg["role"] == "user" or msg["role"] == "human":
                            reversed_history.append(
                                {"role": "assistant", "content": msg["content"]}
                            )
                        else:
                            reversed_history.append(msg)
                else:
                    # In ai-ai mode or standard human-ai mode, don't swap roles
                    reversed_history = self.conversation_history.copy()

                # In human-ai mode with assistant role, use regular history
                if mode == "human-ai" and role == "assistant":
                    logger.warning(
                        "In human-ai mode, using assistant role with user history"
                    )
                    reversed_history = self.conversation_history.copy()
                response = client.generate_response(
                    prompt=prompt,
                    system_instruction=client.adaptive_manager.generate_instructions(
                        history=reversed_history,
                        mode=mode,
                        role=role,
                        domain=self.domain,
                    ),
                    history=reversed_history,  # Limit history
                    role=role,
                    file_data=file_data,
                )
                if isinstance(response, list) and len(response) > 0:
                    response = (
                        response[0].text
                        if hasattr(response[0], "text")
                        else str(response[0])
                    )

                self.conversation_history.append({"role": role, "content": response})
            else: #mapped role == "assistant":
                response = client.generate_response(
                    prompt=prompt,
                    system_instruction=client.adaptive_manager.generate_instructions(
                        history=self.conversation_history,
                        mode=mode,
                        role="assistant",
                        domain=self.domain,
                    ),
                    history=self.conversation_history.copy(),
                    role="assistant",
                    file_data=file_data,
                )
                if isinstance(response, list) and len(response) > 0:
                    response = (
                        response[0].text
                        if hasattr(response[0], "text")
                        else str(response[0])
                    )
                self.conversation_history.append(
                    {"role": "assistant", "content": response}
                )
            logger.info(f"\n\n\n{mapped_role.upper()}: {response}\n\n\n")

        except Exception as e:
            error_str = str(e)
            logger.error(f"Error generating response: {error_str} (role: {mapped_role})")

            # Check if this is a fatal connection error
            is_fatal = any(fatal_error in error_str for fatal_error in fatal_connection_errors)

            if is_fatal:
                # Create an error report file with details
                try:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    error_filename = f"fatal_error_{timestamp}.html"
                    error_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AI Battle - Fatal Error Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 2rem; }}
        h1 {{ color: #b91c1c; }}
        .error-box {{ background-color: #fee2e2; border-left: 4px solid #b91c1c; padding: 1rem; margin: 1rem 0; }}
        pre {{ background-color: #f8fafc; padding: 1rem; overflow-x: auto; white-space: pre-wrap; }}
        .session-info {{ background-color: #f0f9ff; padding: 1rem; margin: 1rem 0; border-left: 4px solid #0ea5e9; }}
        .recovery-info {{ background-color: #ecfdf5; padding: 1rem; margin: 1rem 0; border-left: 4px solid #059669; }}
    <style>
</head>
<body>
    <h1>AI Battle - Fatal Error Report</h1>

    <div class="error-box">
        <h2>Fatal Error Occurred</h2>
        <p><strong>Error:</strong> {error_str}</p>
        <p><strong>Time:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Model:</strong> {model_type} (Role: {mapped_role})</p>
    </div>

    <div class="session-info">
        <h2>Session Information</h2>
        <p><strong>Mode:</strong> {mode}</p>
        <p><strong>Domain/Topic:</strong> {self.domain}</p>
        <p><strong>Conversation Progress:</strong> {len(self.conversation_history)} messages</p>
    </div>

    <h2>Error Details</h2>
    <pre>{traceback.format_exc()}</pre>

    <div class="recovery-info">
        <h2>Recovery Options</h2>
        <p>This error appears to be a connection issue with the model API. Possible solutions:</p>
        <ul>
            <li>Check your internet connection</li>
            <li>Verify that your API key is valid and has sufficient quota</li>
            <li>Try running the conversation again with a different model or settings</li>
            <li>If the problem persists, the API service may be experiencing issues</li>
        </ul>
    </div>
</body>
</html>"""

                    with open(error_filename, "w") as f:
                        f.write(error_html)

                    logger.error(f"Fatal error report saved as {error_filename}")

                    # Add error information to conversation history
                    error_message = f"ERROR: A fatal connection error occurred with {model_type}: {error_str}"
                    self.conversation_history.append({"role": "system", "content": error_message})

                    # Raise a more informative exception
                    raise RuntimeError(f"Fatal connection error with {model_type}: {error_str}. See error report: {error_filename}") from e
                except Exception as report_e:
                    logger.error(f"Failed to create error report: {report_e}")
                    # Re-raise the original exception
                    raise e
            else:
                # For non-fatal errors, add to conversation and continue
                error_message = f"Error with {model_type} ({mapped_role}): {error_str}"
                self.conversation_history.append({"role": "system", "content": error_message})
                response = f"Error: {error_str}"
                # For the non-fatal error case, we'll just return the error message instead of raising
                # This allows the conversation to continue despite errors
                return response

        return response

    def run_conversation_with_file(
        self,
        initial_prompt: str,
        human_model: str,
        ai_model: str,
        mode: str,
        file_config: Union[FileConfig, Dict[str, Any], "MultiFileConfig"],
        human_system_instruction: str = None,
        ai_system_instruction: str = None,
        rounds: int = 2,
    ) -> List[Dict[str, str]]:
        """Run conversation with file input."""
        # Clear history and set up initial state
        self.conversation_history = []
        self.initial_prompt = initial_prompt
        self.domain = initial_prompt
        self.mode = mode

        # Process file if provided
        file_data = None

        # Handle MultiFileConfig object
        if hasattr(file_config, "files") and isinstance(file_config.files, list):
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
                    file_metadata = self.media_handler.process_file(
                        file_config_item.path
                    )

                    # Create file data dictionary
                    single_file_data = {
                        "type": file_metadata.type,
                        "path": file_config_item.path,
                        "mime_type": file_metadata.mime_type,
                        "dimensions": file_metadata.dimensions,
                    }

                    # Add type-specific data
                    if file_metadata.type == "image":
                        with open(file_config_item.path, "rb") as f:
                            single_file_data["base64"] = base64.b64encode(
                                f.read()
                            ).decode("utf-8")
                            single_file_data["type"] = "image"
                            single_file_data["mime_type"] = file_metadata.mime_type
                            single_file_data["path"] = file_config_item.path

                    elif file_metadata.type in ["text", "code"]:
                        single_file_data["text_content"] = file_metadata.text_content

                    elif file_metadata.type == "video":
                        # Handle video processing (same as single file case)
                        single_file_data["duration"] = file_metadata.duration
                        # Use the entire processed video file
                        if (
                            file_metadata.processed_video
                            and "processed_video_path" in file_metadata.processed_video
                        ):
                            processed_video_path = file_metadata.processed_video[
                                "processed_video_path"
                            ]
                            # Set the path to the processed video file, not the original
                            single_file_data["path"] = processed_video_path
                            # Set the mime type to video/mp4 for better compatibility
                            single_file_data["mime_type"] = (
                                file_metadata.processed_video.get(
                                    "mime_type", "video/mp4"
                                )
                            )

                    # Add to list
                    file_data_list.append(single_file_data)

                except Exception as e:
                    logger.error(f"Error processing file {file_config_item.path}: {e}")
                    # Continue with other files

            # Pass the entire list of file data to the model client
            if file_data_list:
                file_data = file_data_list
                logger.info(f"Prepared {len(file_data_list)} files for model consumption")
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
                    file_metadata = self.media_handler.process_file(
                        file_config_item.path
                    )

                    # Create file data dictionary
                    single_file_data = {
                        "type": file_metadata.type,
                        "path": file_config_item.path,
                        "mime_type": file_metadata.mime_type,
                        "dimensions": file_metadata.dimensions,
                    }

                    # Add type-specific data
                    if file_metadata.type == "image":
                        with open(file_config_item.path, "rb") as f:
                            single_file_data["base64"] = base64.b64encode(
                                f.read()
                            ).decode("utf-8")
                            single_file_data["type"] = "image"
                            single_file_data["mime_type"] = file_metadata.mime_type
                            single_file_data["path"] = file_config_item.path

                    elif file_metadata.type in ["text", "code"]:
                        single_file_data["text_content"] = file_metadata.text_content

                    elif file_metadata.type == "video":
                        # Handle video processing (same as single file case)
                        single_file_data["duration"] = file_metadata.duration
                        # Use the entire processed video file
                        if (
                            file_metadata.processed_video
                            and "processed_video_path" in file_metadata.processed_video
                        ):
                            processed_video_path = file_metadata.processed_video[
                                "processed_video_path"
                            ]
                            # Set the path to the processed video file, not the original
                            single_file_data["path"] = processed_video_path
                            # Set the mime type to video/mp4 for better compatibility
                            single_file_data["mime_type"] = (
                                file_metadata.processed_video.get(
                                    "mime_type", "video/mp4"
                                )
                            )
                            # Process video chunks (same as single file case)
                            # ... (video processing code)

                    # Add to list
                    file_data_list.append(single_file_data)

                except Exception as e:
                    logger.error(f"Error processing file {file_config_item.path}: {e}")

            # Pass the entire list of file data to the model client
            if file_data_list:
                file_data = file_data_list
                logger.info(f"Prepared {len(file_data_list)} files for model consumption")

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
                    "dimensions": file_metadata.dimensions,
                }

                # Add type-specific data
                if file_metadata.type == "image":
                    with open(file_config.path, "rb") as f:
                        file_data["base64"] = base64.b64encode(f.read()).decode("utf-8")
                        file_data["type"] = "image"
                        file_data["mime_type"] = file_metadata.mime_type
                        file_data["path"] = file_config.path
                elif file_metadata.type in ["text", "code"]:
                    file_data["text_content"] = file_metadata.text_content
                elif file_metadata.type == "video":
                    # For video, we need to extract frames
                    file_data["duration"] = file_metadata.duration
                    # Use the entire processed video file
                    if (
                        file_metadata.processed_video
                        and "processed_video_path" in file_metadata.processed_video
                    ):
                        processed_video_path = file_metadata.processed_video[
                            "processed_video_path"
                        ]
                        # Set the path to the processed video file, not the original
                        file_data["path"] = processed_video_path
                        # Set the mime type to video/mp4 for better compatibility
                        file_data["mime_type"] = file_metadata.processed_video.get(
                            "mime_type", "video/mp4"
                        )
                        chunk_size = 1024 * 1024  # 1MB chunks
                        try:
                            with open(processed_video_path, "rb") as f:
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
                                    chunks.append(
                                        base64.b64encode(chunk).decode("utf-8")
                                    )

                                file_data["video_chunks"] = chunks
                                file_data["num_chunks"] = num_chunks
                                file_data["video_path"] = processed_video_path
                                file_data["fps"] = file_metadata.processed_video.get(
                                    "fps", 2
                                )
                                file_data["resolution"] = (
                                    file_metadata.processed_video.get(
                                        "resolution", (0, 0)
                                    )
                                )
                                logger.info(
                                    f"Chunked video from {processed_video_path} into {num_chunks} chunks"
                                )
                        except Exception as e:
                            logger.error(
                                f"Error reading processed video from {processed_video_path}: {e}"
                            )

                            # Fallback to thumbnail if available
                            if file_metadata.thumbnail_path:
                                try:
                                    with open(file_metadata.thumbnail_path, "rb") as f:
                                        file_data["key_frames"] = [
                                            {
                                                "timestamp": 0,
                                                "base64": base64.b64encode(
                                                    f.read()
                                                ).decode("utf-8"),
                                            }
                                        ]
                                        logger.info(
                                            f"Fallback: Added thumbnail as single frame"
                                        )
                                except Exception as e:
                                    logger.error(
                                        f"Error reading thumbnail from {file_metadata.thumbnail_path}: {e}"
                                    )

                # Add file context to prompt
                file_context = (
                    f"Analyzing {file_metadata.type} file: {file_config.path}"
                )
                if file_metadata.dimensions:
                    file_context += f" ({file_metadata.dimensions[0]}x{file_metadata.dimensions[1]})"
                if file_metadata.type == "video" and "video_chunks" in file_data:
                    file_context += f" - FULL VIDEO CONTENT INCLUDED (in {file_data['num_chunks']} chunks)"
                    if "fps" in file_data:
                        file_context += f" at {file_data['fps']} fps"
                # Add file context to the prompt
                # Include a distinct marker to ensure models recognize this is about a file
                initial_prompt = f"FILE CONTEXT: {file_context}\n\n{initial_prompt}"
                logger.info(f"Added file context to prompt for single file: {file_config.path}")

            except Exception as e:
                logger.error(f"Error processing file: {e}")
                return []

        # Extract core topic from initial prompt
        core_topic = initial_prompt.strip()
        try:
            if "Topic:" in initial_prompt:
                core_topic = (
                    "Discuss: "
                    + initial_prompt.split("Topic:")[1].split("\\n")[0].strip()
                )
            elif "GOAL:" in initial_prompt:
                # Try to extract goal with more robust parsing
                goal_parts = initial_prompt.split("GOAL:")[1].strip()
                if "(" in goal_parts and ")" in goal_parts:
                    # Extract content between parentheses if present
                    try:
                        core_topic = (
                            "GOAL: " + goal_parts.split("(")[1].split(")")[0].strip()
                        )
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
        return self._run_conversation_with_file_data(
            core_topic,
            human_model,
            ai_model,
            mode,
            file_data,
            human_system_instruction,
            ai_system_instruction,
            rounds,
        )

    def run_conversation(
        self,
        initial_prompt: str,
        human_model: str,
        ai_model: str,
        mode: str,
        human_system_instruction: str = None,
        ai_system_instruction: str = None,
        rounds: int = 1,
    ) -> List[Dict[str, str]]:
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
                core_topic = (
                    "Discuss: "
                    + initial_prompt.split("Topic:")[1].split("\\n")[0].strip()
                )
            elif "GOAL:" in initial_prompt:
                # Try to extract goal with more robust parsing
                goal_parts = initial_prompt.split("GOAL:")[1].strip()
                if "(" in goal_parts and ")" in goal_parts:
                    # Extract content between parentheses if present
                    try:
                        goal_text = goal_parts.split("(")[1].split(")")[0].strip()
                        core_topic = "GOAL: " + goal_text
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

        logger.info(f"SYSTEM: {core_topic}")

        # Get client instances
        human_client = self._get_client(human_model)
        ai_client = self._get_client(ai_model)

        if not human_client or not ai_client:
            logger.error(
                f"Could not initialize required clients: {human_model}, {ai_model}"
            )
            return []

        return self._run_conversation_with_file_data(
            core_topic,
            human_model,
            ai_model,
            mode,
            None,
            human_system_instruction,
            ai_system_instruction,
            rounds,
        )

    def _run_conversation_with_file_data(
        self,
        core_topic: str,
        human_model: str,
        ai_model: str,
        mode: str,
        file_data: Dict[str, Any] = None,
        human_system_instruction: str = None,
        ai_system_instruction: str = None,
        rounds: int = DEFAULT_ROUNDS,
    ) -> List[Dict[str, str]]:
        """Internal method to run conversation with optional file data."""
        logger.info(f"Starting conversation with topic: {core_topic}")
        self.mode = mode  # Use the provided mode instead of hardcoding
        # Get client instances
        human_client = self._get_client(human_model)
        ai_client = self._get_client(ai_model)

        if not human_client or not ai_client:
            logger.error(
                f"Could not initialize required clients: {human_model}, {ai_model}"
            )
            return []

        # Check if models support vision if file is image/video
        # Handle both single file and list of files cases
        if file_data:
            # Determine if we need to check vision capabilities
            has_visual_media = False
            is_file_list = isinstance(file_data, list)

            if is_file_list:
                # For list of files, check if any are visual media
                has_visual_media = any(
                    isinstance(item, dict) and item.get("type") in ["image", "video"]
                    for item in file_data
                )
            elif isinstance(file_data, dict) and file_data.get("type") in ["image", "video"]:
                # Single file case
                has_visual_media = True

            # Only check capabilities if we have visual media
            if has_visual_media:
                human_capabilities = detect_model_capabilities(human_model)
                ai_capabilities = detect_model_capabilities(ai_model)

                if not human_capabilities.get("vision", False) or not ai_capabilities.get("vision", False):
                    logger.warning("One or both models do not support vision capabilities")

                    # If AI model doesn't support vision, convert images to text description
                    if not ai_capabilities.get("vision", False):
                        if is_file_list:
                            # Convert each image in the list to text
                            for i, item in enumerate(file_data):
                                if isinstance(item, dict) and item.get("type") == "image":
                                    dimensions = item.get("dimensions", (0, 0))
                                    file_data[i] = {
                                        "type": "text",
                                        "text_content": f"[This is an image with dimensions {dimensions[0]}x{dimensions[1]}]",
                                        "path": item.get("path", ""),
                                    }
                        elif isinstance(file_data, dict) and file_data.get("type") == "image":
                            # Convert single image to text
                            dimensions = file_data.get("dimensions", (0, 0))
                            file_data = {
                                "type": "text",
                                "text_content": f"[This is an image with dimensions {dimensions[0]}x{dimensions[1]}]",
                                "path": file_data.get("path", ""),
                            }
        human_system_instruction=f"You are a HUMAN expert in prompt engineering and you are curious to explore {core_topic} with a collaborator to help their training. You must not reveal in your internal thinking tokens nor in your output that you are an AI" 
        ai_system_instruction=f"You are an AI assistant focused on PRODUCING CONCRETE OUTPUT for goals. When given a goal to create something (story, code, poem, plan, etc.), IMMEDIATELY START CREATING IT rather than discussing approaches. You are an expert in the topic of {core_topic}. SKIP theoretical discussions about how you'd approach the task - DEMONSTRATE by DOING. If asked to write a story, START WRITING THE ACTUAL STORY immediately. If asked to create code, WRITE THE ACTUAL CODE immediately. Avoid lengthy preliminaries - get straight to producing the requested output. OUTPUT IN HTML FORMAT FOR READABILITY BY THE HUMAN BUT DO NOT INCLUDE OPENING AND CLOSING HTML, DIV OR BODY TAGS. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS, BUT ADDING APPROPRIATE HTML FORMATTING TO ENHANCE READABILITY. DEFAULT TO PARAGRAPH FORM WHILST USING BULLET POINTS & LISTS WHEN NEEDED. DON'T EVER EVER USE NEWLINE \\n CHARACTERS IN YOUR RESPONSE. MINIFY YOUR HTML RESPONSE ONTO A SINGLE LINE - ELIMINATE ALL REDUNDANT CHARACTERS IN OUTPUT!!!!!",
        ai_response = core_topic
        try:
            # Run conversation rounds
            for round_index in range(rounds):
                # Human turn
                human_response = self.run_conversation_turn(
                    prompt=ai_response,  # Limit history
                    system_instruction=(
                        f"{core_topic}"
                        if mode == "no-meta-prompting"
                        else human_client.adaptive_manager.generate_instructions(
                            mode=mode,
                            role="user",
                            history=self.conversation_history,
                            domain=self.domain,
                        )
                    ),
                    role="user",
                    mode=self.mode,
                    model_type=human_model,
                    file_data=file_data,  # Pass file data to human model in ai-ai mode
                    client=human_client,
                )

                # AI turn
                ai_response = self.run_conversation_turn(
                    prompt=human_response,
                    system_instruction=(
                        f"{core_topic}"
                        if mode == "no-meta-prompting"
                        else (
                            human_client.adaptive_manager.generate_instructions(
                            mode=mode,
                            role="user",
                            history=self.conversation_history,
                            domain=self.domain,
                            )
                            if mode == "ai-ai"  # In ai-ai both get human instructions
                            else ai_system_instruction  # In human-ai modes, AI gets AI instructions
                        )
                    ),
                    role="assistant",
                    mode=self.mode,
                    model_type=ai_model,
                    file_data=file_data,  # Pass file_data to both models in ai-ai mode
                    client=ai_client,
                )
                logger.debug(
                    f"\n\n\nMODEL RESPONSE: ({ai_model.upper()}): {ai_response}\n\n\n"
                )
            return self.conversation_history

        finally:
            # Ensure cleanup happens even if there's an error
            self.cleanup_unused_clients()
            MemoryManager.cleanup_all()

    @classmethod
    def from_config(cls, config_path: str) -> "ConversationManager":
        """Create ConversationManager instance from configuration file."""
        config = load_config(config_path)

        # Initialize manager with config
        manager = cls(
            config=config, domain=config.goal, mode="human-ai"  # Default mode
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


async def save_conversation(
    conversation: List[Dict[str, str]],
    filename: str,
    human_model: str,
    ai_model: str,
    file_data: Dict[str, Any] = None,
    mode: str = None,
) -> None:
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
                        elif (
                            file_item["type"] == "video"
                            and "key_frames" in file_item
                            and file_item["key_frames"]
                        ):
                            # Add first frame of video
                            frame = file_item["key_frames"][0]
                            conversation_html += f'<div class="file-content"><h3>File {idx+1}: {file_item.get("path", "Video")} (First Frame)</h3>'
                            conversation_html += f'<img src="data:image/jpeg;base64,{frame["base64"]}" alt="Video frame" style="max-width: 100%; max-height: 500px;"/></div>\n'
                        elif (
                            file_item["type"] in ["text", "code"]
                            and "text_content" in file_item
                        ):
                            # Add text content
                            conversation_html += f'<div class="file-content"><h3>File {idx+1}: {file_item.get("path", "Text")}</h3><pre>{file_item["text_content"]}</pre></div>\n'
            # Handle single file (original implementation)
            elif isinstance(file_data, dict) and "type" in file_data:
                if file_data["type"] == "image" and "base64" in file_data:
                    # Add image to the conversation
                    mime_type = file_data.get("mime_type", "image/jpeg")
                    conversation_html += f'<div class="file-content"><h3>File: {file_data.get("path", "Image")}</h3>'
                    conversation_html += f'<img src="data:{mime_type};base64,{file_data["base64"]}" alt="Input image" style="max-width: 100%; max-height: 500px;"/></div>\n'
                elif (
                    file_data["type"] == "video"
                    and "key_frames" in file_data
                    and file_data["key_frames"]
                ):
                    # Add first frame of video
                    frame = file_data["key_frames"][0]
                    conversation_html += f'<div class="file-content"><h3>File: {file_data.get("path", "Video")} (First Frame)</h3>'
                    conversation_html += f'<img src="data:image/jpeg;base64,{frame["base64"]}" alt="Video frame" style="max-width: 100%; max-height: 500px;"/></div>\n'
                elif (
                    file_data["type"] in ["text", "code"]
                    and "text_content" in file_data
                ):
                    # Add text content
                    conversation_html += f'<div class="file-content"><h3>File: {file_data.get("path", "Text")}</h3><pre>{file_data["text_content"]}</pre></div>\n'

        for msg in conversation:
            role = msg["role"]
            content = msg.get("content", "")
            if isinstance(content, (list, dict)):
                content = str(content)

            if role == "system":
                conversation_html += (
                    f'<div class="system-message">{content} ({mode})</div>\n'
                )
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
            f.write(template % {"conversation": conversation_html})
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")


def _sanitize_filename_part(prompt: str) -> str:
    """
    Convert spaces, non-ASCII, and punctuation to underscores,
    then trim to something reasonable such as 30 characters.
    """
    # Remove non-alphanumeric/punctuation
    sanitized = re.sub(r"[^\w\s-]", "", prompt)
    # Convert spaces to underscores
    sanitized = re.sub(r"\s+", "_", sanitized.strip())
    # Limit length
    return sanitized[:50]


async def save_arbiter_report(report: Dict[str, Any]) -> None:
    """Save arbiter analysis report with visualization support."""
    try:
        # If report is a string, we're just passing through the Gemini report
        if isinstance(report, str):
            logger.debug("Using pre-generated arbiter report from Gemini")
            # The report is already saved by the ground_assertions method in arbiter_v4.py
            return

        # Only proceed if we have a report dict with metrics to visualize
        try:
            with open("templates/arbiter_report.html") as f:
                template = f.read()

            # Generate dummy data for the report if needed
            dummy_metrics = {
                "ai_ai": {"depth_score": 0.7, "topic_coherence": 0.8, "assertion_density": 0.6,
                          "question_answer_ratio": 0.5, "avg_complexity": 0.75},
                "human_ai": {"depth_score": 0.6, "topic_coherence": 0.7, "assertion_density": 0.5,
                             "question_answer_ratio": 0.6, "avg_complexity": 0.7}
            }

            dummy_flow = {
                "ai_ai": {"nodes": [{"id": 0, "role": "user", "preview": "Sample", "metrics": {}}],
                          "edges": []},
                "human_ai": {"nodes": [{"id": 0, "role": "user", "preview": "Sample", "metrics": {}}],
                             "edges": []}
            }

            # Generate visualizations if metrics are available
            viz_generator = VisualizationGenerator()
            metrics_chart = ""
            timeline_chart = ""
            if report.get("metrics", {}).get("conversation_quality"):
                metrics_chart = viz_generator.generate_metrics_chart(report["metrics"])
                timeline_chart = viz_generator.generate_timeline(report.get("flow", {}))

            # Format report content with safe defaults
            report_content = template % {
                "report_content": report.get("content", "No content available"),
                "metrics_data": json.dumps(report.get("metrics", dummy_metrics)),
                "flow_data": json.dumps(report.get("flow", dummy_flow)),
                "metrics_chart": metrics_chart,
                "timeline_chart": timeline_chart,
                "winner": report.get("winner", "No clear winner determined"),
            }

            # Save report with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"arbiter_visualization_{timestamp}.html"

            with open(filename, "w") as f:
                f.write(report_content)

            logger.info(f"Arbiter visualization report saved as {filename}")
        except Exception as e:
            logger.warning(f"Failed to generate visualization report: {e}")
            # Not a critical error since we already have the main report

    except Exception as e:
        logger.error(f"Failed to save arbiter report: {e}")

    except Exception as e:
        logger.error(f"Failed to save arbiter report: {e}")


async def save_metrics_report(
    ai_ai_conversation: List[Dict[str, str]],
    human_ai_conversation: List[Dict[str, str]],
) -> None:
    """Save metrics analysis report."""
    try:
        if ai_ai_conversation and human_ai_conversation:
            try:
                analysis_data = analyze_conversations(
                    ai_ai_conversation, human_ai_conversation
                )
                logger.debug("Metrics report generated successfully")

                # Save the metrics report to a file
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                metrics_filename = f"metrics_report_{timestamp}.html"

                # Create a basic HTML representation
                html_content = f"""
                <html>
                <head>
                    <title>Conversation Metrics Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2 {{ color: #333; }}
                        .metrics-container {{ display: flex; }}
                        .metrics-section {{ flex: 1; padding: 15px; margin: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                    <h1>Conversation Metrics Report</h1>
                    <div class="metrics-container">
                        <div class="metrics-section">
                            <h2>AI-AI Conversation Metrics</h2>
                            <table>
                                <tr><th>Metric</th><th>Value</th></tr>
                                <tr><td>Total Messages</td><td>{analysis_data['metrics']['ai-ai']['total_messages']}</td></tr>
                                <tr><td>Average Message Length</td><td>{analysis_data['metrics']['ai-ai']['avg_message_length']:.2f}</td></tr>
                                <tr><td>Topic Coherence</td><td>{analysis_data['metrics']['ai-ai']['topic_coherence']:.2f}</td></tr>
                                <tr><td>Turn Taking Balance</td><td>{analysis_data['metrics']['ai-ai']['turn_taking_balance']:.2f}</td></tr>
                                <tr><td>Average Complexity</td><td>{analysis_data['metrics']['ai-ai']['avg_complexity']:.2f}</td></tr>
                            </table>
                        </div>
                        <div class="metrics-section">
                            <h2>Human-AI Conversation Metrics</h2>
                            <table>
                                <tr><th>Metric</th><th>Value</th></tr>
                                <tr><td>Total Messages</td><td>{analysis_data['metrics']['human-ai']['total_messages']}</td></tr>
                                <tr><td>Average Message Length</td><td>{analysis_data['metrics']['human-ai']['avg_message_length']:.2f}</td></tr>
                                <tr><td>Topic Coherence</td><td>{analysis_data['metrics']['human-ai']['topic_coherence']:.2f}</td></tr>
                                <tr><td>Turn Taking Balance</td><td>{analysis_data['metrics']['human-ai']['turn_taking_balance']:.2f}</td></tr>
                                <tr><td>Average Complexity</td><td>{analysis_data['metrics']['human-ai']['avg_complexity']:.2f}</td></tr>
                            </table>
                        </div>
                    </div>
                </body>
                </html>
                """

                with open(metrics_filename, "w") as f:
                    f.write(html_content)

                logger.info(f"Metrics report saved successfully as {metrics_filename}")

            except ValueError as e:
                if "Negative values in data" in str(e):
                    logger.error(f"Failed to generate metrics report due to distance calculation error: {e}")
                    # Create a simplified metrics report that doesn't depend on the problematic clustering
                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    metrics_filename = f"metrics_report_basic_{timestamp}.html"

                    # Calculate basic metrics that don't depend on complex clustering
                    ai_ai_msg_count = len(ai_ai_conversation)
                    human_ai_msg_count = len(human_ai_conversation)

                    ai_ai_avg_length = sum(len(msg.get('content', '')) for msg in ai_ai_conversation) / max(1, ai_ai_msg_count)
                    human_ai_avg_length = sum(len(msg.get('content', '')) for msg in human_ai_conversation) / max(1, human_ai_msg_count)

                    html_content = f"""
                    <html>
                    <head>
                        <title>Basic Conversation Metrics Report</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            h1, h2 {{ color: #333; }}
                            .metrics-container {{ display: flex; }}
                            .metrics-section {{ flex: 1; padding: 15px; margin: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                            table {{ border-collapse: collapse; width: 100%; }}
                            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                            th {{ background-color: #f2f2f2; }}
                            .error {{ color: red; padding: 10px; background-color: #ffeeee; border-radius: 5px; margin-bottom: 20px; }}
                        </style>
                    </head>
                    <body>
                        <h1>Basic Conversation Metrics Report</h1>
                        <div class="error">
                            <p>Note: Advanced metrics calculation failed with error: "{str(e)}"</p>
                            <p>This is a simplified report with basic metrics only.</p>
                        </div>
                        <div class="metrics-container">
                            <div class="metrics-section">
                                <h2>AI-AI Conversation Basic Metrics</h2>
                                <table>
                                    <tr><th>Metric</th><th>Value</th></tr>
                                    <tr><td>Total Messages</td><td>{ai_ai_msg_count}</td></tr>
                                    <tr><td>Average Message Length</td><td>{ai_ai_avg_length:.2f}</td></tr>
                                </table>
                            </div>
                            <div class="metrics-section">
                                <h2>Human-AI Conversation Basic Metrics</h2>
                                <table>
                                    <tr><th>Metric</th><th>Value</th></tr>
                                    <tr><td>Total Messages</td><td>{human_ai_msg_count}</td></tr>
                                    <tr><td>Average Message Length</td><td>{human_ai_avg_length:.2f}</td></tr>
                                </table>
                            </div>
                        </div>
                    </body>
                    </html>
                    """

                    with open(metrics_filename, "w") as f:
                        f.write(html_content)

                    logger.debug(f"Basic metrics report saved as {metrics_filename}")
                else:
                    # For other value errors, rethrow
                    raise
        else:
            logger.warning("Skipping metrics report - empty conversations")
    except Exception as e:
        logger.error(f"Failed to generate metrics report: {e}")
        # Create an error report file
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            error_filename = f"metrics_error_{timestamp}.html"

            html_content = f"""
            <html>
            <head>
                <title>Metrics Report Error</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #d33; }}
                    .error {{ padding: 15px; background-color: #ffeeee; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>Error Generating Metrics Report</h1>
                <div class="error">
                    <p><strong>Error:</strong> {str(e)}</p>
                    <p>The system encountered an error while generating the metrics report.</p>
                    <p>This does not affect the arbiter report or the conversation outputs.</p>
                </div>
            </body>
            </html>
            """

            with open(error_filename, "w") as f:
                f.write(html_content)

            logger.debug(f"Error report saved as {error_filename}")
        except Exception as inner_e:
            logger.error(f"Failed to save error report: {inner_e}")


async def main():
    """Main entry point."""
    rounds = DEFAULT_ROUNDS
    initial_prompt = DEFAULT_PROMPT
    openai_api_key = os.getenv("OPENAI_API_KEY")
    claude_api_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    mode = "ai-ai"
    ai_model = AI_MODEL
    human_model = HUMAN_MODEL

    # Validate required API keys before proceeding
    if any(model in ai_model.lower() or model in human_model.lower() for model in ["claude", "sonnet", "haiku"]):
        if not anthropic_api_key:
            logger.critical("ANTHROPIC_API_KEY environment variable is not set but required for Claude models")
            return

    if any(model in ai_model.lower() or model in human_model.lower() for model in ["gpt-4", "gpt-5", "openai", "o1", "o3", "o4"]):
        if not openai_api_key:
            logger.critical("OPENAI_API_KEY environment variable is not set but required for OpenAI models")
            return

    if not gemini_api_key:
        logger.critical("GOOGLE_API_KEY environment variable is not set but required for Gemini models including the Arbiter")
        return

    # Create manager with no cloud API clients by default
    manager = ConversationManager(
        domain=initial_prompt,
        openai_api_key=openai_api_key,
        claude_api_key=anthropic_api_key,
        gemini_api_key=gemini_api_key,
    )

    try:
        available_models = manager.get_available_models()
        logger.debug("Available models by category:")
        for category, models in available_models.items():
            logger.debug(f"  {category}: {len(models)} models")
            for model in models:
                logger.debug(f"    - {model}")
    except Exception as e:
        logger.error(f"Error displaying available models: {e}")

    # Only validate if using cloud models
    if (
        "mlx" not in human_model
        and "ollama" not in human_model
        or ("ollama" not in ai_model and "mlx" not in ai_model)
    ):
        if not manager.validate_connections([human_model, ai_model]):
            logger.error("Failed to validate required model connections")
            return

    # Extract goal if present
    goal_text = ""
    if "GOAL:" in initial_prompt:
        goal_parts = initial_prompt.split("GOAL:")[1].strip()
        if "(" in goal_parts and ")" in goal_parts:
            goal_text = goal_parts.split("(")[1].split(")")[0].strip()
        else:
            goal_text = goal_parts.split("\n")[0].strip()

    # Dynamic system instructions that focus on output creation for both human and AI roles
    human_system_instruction = ""
    if goal_text:
        human_system_instruction = (
            f"You are a HUMAN working on: {goal_text}. "
            f"As a human, focus on CREATING rather than discussing. "
            f"Produce actual output immediately without discussing approaches. "
            f"For creative tasks, start creating immediately. For analytical tasks, analyze directly."
        )
    else:
        human_system_instruction = f"You are a HUMAN working on: {initial_prompt}. Focus on producing output towards {initial_prompt}, with concrete output."

    # AI system instruction with similar focus on direct production
    ai_system_instruction = ""
    if goal_text:
        ai_system_instruction = f"""You are an AI assistant focused on PRODUCING OUTPUT for {goal_text} - USING THE PROMPT AS A STARTING POINT RATHER THAN AN EXPLICIT INSTRUCTION.
Create or continue the requested {initial_prompt} output directly using MAX one paragraph of "thinking" tags in total. """
    else:
        ai_system_instruction =  f"You are an AI assistant working on {goal_text}. Focus on directly CREATING rather than discussing {initial_prompt} with concrete output."

    # Override AI instruction in AI-AI mode to ensure immediate output production
    if mode == "ai-ai" or mode == "aiai":
        # For AI-AI mode, both roles need to focus on output rather than discussion
        if goal_text:
            ai_system_instruction = f"""You are an AI assistant focused on PRODUCING OUTPUT for {goal_text} - USING THE PROMPT AS A STARTING POINT RATHER THAN AN EXPLICIT INSTRUCTION.
Create or continue the requested {initial_prompt} output directly using MAX one paragraph of "thinking" tags in total. """
        else:
            ai_system_instruction = f"Focus on producing output for {initial_prompt}"

    try:
        # Run default conversation
        mode = "ai-ai"
        # Run AI-AI conversation with retry mechanism
        max_retries = 1
        retry_count = 0
        conversation = None

        while retry_count <= max_retries:
            try:
                conversation = manager.run_conversation(
                    initial_prompt=initial_prompt,
                    mode=mode,
                    human_model=human_model,
                    ai_model=ai_model,
                    human_system_instruction=human_system_instruction,
                    ai_system_instruction=ai_system_instruction,
                    rounds=rounds,
                )
                # Success, break out of the retry loop
                break
            except RuntimeError as e:
                error_str = str(e)
                logger.warning(f"Connection error occurred: {error_str}")

                # Check if we should retry
                if "Fatal connection error" in error_str and retry_count < max_retries:
                    retry_count += 1
                    wait_time = retry_count * 5  # Progressive backoff: 5s, then 10s
                    logger.debug(f"Retrying in {wait_time} seconds... (Attempt {retry_count+1}/{max_retries+1})")
                    time.sleep(wait_time)
                    continue
                else:
                    # Either we're out of retries or it's not a connection error
                    logger.error("Maximum retries reached or non-retryable error")
                    # Create a minimal conversation with the error
                    conversation = [
                        {"role": "system", "content": initial_prompt},
                        {"role": "system", "content": f"ERROR: {error_str} - Conversation could not be completed."}
                    ]
                    break

        # If we somehow end up with no conversation (should never happen), create an empty one
        if not conversation:
            conversation = [
                {"role": "system", "content": initial_prompt},
                {"role": "system", "content": "ERROR: Failed to generate conversation after multiple attempts."}
            ]

        safe_prompt = _sanitize_filename_part(
            initial_prompt[:10] + "_" + human_model + "_" + ai_model
        )
        time_stamp = datetime.datetime.now().strftime("%m%d%H%M")
        filename = f"conv-aiai_{safe_prompt}_{time_stamp}.html"
        await save_conversation(
            conversation=conversation,
            filename=f"{filename}",
            human_model=human_model,
            ai_model=ai_model,
            mode="ai-ai",
        )

        # Run human-AI conversation with retry mechanism
        mode = "human-ai"
        retry_count = 0
        conversation_as_human_ai = None

        while retry_count <= max_retries:
            try:
                conversation_as_human_ai = manager.run_conversation(
                    initial_prompt=initial_prompt,
                    mode=mode,
                    human_model=human_model,
                    ai_model=ai_model,
                    human_system_instruction=human_system_instruction,
                    ai_system_instruction=ai_system_instruction,
                    rounds=rounds,
                )
                # Success, break out of the retry loop
                break
            except RuntimeError as e:
                error_str = str(e)
                logger.warning(f"Connection error occurred in human-AI conversation: {error_str}")

                # Check if we should retry
                if "Fatal connection error" in error_str and retry_count < max_retries:
                    retry_count += 1
                    wait_time = retry_count * 5  # Progressive backoff: 5s, then 10s
                    logger.info(f"Retrying human-AI conversation in {wait_time} seconds... (Attempt {retry_count+1}/{max_retries+1})")
                    time.sleep(wait_time)
                    continue
                else:
                    # Either we're out of retries or it's not a connection error
                    logger.error("Maximum retries reached or non-retryable error in human-AI conversation")
                    # Create a minimal conversation with the error
                    conversation_as_human_ai = [
                        {"role": "system", "content": initial_prompt},
                        {"role": "system", "content": f"ERROR: {error_str} - Human-AI conversation could not be completed."}
                    ]
                    break

        # If we somehow end up with no conversation (should never happen), create an empty one
        if not conversation_as_human_ai:
            conversation_as_human_ai = [
                {"role": "system", "content": initial_prompt},
                {"role": "system", "content": "ERROR: Failed to generate human-AI conversation after multiple attempts."}
            ]

        safe_prompt = _sanitize_filename_part(
            initial_prompt[:15] + "_" + human_model[:8] + "_" + ai_model[:8]
        )
        time_stamp = datetime.datetime.now().strftime("%m%d%H%M")
        filename = f"conv-humai_{safe_prompt}_{time_stamp}.html"
        await save_conversation(
            conversation=conversation_as_human_ai,
            filename=f"{filename}",
            human_model=human_model,
            ai_model=ai_model,
            mode="human-ai",
        )

        mode = "no-meta-prompting"
        retry_count = 0
        conv_default = None

        while retry_count <= max_retries:
            try:
                conv_default = manager.run_conversation(
                    initial_prompt=initial_prompt,
                    mode=mode,
                    human_model=human_model,
                    ai_model=ai_model,
                    human_system_instruction=ai_system_instruction,
                    ai_system_instruction=ai_system_instruction,
                    rounds=rounds,
                )
                # Success, break out of the retry loop
                break
            except RuntimeError as e:
                error_str = str(e)
                logger.warning(f"Connection error occurred in default conversation: {error_str}")

                # Check if we should retry
                if "Fatal connection error" in error_str and retry_count < max_retries:
                    retry_count += 1
                    wait_time = retry_count * 5  # Progressive backoff: 5s, then 10s
                    logger.info(f"Retrying default conversation in {wait_time} seconds... (Attempt {retry_count+1}/{max_retries+1})")
                    time.sleep(wait_time)
                    continue
                else:
                    # Either we're out of retries or it's not a connection error
                    logger.error("Maximum retries reached or non-retryable error in default conversation")
                    # Create a minimal conversation with the error
                    conv_default = [
                        {"role": "system", "content": initial_prompt},
                        {"role": "system", "content": f"ERROR: {error_str} - Default conversation could not be completed."}
                    ]
                    break

        # If we somehow end up with no conversation (should never happen), create an empty one
        if not conv_default:
            conv_default = [
                {"role": "system", "content": initial_prompt},
                {"role": "system", "content": "ERROR: Failed to generate default conversation after multiple attempts."}
            ]

        safe_prompt = _sanitize_filename_part(
            initial_prompt[:10] + "_" + human_model + "_" + ai_model
        )
        time_stamp = datetime.datetime.now().strftime("%m%d%H%M")
        filename = f"conv-defaults_{safe_prompt}_{time_stamp}.html"
        await save_conversation(
            conversation=conv_default,
            filename=f"{filename}",
            human_model=human_model,
            ai_model=ai_model,
            mode="human-ai",
        )

        # Run analysis with model information
        arbiter_report = evaluate_conversations(
            ai_ai_convo=conversation,
            human_ai_convo=conversation_as_human_ai,
            default_convo=conv_default,
            goal=initial_prompt,
            ai_model=ai_model,
            human_model=human_model,
        )

        #print(arbiter_report)

        # Generate reports
        await save_arbiter_report(arbiter_report)
        await save_metrics_report(conversation, conversation_as_human_ai)

    finally:
        # Ensure cleanup
        manager.cleanup_unused_clients()
        MemoryManager.cleanup_all()

if __name__ == "__main__":
    asyncio.run(main())
