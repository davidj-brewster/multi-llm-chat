"""Factory for creating and caching model clients."""
import logging
from typing import Dict, Optional
from model_clients import (
    BaseClient,
    OpenAIClient,
    ClaudeClient,
    GeminiClient,
    MLXClient,
    OllamaClient,
)
from lmstudio_client import LMStudioClient
from core.model_registry import get_registry
from shared_resources import MemoryManager

logger = logging.getLogger(__name__)


class ClientFactory:
    """Factory for creating and caching model clients."""

    def __init__(
        self,
        mode: str = None,
        domain: str = "General knowledge",
        openai_api_key: Optional[str] = None,
        claude_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ):
        """Initialize factory with configuration."""
        self._cache: Dict[str, BaseClient] = {}
        self._registry = get_registry()
        self.mode = mode
        self.domain = domain
        self.openai_api_key = openai_api_key
        self.claude_api_key = claude_api_key
        self.gemini_api_key = gemini_api_key
        self._ollama_models = {}
        self._lmstudio_models = {}

    def get_client(self, model_name: str) -> Optional[BaseClient]:
        """
        Get or create a client for the specified model.

        Args:
            model_name: Name of the model

        Returns:
            Client instance (cached if previously created)
        """
        # Check cache first
        if model_name in self._cache:
            logger.debug(f"Using cached client for: {model_name}")
            return self._cache[model_name]

        logger.info(f"Creating new client for: {model_name}")

        # Create new client
        client = self._create_client(model_name)

        # Cache and return
        if client:
            self._cache[model_name] = client
            logger.debug(MemoryManager.get_memory_usage())

        return client

    def _create_client(self, model_name: str) -> Optional[BaseClient]:
        """Create a new client instance based on model name."""
        try:
            # Handle Claude models
            claude_config = self._registry.claude_models.get(model_name)
            if claude_config:
                client = ClaudeClient(
                    role=None,
                    api_key=self.claude_api_key,
                    mode=self.mode,
                    domain=self.domain,
                    model=claude_config["model"],
                )

                if claude_config["reasoning_level"] is not None:
                    client.reasoning_level = claude_config["reasoning_level"]
                    logger.debug(f"Set reasoning level to '{claude_config['reasoning_level']}' for {model_name}")

                if claude_config.get("extended_thinking", False):
                    budget_tokens = claude_config.get("budget_tokens", None)
                    client.set_extended_thinking(True, budget_tokens)
                    logger.info(f"Enabled extended thinking with budget_tokens={budget_tokens} for {model_name}")

                return client

            # Handle OpenAI models
            openai_config = self._registry.openai_models.get(model_name)
            if openai_config:
                client = OpenAIClient(
                    api_key=self.openai_api_key,
                    role=None,
                    mode=self.mode,
                    domain=self.domain,
                    model=openai_config["model"],
                )

                if openai_config["reasoning_level"] is not None:
                    client.reasoning_level = openai_config["reasoning_level"]
                    logger.debug(f"Set reasoning level to '{openai_config['reasoning_level']}' for {model_name}")

                return client

            # Handle Gemini models
            gemini_config = self._registry.gemini_models.get(model_name)
            if gemini_config:
                return GeminiClient(
                    api_key=self.gemini_api_key,
                    role=None,
                    mode=self.mode,
                    domain=self.domain,
                    model=gemini_config["model"],
                )

            # Handle Ollama models dynamically
            if model_name.startswith("ollama-"):
                return self._create_ollama_client(model_name)

            # Handle LMStudio models dynamically
            if model_name.startswith("lmstudio-"):
                return self._create_lmstudio_client(model_name)

            # Handle MLX models
            if model_name == "mlx-llama-3.1-abb":
                return MLXClient(
                    mode=self.mode,
                    domain=self.domain,
                    model="mlx-community/Meta-Llama-3.1-8B-Instruct-abliterated-8bit",
                )

            logger.error(f"Unknown model: {model_name}")
            return None

        except Exception as e:
            logger.error(f"Failed to create client for {model_name}: {e}")
            return None

    def _create_ollama_client(self, model_name: str) -> Optional[OllamaClient]:
        """Create Ollama client with model discovery."""
        # Get available Ollama models if not cached
        if not self._ollama_models:
            try:
                temp_client = OllamaClient(mode=self.mode, domain=self.domain, model="")
                model_list = temp_client.client.list()

                for model_info in model_list.models:
                    actual_name = model_info.model
                    if actual_name:
                        base_name = actual_name.split(':')[0].split('/')[-1].lower()
                        friendly_name = f"ollama-{base_name}"
                        full_friendly_name = f"ollama-{actual_name.lower()}"
                        self._ollama_models[friendly_name] = actual_name
                        self._ollama_models[full_friendly_name] = actual_name

                logger.info(f"Found {len(self._ollama_models)} available Ollama models")
            except Exception as e:
                logger.error(f"Failed to fetch Ollama models: {e}")

        # Get the actual model name
        actual_model = self._ollama_models.get(model_name)
        if not actual_model:
            # Try direct model name
            model_suffix = model_name[len("ollama-"):]
            if ":" not in model_suffix:
                model_suffix = f"{model_suffix}:latest"
            actual_model = model_suffix
            logger.warning(f"Using direct model name: {actual_model}")

        client = OllamaClient(mode=self.mode, domain=self.domain, model=actual_model)

        # Apply thinking configuration if applicable
        actual_model_lower = actual_model.lower()
        for keyword, thinking_config in self._registry.ollama_thinking_config.items():
            if keyword in actual_model_lower:
                if thinking_config.get("reasoning_level"):
                    client.reasoning_level = thinking_config["reasoning_level"]
                    logger.info(f"Set reasoning level to '{thinking_config['reasoning_level']}' for {model_name}")
                if thinking_config.get("extended_thinking", False):
                    client.set_extended_thinking(True)
                    logger.info(f"Enabled extended thinking for {model_name}")
                if "num_ctx" in thinking_config:
                    client.num_ctx = thinking_config["num_ctx"]
                break

        return client

    def _create_lmstudio_client(self, model_name: str) -> Optional[LMStudioClient]:
        """Create LMStudio client with model discovery."""
        # Get available LMStudio models if not cached
        if not self._lmstudio_models:
            try:
                temp_client = LMStudioClient(mode=self.mode, domain=self.domain)
                for actual_name in temp_client.available_models:
                    if actual_name:
                        parts = actual_name.split('/')
                        base_name = parts[-1].lower() if len(parts) > 1 else actual_name.lower()
                        friendly_name = f"lmstudio-{base_name}"
                        self._lmstudio_models[friendly_name] = actual_name

                logger.info(f"Found {len(self._lmstudio_models)} available LMStudio models")
            except Exception as e:
                logger.warning(f"Failed to fetch LMStudio models: {e}")

        # Get the actual model name
        actual_model = None
        if self._lmstudio_models:
            for lms_name, lms_actual in self._lmstudio_models.items():
                if model_name == lms_name or model_name.startswith(lms_name) or lms_name.startswith(model_name):
                    actual_model = lms_actual
                    break

        if actual_model:
            client = LMStudioClient(mode=self.mode, domain=self.domain, model=actual_model)
            logger.info(f"Using LMStudio model: {model_name} -> {actual_model}")
        else:
            # Create client with no specific model - will use first available
            client = LMStudioClient(mode=self.mode, domain=self.domain)
            logger.warning(f"LMStudio model {model_name} not found, using default")

        return client

    def clear_cache(self) -> None:
        """Clear the client cache."""
        logger.info(f"Clearing client cache ({len(self._cache)} clients)")
        self._cache.clear()

    def get_cached_models(self) -> list:
        """Get list of models with cached clients."""
        return list(self._cache.keys())
