"""Model registry for managing model configurations."""
import yaml
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for model configurations."""

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize registry from YAML file.

        Args:
            registry_path: Path to model registry YAML file.
                          Defaults to src/data/model_registry.yaml
        """
        if registry_path is None:
            # Default path relative to this file
            registry_path = Path(__file__).parent.parent / "data" / "model_registry.yaml"

        logger.debug(f"Loading model registry from: {registry_path}")

        with open(registry_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.openai_models = data.get("openai_models", {})
        self.gemini_models = data.get("gemini_models", {})
        self.claude_models = data.get("claude_models", {})
        self.ollama_thinking_config = data.get("ollama_thinking_config", {})

        logger.info(
            f"Loaded model registry: {len(self.openai_models)} OpenAI, "
            f"{len(self.gemini_models)} Gemini, {len(self.claude_models)} Claude models"
        )

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific model.

        Args:
            model_name: Name of the model (e.g., "o3", "claude-sonnet-4")

        Returns:
            Model configuration dictionary or None if not found
        """
        # Check all registries
        for registry in [self.openai_models, self.gemini_models, self.claude_models]:
            if model_name in registry:
                return registry[model_name]

        logger.debug(f"Model not found in registry: {model_name}")
        return None

    def list_models(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        List all models or models from a specific provider.

        Args:
            provider: Provider name ("openai", "gemini", "claude") or None for all

        Returns:
            Dictionary of model configurations
        """
        if provider == "openai":
            return self.openai_models
        if provider == "gemini":
            return self.gemini_models
        if provider == "claude":
            return self.claude_models
        if provider is None:
            # Return all models
            return {
                **self.openai_models,
                **self.gemini_models,
                **self.claude_models,
            }
        raise ValueError(f"Unknown provider: {provider}")

    def get_ollama_thinking_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get Ollama thinking configuration for a model.

        Args:
            model_name: Ollama model name

        Returns:
            Thinking configuration or None
        """
        return self.ollama_thinking_config.get(model_name)


# Singleton instance and thread lock
_registry: Optional[ModelRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    """
    Get the global model registry instance (thread-safe).

    Returns:
        Singleton ModelRegistry instance
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            # Double-check locking pattern
            if _registry is None:
                _registry = ModelRegistry()
    return _registry


def reload_registry(registry_path: Optional[Path] = None) -> ModelRegistry:
    """
    Reload the model registry from file (thread-safe).

    Args:
        registry_path: Optional path to registry file

    Returns:
        Reloaded ModelRegistry instance
    """
    global _registry
    with _registry_lock:
        _registry = ModelRegistry(registry_path)
    return _registry
