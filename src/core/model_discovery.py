"""Discovery of locally available models (Ollama, LMStudio)."""
import logging
from typing import List, Dict, Any
from model_clients import OllamaClient
from lmstudio_client import LMStudioClient

logger = logging.getLogger(__name__)


class ModelDiscovery:
    """Discovers locally available models."""

    @staticmethod
    def discover_ollama_models() -> Dict[str, str]:
        """
        Discover available Ollama models via API.

        Returns:
            Dictionary mapping friendly names to actual model names
        """
        models = {}
        try:
            temp_client = OllamaClient(mode=None, domain="", model="")
            model_list = temp_client.client.list()

            for model_info in model_list.models:
                model_name = model_info.model
                if model_name:
                    base_name = model_name.split(':')[0].split('/')[-1].lower()
                    friendly_name = f"ollama-{base_name}"
                    full_friendly_name = f"ollama-{model_name.lower()}"
                    models[friendly_name] = model_name
                    models[full_friendly_name] = model_name

            logger.info(f"Discovered {len(models)} Ollama models")
        except Exception as e:
            logger.debug(f"Could not discover Ollama models: {e}")

        return models

    @staticmethod
    def discover_lmstudio_models() -> Dict[str, str]:
        """
        Discover available LMStudio models via API.

        Returns:
            Dictionary mapping friendly names to actual model names
        """
        models = {}
        try:
            temp_client = LMStudioClient(mode=None, domain="")

            for model_name in temp_client.available_models:
                if model_name:
                    parts = model_name.split('/')
                    base_name = parts[-1].lower() if len(parts) > 1 else model_name.lower()
                    friendly_name = f"lmstudio-{base_name}"
                    models[friendly_name] = model_name

            logger.info(f"Discovered {len(models)} LMStudio models")
        except Exception as e:
            logger.debug(f"Could not discover LMStudio models: {e}")

        return models

    @staticmethod
    def get_all_local_models() -> Dict[str, Dict[str, str]]:
        """
        Get all locally available models from all providers.

        Returns:
            Dictionary with provider names as keys and model dictionaries as values
        """
        return {
            "ollama": ModelDiscovery.discover_ollama_models(),
            "lmstudio": ModelDiscovery.discover_lmstudio_models(),
        }
