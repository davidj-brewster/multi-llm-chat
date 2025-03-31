"""
LMStudio client implementation for Claude-Gemini-Chat framework.

This module provides a client for interacting with locally running LMStudio models
through their OpenAI-compatible API endpoint.
"""

import os
import logging
from typing import List, Dict, Optional, Any
import json
import requests
from openai import OpenAI

# Import base client and configuration
from model_clients import OpenAIClient, ModelConfig, BaseClient

logger = logging.getLogger(__name__)


class LMStudioClient(OpenAIClient):
    """
    Client for LMStudio API interactions through the OpenAI-compatible endpoint.

    LMStudio provides a local server with OpenAI-compatible endpoints for running
    large language models on your own hardware. This client extends the OpenAIClient
    to support connecting to a local LMStudio server.

    Attributes:
        base_url (str): Base URL for the LMStudio server (defaults to http://localhost:1234/v1)
        available_models (List[str]): List of available models in the local LMStudio instance
    """

    def __init__(
        self,
        role: str = "assistant",
        mode: str = "ai-ai",
        domain: str = "General Knowledge",
        model: str = "local-model",
        base_url: str = "http://localhost:1234/v1",
    ):
        """
        Initialize a new LMStudioClient.

        Args:
            role (str, optional): The role this client plays in conversations.
                Defaults to "assistant".
            mode (str, optional): Conversation mode ("ai-ai", "human-ai", etc).
                Defaults to "ai-ai".
            domain (str, optional): Knowledge domain for the client.
                Defaults to "General Knowledge".
            model (str, optional): Model name to use.
                Defaults to "local-model".
            base_url (str, optional): Base URL for the LMStudio server.
                Defaults to "http://localhost:1234/v1".
        """
        # Use a minimal API key for construction
        minimal_api_key = "lmstudio-local"

        # Initialize the OpenAI client parent with the base URL
        super().__init__(
            role=role, api_key=minimal_api_key, mode=mode, domain=domain, model=model
        )

        # Override the base URL to point to the local LMStudio server
        self.base_url = base_url

        # Initialize the OpenAI client with the local base URL
        self.client = OpenAI(base_url=base_url, api_key=minimal_api_key)

        # Track available models
        self.available_models = []

        # Log initialization
        logger.info(f"Initialized LMStudioClient with base URL: {base_url}")

        # Try to fetch available models
        self._fetch_available_models()

    def _fetch_available_models(self):
        """Fetch available models from the LMStudio server."""
        try:
            # Try to get models directly with the client
            response = self.client.models.list()
            self.available_models = [model.id for model in response.data]
            logger.info(f"Available LMStudio models: {self.available_models}")
        except Exception as e:
            # Fallback to direct HTTP request if client method fails
            try:
                response = requests.get(f"{self.base_url}/models")
                if response.status_code == 200:
                    models_data = response.json()
                    if isinstance(models_data, dict) and "data" in models_data:
                        self.available_models = [
                            model["id"] for model in models_data["data"]
                        ]
                    elif isinstance(models_data, list):
                        self.available_models = [model["id"] for model in models_data]
                    logger.info(f"Available LMStudio models: {self.available_models}")
                else:
                    logger.warning(
                        f"Failed to fetch models from LMStudio: {response.status_code}"
                    )
            except Exception as inner_e:
                logger.warning(f"Could not fetch available models: {inner_e}")
                # Use a default model name as fallback
                self.available_models = ["local-model"]

    def validate_connection(self) -> bool:
        """Test connection to LMStudio server."""
        try:
            # Try to get models
            self._fetch_available_models()
            if self.available_models:
                logger.info(
                    f"Successfully connected to LMStudio server at {self.base_url}"
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to LMStudio server: {e}")
            return False

    def generate_response(
        self,
        prompt: str,
        system_instruction: str = None,
        history: List[Dict[str, str]] = None,
        file_data: Dict[str, Any] = None,
        role: str = None,
        model_config: Optional[ModelConfig] = None,
    ) -> str:
        """
        Generate a response using the local LMStudio model.

        Args:
            prompt (str): The text prompt to send to the model.
            system_instruction (str, optional): System instructions for the model.
            history (List[Dict[str, str]], optional): Conversation history.
            file_data (Dict[str, Any], optional): File data for multimodal inputs.
            role (str, optional): The role for this generation.
            model_config (ModelConfig, optional): Configuration for the model.

        Returns:
            str: The generated response text.

        Raises:
            Exception: If there is an error generating the response.
        """
        # LMStudio doesn't support multimodal inputs currently
        if file_data and (
            isinstance(file_data, dict) and file_data.get("type") in ["image", "video"]
        ):
            logger.warning(
                "LMStudio does not support image or video inputs. Ignoring file data."
            )
            file_data = None

        # Call the parent class's generate_response for compatible interface
        try:
            # Use the first available model if the current one isn't available
            if self.model not in self.available_models and self.available_models:
                original_model = self.model
                self.model = self.available_models[0]
                logger.info(
                    f"Model {original_model} not available, using {self.model} instead"
                )

            history = history if history else [{"role": "user", "content": prompt}]

            if role == "user" or role == "human" or self.mode == "ai-ai":
                current_instructions = self.adaptive_manager.generate_instructions(
                    history, role=role, domain=self.domain, mode=self.mode
                )
            else:
                current_instructions = (
                    system_instruction
                    if system_instruction is not None
                    else (
                        self.instructions
                        if self.instructions and self.instructions is not None
                        else f"You are an expert in {self.domain}. Respond at expert level using step by step thinking where appropriate"
                    )
                )

            # Format messages for OpenAI API
            formatted_messages = []

            # Add system message
            formatted_messages.append({"role": "system", "content": current_instructions})

            # Add history messages
            if history:
                for msg in history:
                    old_role = msg["role"]
                    if old_role in ["user", "assistant", "moderator", "system"]:
                        new_role = (
                            "system"
                            if old_role in ["system", "Moderator","developer"]
                            else (
                                "user"
                                if old_role in ["user", "human", "moderator"]
                                else "assistant"
                            )
                        )
                        formatted_messages.append(
                            {"role": new_role, "content": msg["content"]}
                        )
            prompt = (
                self.generate_human_prompt(history)
                if (role == "human" or role == "user" or self.mode == "ai-ai")
                else f"{prompt}"
            )

            custom_instruction = self.adaptive_manager.generate_instructions(
                history=formatted_messages,
                domain=self.domain,
                mode=self.mode,
                role=role,
            )

            # Use the custom instruction if available
            if custom_instruction:
                system_instruction = custom_instruction
            # Use the system instruction if provided
            elif system_instruction:
                system_instruction = system_instruction
            
            # Use a plain configuration without reasoning parameters
            # since LMStudio models typically don't support them
            if model_config is None:
                model_config = ModelConfig(temperature=0.7, max_tokens=1024)
                model_config.seed = None  # Explicitly set seed to None

            logger.info(
                f"Generating response with LMStudio model {self.model} using prompt: {prompt}"
            )
            return super().generate_response(
                prompt=prompt,
                system_instruction=custom_instruction,
                history=formatted_messages,
                file_data=file_data,
                role=role,
                model_config=model_config,
            )
        except Exception as e:
            logger.error(f"Error generating response from LMStudio: {e}")
            return f"Error: Could not generate response from LMStudio server. Please check that the server is running at {self.base_url} and that you have a model loaded. Error details: {e}"


# Simple test function
def test_lmstudio_client():
    """Test the LMStudio client with a simple prompt."""
    client = LMStudioClient()
    if client.validate_connection():
        print(
            f"Connected to LMStudio server. Available models: {client.available_models}"
        )

        response = client.generate_response(
            prompt="Explain how large language models work in one paragraph.",
            system_instruction="You are a helpful AI assistant.",
        )

        print("\nResponse from LMStudio:")
        print("=" * 80)
        print(response)
        print("=" * 80)
    else:
        print(
            "Failed to connect to LMStudio server. Make sure it's running at http://localhost:1234"
        )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run the test
    test_lmstudio_client()
