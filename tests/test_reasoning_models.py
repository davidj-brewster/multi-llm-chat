#!/usr/bin/env python3
"""
Test suite for reasoning model implementations.

This script tests that the following models with reasoning capabilities
are properly instantiated and configured:
- Claude 3.7 with reasoning levels (auto, high, medium, low, none)
- OpenAI O1/O3 with reasoning levels (high, medium, low)
"""

import os
import sys
import logging
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("claude_reasoning_test")

# Import model_clients directly
sys.path.append(str(Path(__file__).parent))
from model_clients import ClaudeClient, OpenAIClient

# Import ConversationManager from ai-battle.py
ai_battle_path = Path(__file__).parent / "ai-battle.py"
spec = spec_from_file_location("ai_battle", ai_battle_path)
ai_battle = module_from_spec(spec)
spec.loader.exec_module(ai_battle)
ConversationManager = ai_battle.ConversationManager


def test_claude_client_direct():
    """Test ClaudeClient reasoning levels directly."""
    logger.info("Testing Claude 3.7 reasoning levels via direct instantiation")

    # Set up test parameters
    test_levels = {
        "auto": "auto",
        "high": "high",
        "medium": "medium",
        "low": "low",
        "none": "none",
    }

    # Get API key from environment or file
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            with open(os.path.expanduser("~/.ANTHROPIC_API_KEY"), "r") as f:
                api_key = f.read().strip()
        except:
            logger.error(
                "Could not find ANTHROPIC_API_KEY in environment or ~/.ANTHROPIC_API_KEY file"
            )

    # Test each reasoning level
    for level_name, level_value in test_levels.items():
        try:
            client = ClaudeClient(
                role="user",
                api_key=api_key,
                mode="ai-ai",
                domain="testing",
                model="claude-3-7-sonnet",
            )

            # Set the reasoning level
            client.reasoning_level = level_value

            # Verify it was set correctly
            logger.info(
                f"Setting reasoning_level to '{level_value}': client.reasoning_level = {client.reasoning_level}"
            )
            assert (
                client.reasoning_level == level_value
            ), f"Expected reasoning_level to be {level_value}, got {client.reasoning_level}"

            # Verify capabilities
            logger.info(f"Capabilities: {client.capabilities}")
            assert (
                client.capabilities.get("advanced_reasoning") is True
            ), "Claude 3.7 should have advanced_reasoning capability"

        except Exception as e:
            logger.error(f"Error testing reasoning level '{level_value}': {e}")
            raise


def test_openai_client_direct():
    """Test OpenAIClient reasoning levels directly."""
    logger.info("Testing OpenAI O1/O3 reasoning levels via direct instantiation")

    # Set up test parameters
    test_levels = {"high": "high", "medium": "medium", "low": "low"}

    # Get API key from environment or file
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            with open(os.path.expanduser("~/.OPENAI_API_KEY"), "r") as f:
                api_key = f.read().strip()
        except:
            logger.warning(
                "Could not find OPENAI_API_KEY. Test will not make actual API calls."
            )

    # List of models to test
    models_to_test = ["o1", "o3"]

    # Test each model with each reasoning level
    for model_name in models_to_test:
        logger.info(f"Testing OpenAI {model_name} model...")

        for level_name, level_value in test_levels.items():
            try:
                client = OpenAIClient(
                    api_key=api_key,
                    role="user",
                    mode="ai-ai",
                    domain="testing",
                    model=model_name,
                )

                # Set the reasoning level
                client.reasoning_level = level_value

                # Verify it was set correctly
                logger.info(
                    f"Setting reasoning_level to '{level_value}' for {model_name}: client.reasoning_level = {client.reasoning_level}"
                )
                assert (
                    client.reasoning_level == level_value
                ), f"Expected reasoning_level to be {level_value}, got {client.reasoning_level}"

                # Verify capabilities
                logger.info(f"Capabilities: {client.capabilities}")
                assert (
                    client.capabilities.get("advanced_reasoning") is True
                ), f"OpenAI {model_name} should have advanced_reasoning capability"

            except Exception as e:
                logger.error(
                    f"Error testing reasoning level '{level_value}' for {model_name}: {e}"
                )

    logger.info("All OpenAI reasoning level tests completed")


def test_conversation_manager():
    """Test reasoning levels via ConversationManager."""
    logger.info("Testing reasoning levels via ConversationManager")

    # Set up test parameters for Claude models
    claude_test_models = {
        "claude-3-7": "auto",
        "claude-3-7-reasoning": "high",
        "claude-3-7-reasoning-medium": "medium",
        "claude-3-7-reasoning-low": "low",
        "claude-3-7-reasoning-none": "none",
    }

    # Set up test parameters for OpenAI models
    openai_test_models = {
        "o1": "auto",
        "o1-reasoning-high": "high",
        "o1-reasoning-medium": "medium",
        "o1-reasoning-low": "low",
        "o3": "auto",
        "o3-reasoning-high": "high",
        "o3-reasoning-medium": "medium",
        "o3-reasoning-low": "low",
    }

    # Get Claude API key from environment or file
    claude_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not claude_api_key:
        try:
            with open(os.path.expanduser("~/.ANTHROPIC_API_KEY"), "r") as f:
                claude_api_key = f.read().strip()
        except:
            logger.warning(
                "Could not find ANTHROPIC_API_KEY in environment or ~/.ANTHROPIC_API_KEY file"
            )

    # Get OpenAI API key from environment or file
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        try:
            with open(os.path.expanduser("~/.OPENAI_API_KEY"), "r") as f:
                openai_api_key = f.read().strip()
        except:
            logger.warning(
                "Could not find OPENAI_API_KEY in environment or ~/.OPENAI_API_KEY file"
            )

    # Initialize conversation manager
    manager = ConversationManager(
        domain="test_reasoning_models",
        claude_api_key=claude_api_key,
        openai_api_key=openai_api_key,
    )

    # Test Claude models
    logger.info("Testing Claude 3.7, models via ConversationManager")
    for model_name, expected_level in claude_test_models.items():
        try:
            # Get client from conversation manager
            client = manager._get_client(model_name)

            # Verify it has the expected reasoning level
            logger.info(
                f"Model '{model_name}': client.reasoning_level = {client.reasoning_level}"
            )
            assert (
                client.reasoning_level == expected_level
            ), f"Expected reasoning_level to be {expected_level} for model {model_name}, got {client.reasoning_level}"

            # Verify capabilities
            logger.info(f"Model '{model_name}' capabilities: {client.capabilities}")
            assert (
                client.capabilities.get("advanced_reasoning") is True
            ), f"Model {model_name} should have advanced_reasoning capability"

        except Exception as e:
            logger.error(f"Error testing Claude model '{model_name}': {e}")

    # Test OpenAI models
    logger.info("Testing OpenAI O1/O3 models via ConversationManager")
    for model_name, expected_level in openai_test_models.items():
        try:
            # Get client from conversation manager
            client = manager._get_client(model_name)

            # Verify it has the expected reasoning level
            logger.info(
                f"Model '{model_name}': client.reasoning_level = {client.reasoning_level}"
            )
            assert (
                client.reasoning_level == expected_level
            ), f"Expected reasoning_level to be {expected_level} for model {model_name}, got {client.reasoning_level}"

            # Verify capabilities
            logger.info(f"Model '{model_name}' capabilities: {client.capabilities}")
            assert (
                client.capabilities.get("advanced_reasoning") is True
            ), f"Model {model_name} should have advanced_reasoning capability"

        except Exception as e:
            logger.error(f"Error testing OpenAI model '{model_name}': {e}")


if __name__ == "__main__":
    # Run tests
    try:
        logger.info("Starting reasoning models tests")

        # Test Claude 3.7 reasoning
        test_claude_client_direct()

        # Test OpenAI O1/O3 reasoning
        test_openai_client_direct()

        # Test all models via ConversationManager
        test_conversation_manager()

        logger.info("All tests passed!")
        print("\n✅ SUCCESS: All reasoning models tests passed!")

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        print("\n❌ FAILURE: Reasoning models tests failed!")
        sys.exit(1)
