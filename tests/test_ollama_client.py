#!/usr/bin/env python3
"""
Test script for the OllamaClient implementation.

Tests basic connectivity, chat generation, model listing, and SDK 0.6.x
compatibility before and after the OllamaClient upgrade.

Usage:
    uv run python tests/test_ollama_client.py
    uv run pytest tests/test_ollama_client.py -v
"""

import os
import sys
import json
import logging

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, "src"))

from model_clients import OllamaClient, ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Test model - small, fast, should be available locally
TEST_MODEL = "gemma3:1b-it-q8_0"


def test_ollama_connection():
    """Test that we can connect to the local Ollama instance."""
    logger.info("Test 1: Ollama connection test")

    client = OllamaClient(
        mode="default",
        domain="Testing",
        model=TEST_MODEL,
    )

    try:
        client.test_connection()
        logger.info("PASS: Ollama connection successful")
        print("\n=== Connection Test: PASSED ===\n")
        return True
    except Exception as e:
        logger.error(f"FAIL: Ollama connection failed: {e}")
        print(f"\n=== Connection Test: FAILED ({e}) ===\n")
        return False


def test_ollama_basic_chat():
    """Test basic chat generation with a small model."""
    logger.info("Test 2: Basic chat generation")

    client = OllamaClient(
        mode="default",
        domain="Testing",
        model=TEST_MODEL,
    )

    prompt = 'Reply with exactly this JSON and nothing else: {"status": "ok", "model": "gemma3"}'
    system_instruction = "You are a helpful assistant. Always reply with valid JSON only, no markdown formatting, no explanation."

    response = client.generate_response(
        prompt=prompt,
        system_instruction=system_instruction,
        history=[],
        model_config=ModelConfig(temperature=0.1, max_tokens=100),
    )

    logger.info(f"Response: {response}")
    print(f"\n=== Basic Chat Test Result ===")
    print(f"Response: {response}")

    # Validate response
    assert response is not None, "Response should not be None"
    assert len(response) > 0, "Response should not be empty"
    assert "Error generating response" not in response, f"Got error response: {response}"

    # Check for expected content (model may not perfectly follow JSON instruction,
    # but should contain some recognizable content)
    response_lower = response.lower()
    assert any(
        keyword in response_lower for keyword in ["ok", "status", "gemma"]
    ), f"Response doesn't contain expected keywords: {response}"

    print("=== Basic Chat Test: PASSED ===\n")
    return response


def test_ollama_model_list():
    """Test model listing - validates SDK 0.6.x list() return type."""
    logger.info("Test 3: Model listing (SDK 0.6.x compatibility)")

    client = OllamaClient(
        mode="default",
        domain="Testing",
        model=TEST_MODEL,
    )

    model_list = client.client.list()
    logger.info(f"list() return type: {type(model_list)}")
    logger.info(f"list() dir: {[attr for attr in dir(model_list) if not attr.startswith('_')]}")

    # SDK 0.6.x returns a typed ListResponse object, not a dict
    # Check what attributes are available
    print(f"\n=== Model List Test ===")
    print(f"Return type: {type(model_list)}")

    # Try the new SDK 0.6.x way (typed object with .models attribute)
    models_found = []
    try:
        for model_info in model_list.models:
            # SDK 0.6.x uses .model attribute (not .name)
            model_name = model_info.model
            models_found.append(model_name)
            logger.debug(f"  Found model (new API): {model_name}")
        print(f"SDK 0.6.x API (.models/.model): Found {len(models_found)} models")
        print(f"Models: {models_found[:5]}{'...' if len(models_found) > 5 else ''}")
    except AttributeError as e:
        logger.warning(f"SDK 0.6.x typed access failed: {e}")
        print(f"SDK 0.6.x typed access FAILED: {e}")

    # Also try the old dict-style way to see if it still works
    old_style_works = False
    try:
        old_models = model_list.get('models', [])
        if old_models:
            old_name = old_models[0].get('name')
            old_style_works = True
            logger.info(f"Old dict-style API still works, first model: {old_name}")
            print(f"Old dict-style API: Still works (first model: {old_name})")
    except (AttributeError, TypeError) as e:
        logger.info(f"Old dict-style API no longer works: {e}")
        print(f"Old dict-style API: No longer works ({e})")

    assert len(models_found) > 0, "Should find at least one model"
    print(f"=== Model List Test: PASSED ===\n")

    return {
        "models": models_found,
        "old_style_works": old_style_works,
    }


def test_ollama_response_access():
    """Test response object access patterns - dict vs object style."""
    logger.info("Test 4: Response access patterns")

    client = OllamaClient(
        mode="default",
        domain="Testing",
        model=TEST_MODEL,
    )

    from ollama import ChatResponse

    # Make a direct chat call to inspect the raw response
    response = client.client.chat(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": "Say hello"}],
        stream=False,
    )

    logger.info(f"Response type: {type(response)}")
    print(f"\n=== Response Access Test ===")
    print(f"Response type: {type(response)}")

    # Test object-style access (SDK 0.6.x preferred)
    try:
        content = response.message.content
        print(f"Object access (response.message.content): '{content[:50]}...'")
        logger.info(f"Object access works: {content[:50]}")
    except AttributeError as e:
        print(f"Object access FAILED: {e}")
        logger.error(f"Object access failed: {e}")

    # Test dict-style access (old way)
    try:
        content = response['message']['content']
        print(f"Dict access (response['message']['content']): '{content[:50]}...'")
        logger.info(f"Dict access works: {content[:50]}")
    except (TypeError, KeyError) as e:
        print(f"Dict access FAILED: {e}")
        logger.info(f"Dict access no longer works: {e}")

    # Check for thinking field availability
    has_thinking = hasattr(response.message, 'thinking')
    print(f"response.message.thinking field exists: {has_thinking}")
    if has_thinking:
        thinking_value = response.message.thinking
        print(f"  thinking value: {thinking_value}")

    # Check message fields
    msg_fields = [attr for attr in dir(response.message) if not attr.startswith('_')]
    print(f"Message fields: {msg_fields}")

    print(f"=== Response Access Test: PASSED ===\n")
    return True


def test_ollama_thinking():
    """Test thinking/reasoning with a thinking-capable model (gpt-oss:20b)."""
    logger.info("Test 5: Thinking model (gpt-oss:20b)")

    client = OllamaClient(
        mode="default",
        domain="Testing",
        model="gpt-oss:20b",
    )

    # Verify capabilities were auto-detected
    assert client.capabilities.get("advanced_reasoning", False), \
        "gpt-oss should have advanced_reasoning capability"
    print(f"\n=== Thinking Model Test ===")
    print(f"Capabilities: {client.capabilities}")
    print(f"reasoning_level: {client.reasoning_level}")
    print(f"extended_thinking: {client.extended_thinking}")

    # Enable thinking (as ai_battle.py would do via OLLAMA_THINKING_CONFIG)
    client.reasoning_level = "high"
    client.set_extended_thinking(True)

    prompt = "What is 15 * 37? Show your work step by step."
    system_instruction = "You are a math tutor. Think through problems carefully."

    response = client.generate_response(
        prompt=prompt,
        system_instruction=system_instruction,
        history=[],
        model_config=ModelConfig(temperature=0.3, max_tokens=500),
    )

    logger.info(f"Response length: {len(response)} chars")
    print(f"Response ({len(response)} chars):")

    # Check if thinking content was returned
    has_thinking = "<thinking>" in response
    print(f"Has <thinking> tags: {has_thinking}")

    if has_thinking:
        # Extract just the thinking portion for display
        import re
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
        if thinking_match:
            thinking_text = thinking_match.group(1)[:200]
            print(f"Thinking preview: {thinking_text}...")

    # Show main content (after thinking tags)
    import re
    main_content = re.sub(r'<thinking>.*?</thinking>\s*', '', response, flags=re.DOTALL)
    print(f"Main content: {main_content[:300]}")

    assert response is not None, "Response should not be None"
    assert len(response) > 0, "Response should not be empty"
    assert "555" in response, f"Expected 555 (15*37) in response: {response[:200]}"

    print(f"=== Thinking Model Test: PASSED ===\n")
    return response


def test_ollama_vision():
    """Test vision capability with gemma3 and a test image."""
    logger.info("Test 6: Vision model (gemma3:4b-it-q8_0)")

    client = OllamaClient(
        mode="default",
        domain="Testing",
        model="gemma3:4b-it-q8_0",
    )

    # Verify vision capability was auto-detected
    assert client.capabilities.get("vision", False), \
        "gemma3 should have vision capability"
    print(f"\n=== Vision Model Test ===")
    print(f"Vision capability: {client.capabilities.get('vision')}")

    # Create a valid test image using PIL
    import base64
    from io import BytesIO
    from PIL import Image

    # Create a 64x64 red square image
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    file_data = {
        "type": "image",
        "base64": image_base64,
        "mime_type": "image/png",
    }

    prompt = "What color is this image? Reply with just the color name."
    system_instruction = "You are an image analysis assistant. Be brief and direct."

    response = client.generate_response(
        prompt=prompt,
        system_instruction=system_instruction,
        history=[],
        file_data=file_data,
        model_config=ModelConfig(temperature=0.1, max_tokens=100),
    )

    logger.info(f"Vision response: {response}")
    print(f"Response: {response}")

    assert response is not None, "Response should not be None"
    assert len(response) > 0, "Response should not be empty"
    assert "Error generating response" not in response, f"Got error: {response}"

    print(f"=== Vision Model Test: PASSED ===\n")
    return response


def main():
    """Run all OllamaClient tests."""
    logger.info("Starting OllamaClient tests")
    print("=" * 60)
    print("OllamaClient Tests (SDK 0.6.x)")
    print("=" * 60)

    results = {}

    # Test 1: Connection
    results["connection"] = test_ollama_connection()
    if not results["connection"]:
        print("\nConnection failed - cannot proceed with remaining tests.")
        print("Make sure Ollama is running: ollama serve")
        return results

    # Test 2: Basic chat
    try:
        results["basic_chat"] = test_ollama_basic_chat() is not None
    except AssertionError as e:
        results["basic_chat"] = False
        logger.error(f"Basic chat test failed: {e}")
    except Exception as e:
        results["basic_chat"] = False
        logger.error(f"Basic chat test error: {e}")

    # Test 3: Model listing
    try:
        list_result = test_ollama_model_list()
        results["model_list"] = list_result is not None
        results["old_style_works"] = list_result.get("old_style_works", False) if list_result else False
    except Exception as e:
        results["model_list"] = False
        logger.error(f"Model list test error: {e}")

    # Test 4: Response access patterns
    try:
        results["response_access"] = test_ollama_response_access()
    except Exception as e:
        results["response_access"] = False
        logger.error(f"Response access test error: {e}")

    # Test 5: Thinking model (requires gpt-oss:20b)
    try:
        results["thinking"] = test_ollama_thinking() is not None
    except AssertionError as e:
        results["thinking"] = False
        logger.error(f"Thinking test failed: {e}")
    except Exception as e:
        results["thinking"] = False
        logger.error(f"Thinking test error: {e}")

    # Test 6: Vision model (requires gemma3:4b-it-q8_0)
    try:
        results["vision"] = test_ollama_vision() is not None
    except AssertionError as e:
        results["vision"] = False
        logger.error(f"Vision test failed: {e}")
    except Exception as e:
        results["vision"] = False
        logger.error(f"Vision test error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
    print("=" * 60)

    all_passed = all(results.values())
    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed - review output above for details")

    return results


if __name__ == "__main__":
    main()
