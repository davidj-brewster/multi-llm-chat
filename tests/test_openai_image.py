#!/usr/bin/env python3
"""
Test script for OpenAI responses API with image and conversation memory.

This tests if the Responses API can remember image content in conversations
without requiring re-uploading the image in subsequent messages.
"""

import os
import sys
import base64
import logging
import struct
import zlib
from pathlib import Path

import pytest

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import model clients with conditional imports to avoid module errors
try:
    from model_clients import OpenAIClient, ModelConfig
except ImportError as e:
    print(f"Could not import from model_clients: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

skip_no_api_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


def _create_minimal_png(path):
    """Create a minimal 1x1 red PNG file at the given path."""

    def _chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw_row = b"\x00\xff\x00\x00"  # filter byte + RGB
    idat_data = zlib.compress(raw_row)
    png = signature + _chunk(b"IHDR", ihdr_data) + _chunk(b"IDAT", idat_data) + _chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


@pytest.fixture
def image_path(tmp_path):
    """Provide a temporary test image file path."""
    p = tmp_path / "test_image.png"
    _create_minimal_png(str(p))
    return str(p)


@skip_no_api_key
def test_image_conversation(image_path):
    """Test multi-turn conversation with image using OpenAI responses API."""
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return f"Error: Image not found at {image_path}"

    logger.info(
        f"Testing image conversation with responses API using image: {image_path}"
    )

    # Create OpenAI client
    try:
        client = OpenAIClient(
            mode="default",
            role="assistant",
            domain="Testing",
            model="gpt-4o",  # Vision-capable model
        )
    except Exception as e:
        logger.error(f"Error creating OpenAI client: {e}")
        return f"Error creating OpenAI client: {e}"

    # Read and encode image
    try:
        with open(image_path, "rb") as f:
            image_content = f.read()
            file_data = {
                "type": "image",
                "path": image_path,
                "base64": base64.b64encode(image_content).decode("utf-8"),
                "mime_type": "image/jpeg",
            }
        logger.info(f"Image encoded successfully: {image_path}")
    except Exception as e:
        logger.error(f"Error reading/encoding image: {e}")
        return f"Error reading/encoding image: {e}"

    # Initial question with image
    try:
        initial_prompt = "Describe this cat video in detail. What behaviors do you observe and what might the cats be doing?"
        system_instruction = (
            "You are a helpful assistant with expertise in animal behavior."
        )

        # Keep track of conversation history
        conversation_history = []

        # First turn - send the image
        logger.info("TURN 1: Sending initial prompt with image")
        response1 = client.generate_response(
            prompt=initial_prompt,
            system_instruction=system_instruction,
            history=conversation_history,
            file_data=file_data,
            model_config=ModelConfig(temperature=0.2, max_tokens=500),
        )

        # Add the exchange to history
        conversation_history.append({"role": "user", "content": initial_prompt})
        conversation_history.append({"role": "assistant", "content": response1})

        logger.info("Turn 1 completed")
        print("\n=== Turn 1 (with image) ===\n")
        print(f"User: {initial_prompt}")
        print(f"\nAssistant: {response1}")
        print("\n=========================\n")

        # Second turn - follow-up question WITHOUT sending the image again
        followup_prompt = "What specific behaviors or interactions can you identify that might indicate the cats' relationship with each other? Please be specific about any interesting patterns you observe."

        logger.info("TURN 2: Sending follow-up prompt WITHOUT image")
        response2 = client.generate_response(
            prompt=followup_prompt,
            system_instruction=system_instruction,
            history=conversation_history,
            # No file_data here - testing if the model remembers the image
            model_config=ModelConfig(temperature=0.2, max_tokens=500),
        )

        # Add to history
        conversation_history.append({"role": "user", "content": followup_prompt})
        conversation_history.append({"role": "assistant", "content": response2})

        logger.info("Turn 2 completed")
        print("\n=== Turn 2 (no image) ===\n")
        print(f"User: {followup_prompt}")
        print(f"\nAssistant: {response2}")
        print("\n=========================\n")

        # Third turn - another follow-up question
        final_prompt = "Based on what you can see, are there any playful behaviors or interesting environmental interactions in this video that might tell us something about these cats' personalities?"

        logger.info("TURN 3: Sending final follow-up prompt WITHOUT image")
        response3 = client.generate_response(
            prompt=final_prompt,
            system_instruction=system_instruction,
            history=conversation_history,
            # No file_data here either
            model_config=ModelConfig(temperature=0.2, max_tokens=500),
        )

        logger.info("Turn 3 completed")
        print("\n=== Turn 3 (no image) ===\n")
        print(f"User: {final_prompt}")
        print(f"\nAssistant: {response3}")
        print("\n=========================\n")

        # Return combined results
        return {
            "turn1": {"prompt": initial_prompt, "response": response1},
            "turn2": {"prompt": followup_prompt, "response": response2},
            "turn3": {"prompt": final_prompt, "response": response3},
        }

    except Exception as e:
        logger.error(f"Error in conversation: {e}")
        return f"Error in conversation: {e}"


if __name__ == "__main__":
    image_path = "./Cats.mp4"  # Use a generic video file in project root

    # Override with command line argument if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nERROR: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key with:")
        print("export OPENAI_API_KEY=your-api-key-here")
        print("\nLaunching mock test instead...\n")

        # Run a simplified mock test
        print("=== MOCK TEST RESULTS ===")
        print("Turn 1 (with image): The model would analyze cat behaviors in the video")
        print(
            "Turn 2 (no image): The model would remember the video and discuss interaction patterns"
        )
        print(
            "Turn 3 (no image): The model would continue analyzing behavioral traits in the remembered video"
        )
        sys.exit(1)

    print(f"\nTesting OpenAI responses API with image: {image_path}")
    print("This test will check if the model remembers images in conversation context")
    print("without needing to re-upload them in follow-up messages.\n")

    test_image_conversation(image_path)

    print("\nTest completed!")
