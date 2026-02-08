#!/usr/bin/env python3
"""
Test script for the OpenAI responses API implementation.

This script creates an instance of the OpenAIClient with the responses API
and tests basic text generation as well as handling of multiple images.
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

# Import required modules
from model_clients import OpenAIClient, ModelConfig

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
def image_paths(tmp_path):
    """Provide a list of temporary test image file paths."""
    paths = []
    for i in range(2):
        p = tmp_path / f"test_image_{i}.png"
        _create_minimal_png(str(p))
        paths.append(str(p))
    return paths


@skip_no_api_key
def test_text_generation():
    """Test basic text generation with the responses API."""
    logger.info("Testing text generation with responses API")

    # Create an OpenAI client
    client = OpenAIClient(
        mode="default",
        role="assistant",
        api_key=os.environ.get("OPENAI_API_KEY"),
        domain="Testing",
        model="gpt-4o",  # Use a model compatible with responses API
    )

    # Generate a response
    prompt = (
        "Write a very short three-sentence story about a unicorn for a bedtime story."
    )
    system_instruction = (
        "You are a helpful assistant that specializes in short, creative stories."
    )

    response = client.generate_response(
        prompt=prompt,
        system_instruction=system_instruction,
        history=[],
        model_config=ModelConfig(temperature=0.7, max_tokens=300),
    )

    logger.info(f"Response from API: {response}")
    print("\n=== Text Generation Test Result ===\n")
    print(response)
    print("\n===============================\n")

    return response


@skip_no_api_key
def test_image_analysis(image_paths):
    """Test image analysis with the responses API."""
    if not image_paths:
        logger.warning("No image paths provided. Skipping image analysis test.")
        return

    logger.info(
        f"Testing image analysis with responses API using {len(image_paths)} images"
    )

    # Create an OpenAI client
    client = OpenAIClient(
        mode="default",
        role="assistant",
        api_key=os.environ.get("OPENAI_API_KEY"),
        domain="Testing",
        model="gpt-4o",  # Use a model compatible with responses API
    )

    # Read and encode images
    file_data = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue

        try:
            with open(image_path, "rb") as f:
                image_content = f.read()
                file_item = {
                    "type": "image",
                    "path": image_path,
                    "base64": base64.b64encode(image_content).decode("utf-8"),
                    "mime_type": (
                        "image/jpeg"
                        if image_path.lower().endswith((".jpg", ".jpeg"))
                        else "image/png"
                    ),
                }
                file_data.append(file_item)
            logger.info(f"Added image: {image_path}")
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {e}")

    if not file_data:
        logger.warning("No valid images found. Skipping image analysis test.")
        return

    # Generate a response with images
    prompt = f"Describe what you see in {'these images' if len(file_data) > 1 else 'this image'}. Be brief but descriptive."
    system_instruction = (
        "You are a helpful assistant that specializes in image analysis."
    )

    response = client.generate_response(
        prompt=prompt,
        system_instruction=system_instruction,
        history=[],
        file_data=file_data,
        model_config=ModelConfig(temperature=0.7, max_tokens=500),
    )

    logger.info(f"Response from API: {response}")
    print("\n=== Image Analysis Test Result ===\n")
    print(response)
    print("\n===============================\n")

    return response


@skip_no_api_key
def test_chunking():
    """Test text chunking with a large input."""
    logger.info("Testing text chunking with responses API")

    # Create an OpenAI client
    client = OpenAIClient(
        mode="default",
        role="assistant",
        api_key=os.environ.get("OPENAI_API_KEY"),
        domain="Testing",
        model="gpt-4o",  # Use a model compatible with responses API
    )

    # Generate a large text input (about 150K characters)
    large_text = "This is a test of text chunking. " * 5000

    # Generate a response
    prompt = f"Summarize the following text in 3 bullet points:\n\n{large_text}"
    system_instruction = (
        "You are a helpful assistant that specializes in summarization."
    )

    response = client.generate_response(
        prompt=prompt,
        system_instruction=system_instruction,
        history=[],
        model_config=ModelConfig(temperature=0.7, max_tokens=300),
    )

    logger.info(f"Response from API: {response}")
    print("\n=== Text Chunking Test Result ===\n")
    print(response)
    print("\n===============================\n")

    return response


def main():
    """Run tests for the OpenAI responses API."""
    logger.info("Starting OpenAI responses API tests")

    # Get image paths from command line arguments
    image_paths = sys.argv[1:] if len(sys.argv) > 1 else []

    # Test text generation
    test_text_generation()

    # Test text chunking
    test_chunking()

    # Test image analysis if image paths provided
    if image_paths:
        test_image_analysis(image_paths)
    else:
        logger.info("No image paths provided. Skipping image analysis test.")
        print(
            "\nTo test image analysis, run: python test_openai_responses.py path/to/image1.jpg path/to/image2.jpg"
        )

    logger.info("OpenAI responses API tests completed")


if __name__ == "__main__":
    main()
