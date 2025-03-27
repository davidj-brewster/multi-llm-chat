#!/usr/bin/env python3
"""
A simplified mock test for OpenAI responses API implementation.
"""

import os
import sys
from unittest.mock import Mock, MagicMock
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


# Mock the key parts of OpenAI client
class MockOpenAI:
    """Mock OpenAI client for testing."""

    def __init__(self, api_key=None):
        self.responses = MockResponses()


class MockResponses:
    """Mock responses API endpoint."""

    def __init__(self):
        pass

    def create(
        self,
        model=None,
        system=None,
        input=None,
        max_tokens=None,
        temperature=None,
        **kwargs,
    ):
        """Mock create method that returns a predefined response."""
        logger.info(f"MockResponses.create called with model={model}")
        logger.info(f"System prompt: {system}")
        logger.info(
            f"Input: {input[:100] if isinstance(input, list) and input else input}"
        )

        # Mock response structure
        mock_response = MagicMock()

        # Create a mock message that simulates an assistant response
        mock_message = MagicMock()
        mock_message.role = "assistant"

        # Create a mock content item with text
        mock_content_item = MagicMock()
        mock_content_item.type = "text"
        mock_content_item.text = (
            "This is a mock response from the OpenAI responses API."
        )

        # Assemble the response structure
        mock_message.content = [mock_content_item]
        mock_response.data = [mock_message]

        return mock_response


def test_mock_responses_api():
    """Test the mock responses API."""
    # Create a mock OpenAI client
    client = MockOpenAI(api_key="mock-api-key")

    # Test basic text generation
    response = client.responses.create(
        model="gpt-4o",
        system="You are a helpful assistant",
        input=[{"type": "text", "text": "Write a short story about a unicorn."}],
        max_tokens=100,
        temperature=0.7,
    )

    # Extract the response text
    assistant_text = None
    for message in response.data:
        if message.role == "assistant":
            for content_item in message.content:
                if content_item.type == "text":
                    assistant_text = content_item.text

    print("\n=== Mock Response API Test Result ===\n")
    print(f"Response: {assistant_text}")
    print("\n=====================================\n")

    return assistant_text


def main():
    """Run the mock test for the OpenAI responses API."""
    logger.info("Starting mock test for OpenAI responses API")

    # Test the mock responses API
    test_mock_responses_api()

    logger.info("Mock test completed")


if __name__ == "__main__":
    main()
