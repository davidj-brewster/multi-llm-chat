"""Unit tests for file_handler module."""

import os
import pytest
from pathlib import Path
from PIL import Image
import io
from file_handler import (
    FileMetadata,
    FileConfig,
    ConversationMediaHandler,
    MediaProcessingError,
    UnsupportedMediaTypeError,
    MediaValidationError,
)


@pytest.fixture
def test_output_dir(tmp_path):
    """Create temporary directory for test outputs."""
    return tmp_path / "test_output"


@pytest.fixture
def media_handler(test_output_dir):
    """Create ConversationMediaHandler instance for testing."""
    return ConversationMediaHandler(str(test_output_dir))


@pytest.fixture
def test_image(tmp_path):
    """Create a test image file."""
    image_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(image_path)
    return image_path


@pytest.fixture
def test_text_file(tmp_path):
    """Create a test text file."""
    text_path = tmp_path / "test.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("Test content")
    return text_path


@pytest.fixture
def large_image(tmp_path):
    """Create a test image that exceeds size limits."""
    image_path = tmp_path / "large.jpg"
    img = Image.new("RGB", (10000, 10000), color="blue")
    img.save(image_path)
    return image_path


class TestFileConfig:
    """Tests for FileConfig class."""

    def test_get_file_type_valid(self):
        """Test file type detection for valid extensions."""
        assert FileConfig.get_file_type("test.jpg") == "image"
        assert FileConfig.get_file_type("doc.txt") == "text"
        assert FileConfig.get_file_type("video.mp4") == "video"
        assert FileConfig.get_file_type("script.py") == "code"

    def test_get_file_type_invalid(self):
        """Test file type detection for invalid extensions."""
        assert FileConfig.get_file_type("invalid.xyz") is None

    def test_can_handle_media(self):
        """Test media handling capability check for FileConfig.can_handle_media."""

        # --- Image Tests ---
        # Specific supported models for images
        assert FileConfig.can_handle_media("gemini-pro-vision", "image") is True
        assert FileConfig.can_handle_media("gemini-2-pro", "image") is True # Example, adjust if needed
        assert FileConfig.can_handle_media("claude-3-sonnet", "image") is True
        assert FileConfig.can_handle_media("claude-3-opus", "image") is True
        assert FileConfig.can_handle_media("gpt-4-vision", "image") is True
        assert FileConfig.can_handle_media("gpt-4o", "image") is True

        # Provider keyword matching for images
        assert FileConfig.can_handle_media("gemini-1.5-pro", "image") is True # "gemini" keyword
        assert FileConfig.can_handle_media("claude-instant", "image") is True # "claude" keyword
        assert FileConfig.can_handle_media("chatgpt-turbo", "image") is True # "chatgpt" keyword

        # Case insensitivity for images
        assert FileConfig.can_handle_media("GEMINI-PRO-VISION", "image") is True
        assert FileConfig.can_handle_media("Claude-3-Haiku", "image") is True

        # Model not supporting images (e.g., a text-only Gemini model for image file type)
        assert FileConfig.can_handle_media("gemini-pro", "image") is False # Assuming gemini-pro is text only for this test against 'image'

        # --- Video Tests ---
        # Specific supported models for video
        assert FileConfig.can_handle_media("gemini-pro-vision", "video") is True

        # Model not supporting video
        assert FileConfig.can_handle_media("claude-3-opus", "video") is False
        assert FileConfig.can_handle_media("gpt-4", "video") is False
        assert FileConfig.can_handle_media("gemini-pro", "video") is False # Text-only gemini

        # --- Text Tests ---
        # Text files have no "supported_models" key, implying universal support
        assert FileConfig.can_handle_media("any-model-whatsoever", "text") is True
        assert FileConfig.can_handle_media("gemini-pro-vision", "text") is True
        assert FileConfig.can_handle_media("claude-3-sonnet", "text") is True
        assert FileConfig.can_handle_media("gpt-4o", "text") is True
        assert FileConfig.can_handle_media("ollama-llama2", "text") is True

        # --- Code Tests ---
        # Specific supported models for code
        assert FileConfig.can_handle_media("gemini-pro", "code") is True
        assert FileConfig.can_handle_media("claude-3-opus", "code") is True
        assert FileConfig.can_handle_media("gpt-4", "code") is True
        assert FileConfig.can_handle_media("ollama-llava", "code") is True # Assuming llava is in the ollama list for code
        assert FileConfig.can_handle_media("ollama-gemma3", "code") is True # Assuming gemma3 is in the ollama list

        # Provider keyword matching for code
        assert FileConfig.can_handle_media("gemini-1.5-flash", "code") is True # "gemini" keyword
        assert FileConfig.can_handle_media("claude-3-5-sonnet-extension", "code") is True # "claude" keyword
        assert FileConfig.can_handle_media("gpt4.1-mini-code", "code") is True # "gpt" keyword

        # Model that doesn't support code (e.g., a vision-only model if one existed and was distinct)
        # For this, we'll test a model from a known provider that isn't in the code list
        # Example: if "gemini-nano" is not listed for code, this should be False
        # This depends on the exact content of SUPPORTED_FILE_TYPES["code"]["supported_models"]["gemini"]
        # Assuming "gemini-nano" is NOT in the list for code:
        # assert FileConfig.can_handle_media("gemini-nano", "code") is False

        # Ollama model not explicitly listed for code (but provider is known)
        # If ollama list for code is empty (meaning all ollama models support code), this would be True.
        # If ollama list for code is specific, e.g. ["llava", "gemma3", "phi4"], then "ollama-mistral" for "code" should be False.
        # Based on current FileConfig, "ollama": ["llava", "gemma3", "phi4"] for code.
        assert FileConfig.can_handle_media("ollama-mistral", "code") is False
        assert FileConfig.can_handle_media("ollama-llama3", "code") is False
        assert FileConfig.can_handle_media("ollama-phi4", "code") is True # Explicitly listed

        # --- General / Edge Cases ---
        # Unknown file type
        assert FileConfig.can_handle_media("gemini-pro-vision", "unknown_file_type") is False

        # Unknown model name for a known file type
        assert FileConfig.can_handle_media("unknown-model-9000", "image") is False
        assert FileConfig.can_handle_media("super-claude-x", "image") is False # "claude" keyword but not specific enough if lists are restrictive

        # Empty or None inputs
        assert FileConfig.can_handle_media(None, "image") is False
        assert FileConfig.can_handle_media("gemini-pro-vision", None) is False
        assert FileConfig.can_handle_media(None, None) is False
        assert FileConfig.can_handle_media("", "image") is False
        assert FileConfig.can_handle_media("gemini-pro-vision", "") is False
        assert FileConfig.can_handle_media("", "") is False

        # Model from a known provider, but file type doesn't list that provider
        # Example: if "openai" is not listed under "video"
        assert FileConfig.can_handle_media("gpt-4o", "video") is False # Correct, as OpenAI not in video's supported_models

        # Check a model known to support a file type against another file type it doesn't support.
        # claude-3-opus supports 'image' and 'code', but not 'video'.
        assert FileConfig.can_handle_media("claude-3-opus", "image") is True
        assert FileConfig.can_handle_media("claude-3-opus", "code") is True
        assert FileConfig.can_handle_media("claude-3-opus", "video") is False


class TestConversationMediaHandler:
    """Tests for ConversationMediaHandler class."""

    def test_process_image_file(self, media_handler, test_image):
        """Test processing of valid image file."""
        metadata = media_handler.process_file(str(test_image))
        assert metadata is not None
        assert metadata.type == "image"
        assert metadata.dimensions == (100, 100)
        assert metadata.thumbnail_path is not None
        assert Path(metadata.thumbnail_path).exists()

    def test_process_text_file(self, media_handler, test_text_file):
        """Test processing of valid text file."""
        metadata = media_handler.process_file(str(test_text_file))
        assert metadata is not None
        assert metadata.type == "text"
        assert metadata.text_content == "Test content"

    def test_process_nonexistent_file(self, media_handler):
        """Test handling of non-existent file."""
        with pytest.raises(MediaValidationError):
            media_handler.process_file("nonexistent.jpg")

    def test_process_unsupported_file(self, media_handler, tmp_path):
        """Test handling of unsupported file type."""
        unsupported = tmp_path / "test.xyz"
        unsupported.touch()
        with pytest.raises(UnsupportedMediaTypeError):
            media_handler.process_file(str(unsupported))

    def test_process_large_image(self, media_handler, large_image):
        """Test handling of oversized image."""
        with pytest.raises(MediaValidationError):
            media_handler.process_file(str(large_image))

    def test_prepare_media_message(self, media_handler, test_image):
        """Test preparation of media message."""
        message = media_handler.prepare_media_message(
            str(test_image), conversation_context="Test context", role="user"
        )
        assert message is not None
        assert message["role"] == "user"
        # The content structure depends on the prepare_media_message implementation
        # For images, it usually includes text context and then image data
        assert any(item["type"] == "text" and item["text"] == "Test context" for item in message["content"])
        assert any(item["type"] == "image" for item in message["content"])


    def test_create_media_prompt(self, media_handler, test_image):
        """Test creation of media analysis prompt."""
        metadata = media_handler.process_file(str(test_image))
        prompt = media_handler.create_media_prompt(
            metadata, context="Test analysis", task="analyze"
        )
        assert "Test analysis" in prompt
        assert "100x100 pixels" in prompt
        assert "analyze" in prompt.lower()


def test_file_metadata_creation():
    """Test FileMetadata dataclass."""
    metadata = FileMetadata(
        path="test.jpg",
        type="image",
        size=1000,
        mime_type="image/jpeg",
        dimensions=(100, 100),
    )
    assert metadata.path == "test.jpg"
    assert metadata.type == "image"
    assert metadata.size == 1000
    assert metadata.dimensions == (100, 100)
    assert metadata.duration is None
    assert metadata.text_content is None
    assert metadata.thumbnail_path is None
