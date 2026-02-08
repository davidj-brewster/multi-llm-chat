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

    def test_get_file_type_invalid(self):
        """Test file type detection for invalid extensions."""
        assert FileConfig.get_file_type("invalid.xyz") is None

    def test_can_handle_media(self):
        """Test media handling capability check."""
        assert FileConfig.can_handle_media("gemini-pro-vision", "image") is True
        assert FileConfig.can_handle_media("claude-3-sonnet", "image") is True
        assert FileConfig.can_handle_media("gpt-4", "video") is False


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
        """Test handling of oversized image - should be resized, not rejected."""
        metadata = media_handler.process_file(str(large_image))
        assert metadata is not None
        assert metadata.type == "image"
        # Image should have been resized to fit within max_image_resolution (1024x1024)
        assert metadata.dimensions[0] <= 1024
        assert metadata.dimensions[1] <= 1024

    def test_prepare_media_message(self, media_handler, test_image):
        """Test preparation of media message."""
        message = media_handler.prepare_media_message(
            str(test_image), conversation_context="Test context", role="user"
        )
        assert message is not None
        assert message["role"] == "user"
        assert len(message["content"]) == 2  # Context and image
        assert message["content"][0]["type"] == "text"
        assert message["content"][1]["type"] == "image"

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
