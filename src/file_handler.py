"""
File and media handling integrated with conversation flow.

This module provides comprehensive handling of various media types (images, videos, text)
within a conversation context. It includes validation, processing, and preparation of media
for model consumption.

Key components:
- FileMetadata: Data structure for file information
- FileConfig: Configuration and validation rules
- ConversationMediaHandler: Main class for media processing
"""

import logging
import mimetypes
import os
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

from PIL import Image
from PIL import UnidentifiedImageError

logger = logging.getLogger(__name__)


class MediaProcessingError(Exception):
    """Base exception for media processing errors."""


class UnsupportedMediaTypeError(MediaProcessingError):
    """Raised when file type is not supported."""


class MediaValidationError(MediaProcessingError):
    """Raised when file fails validation checks."""


class MediaProcessingConfigError(MediaProcessingError):
    """Raised when there's a configuration-related error."""


class ImageProcessingError(MediaProcessingError):
    """Raised when there's an error processing an image."""


class VideoProcessingError(MediaProcessingError):
    """Raised when there's an error processing a video."""


class TextProcessingError(MediaProcessingError):
    """Raised when there's an error processing a text file."""


class CodeProcessingError(MediaProcessingError):
    """Raised when there's an error processing a code file."""


class FileIOError(MediaProcessingError):
    """Raised when there's an error reading from or writing to a file."""


@dataclass
class FileMetadata:
    """
    Metadata container for processed files.

    Attributes:
        path (str): File system path to the media file
        type (str): Type of media (image, video, text)
        size (int): File size in bytes
        mime_type (str): MIME type of the file
        dimensions (Optional[Tuple[int, int]]): Width and height for images/videos
        duration (Optional[float]): Duration in seconds for video files
        text_content (Optional[str]): Extracted text content for text files
        thumbnail_path (Optional[str]): Path to generated thumbnail if applicable
        processed_video (Optional[Dict]): Information about processed video (fps, resolution, frames)
    """

    path: str
    type: str
    size: int
    mime_type: str
    dimensions: Optional[Tuple[int, int]] = None
    duration: Optional[float] = None
    text_content: Optional[str] = None
    thumbnail_path: Optional[str] = None
    processed_video: Optional[Dict] = None


class FileConfig:
    """
    Configuration for file type handling and validation.

    This class defines supported file types, their constraints, and compatibility
    with different AI models. It provides methods for file type detection and
    model compatibility checking.
    """

    SUPPORTED_FILE_TYPES = {
        "image": {
            "extensions": [".jpg", ".jpeg", ".png", ".gif", ".webp"],
            "max_size": 10 * 1024 * 1024,  # 10MB
            "max_resolution": (8192, 8192),
            "supported_models": {
                "gemini": [
                    "gemini-pro-vision",
                    "gemini-2-pro",
                    "gemini-2-reasoning",
                    "gemini-2-flash-lite",
                    "gemini-2.0-flash-exp",
                    "gemini",
                ],
                "claude": [
                    "claude",
                    "sonnet",
                    "haiku",
                    "claude-3-sonnet",
                    "claude-3-haiku",
                    "claude-3-opus",
                    "claude-3-5-sonnet",
                    "claude-3-5-haiku",
                    "claude-3-7",
                    "claude-3-7-sonnet",
                    "claude-3-7-reasoning",
                    "claude-3-7-reasoning-medium",
                    "claude-3-7-reasoning-low",
                    "claude-3-7-reasoning-none",
                ],
                "openai": ["gpt-4-vision", "gpt-4o", "o1", "chatgpt-latest"],
            },
        },
        "video": {
            "extensions": [".mp4", ".mov", ".avi", ".webm"],
            "max_size": 300 * 1024 * 1024,  # 300MB
            "max_resolution": (3840, 2160),  # 4K
            "supported_models": {"gemini": ["gemini-pro-vision"]},
        },
        "text": {
            "extensions": [".txt", ".md", ".csv", ".json", ".yaml", ".yml"],
            "max_size": 20 * 1024 * 1024,  # 20MB
        },
        "code": {
            "extensions": [
                ".py",
                ".js",
                ".html",
                ".css",
                ".java",
                ".cpp",
                ".c",
                ".h",
                ".cs",
                ".php",
                ".rb",
                ".go",
                ".rs",
                ".ts",
                ".swift",
            ],
            "max_size": 5 * 1024 * 1024,  # 5MB
            "supported_models": {
                "gemini": ["gemini-pro", "gemini-pro-vision"],
                "claude": [
                    "claude-3-sonnet",
                    "claude-3-haiku",
                    "claude-3-opus",
                    "claude-3-7-sonnet",
                    "claude-3-5-haiku",
                    "claude-3-7",
                    "claude-3-7-reasoning",
                    "claude-3-7-reasoning-medium",
                    "claude-3-7-reasoning-low",
                    "claude-3-7-reasoning-none",
                ],
                "openai": ["gpt-4", "gpt-4o","gpt4.1", "gpt4.1-mini", "gpt4.1-nano"],
                "ollama": ["llava", "gemma3", "phi4"],
            },
        },
    }

    @classmethod
    def get_file_type(cls, file_path: str) -> Optional[str]:
        """
        Determine file type from extension.

        Args:
            file_path: Path to the file

        Returns:
            Optional[str]: File type if supported, None otherwise
        """
        ext = os.path.splitext(file_path)[1].lower()
        for file_type, config in cls.SUPPORTED_FILE_TYPES.items():
            if ext in config["extensions"]:
                return file_type
        logger.debug(f"Unsupported file extension: {ext}")
        return None

    @classmethod
    def can_handle_media(cls, model_name: str, file_type: str) -> bool:
        """
        Check if model can handle media type.

        Args:
            model_name: Name of the AI model
            file_type: Type of media to check

        Returns:
            bool: True if model supports the media type
        """
        if file_type not in cls.SUPPORTED_FILE_TYPES:
            logger.debug(f"Unsupported file type: {file_type}")
            return False

        config = cls.SUPPORTED_FILE_TYPES[file_type]
        if "supported_models" not in config:
            return False

        # Map model names to providers
        provider_map = {
            "claude": "claude",
            "sonnet": "claude",
            "haiku": "claude",
            "gemini": "gemini",
            "flash": "gemini",
            "gpt": "openai",
            "chatgpt": "openai",
            "o1": "openai",
        }

        # Get provider from model name
        provider = next(
            (p for k, p in provider_map.items() if k in model_name.lower()), None
        )
        if not provider:
            return False

        return provider in config["supported_models"]


class ConversationMediaHandler:
    """
    Handles media analysis within conversation context.

    This class provides functionality for processing media files, preparing them
    for model consumption, and generating context-aware prompts for analysis.

    Args:
        output_dir: Directory for processed files and thumbnails
    """

    def __init__(self, output_dir: str = "processed_files"):
        self.output_dir = Path(output_dir)
        self.max_image_resolution = (1024, 1024)  # Default max resolution for images
        self.output_dir.mkdir(exist_ok=True)
        logger.info(
            f"Initialized ConversationMediaHandler with output directory: {output_dir}"
        )

    def process_directory(
        self, directory_path: str, file_pattern: str = None, max_files: int = 10
    ) -> List[FileMetadata]:
        """
        Process all files in a directory that match the pattern.

        Args:
            directory_path: Path to the directory
            file_pattern: Optional glob pattern to filter files (e.g., "*.jpg")
            max_files: Maximum number of files to process

        Returns:
            List[FileMetadata]: List of metadata for processed files

        Raises:
            MediaValidationError: If directory does not exist or is not accessible
            MediaProcessingError: For general processing errors
        """
        try:
            directory_path = Path(directory_path)
            if not directory_path.exists():
                raise MediaValidationError(f"Directory not found: {directory_path}")

            if not directory_path.is_dir():
                raise MediaValidationError(f"Path is not a directory: {directory_path}")

            # Get list of files matching the pattern
            if file_pattern:
                file_paths = list(directory_path.glob(file_pattern))
            else:
                file_paths = [p for p in directory_path.iterdir() if p.is_file()]

            # Sort files by name for consistent ordering
            file_paths.sort()

            # Limit the number of files
            if max_files > 0 and len(file_paths) > max_files:
                logger.warning(
                    f"Directory contains {len(file_paths)} files, limiting to {max_files}"
                )
                file_paths = file_paths[:max_files]

            # Process each file
            return self.process_multiple_files([str(p) for p in file_paths])[
                0
            ]  # Return successful files only

        except MediaValidationError as e:
            logger.error(f"Directory validation error: {e}")
            raise
        except Exception as e:
            logger.exception(f"Error processing directory {directory_path}")
            raise MediaProcessingError(
                f"Failed to process directory {directory_path}: {e}"
            ) from e

    def process_multiple_files(
        self, file_paths: List[str]
    ) -> Tuple[List[FileMetadata], List[Dict[str, Any]]]:
        """
        Process multiple files and return their metadata along with error information.

        Args:
            file_paths: List of file paths to process

        Returns:
            Tuple[List[FileMetadata], List[Dict[str, Any]]]:
                - List of metadata for successfully processed files
                - List of error information for failed files
        """
        successful_files = []
        failed_files = []

        for file_path in file_paths:
            try:
                # Process file
                metadata = self.process_file(file_path)
                successful_files.append(metadata)
                logger.info(f"Successfully processed file: {file_path}")
            except UnsupportedMediaTypeError as e:
                # Log but ignore files with unsupported media types
                failed_files.append(
                    {
                        "path": file_path,
                        "error": str(e),
                        "error_type": "unsupported_media_type",
                    }
                )
                logger.warning(f"Ignoring unsupported media type: {file_path} - {e}")
            except MediaValidationError as e:
                # Log but ignore files that fail validation
                failed_files.append(
                    {
                        "path": file_path,
                        "error": str(e),
                        "error_type": "validation_error",
                    }
                )
                logger.warning(
                    f"Ignoring file that failed validation: {file_path} - {e}"
                )
            except Exception as e:
                # Log but ignore files that fail for other reasons
                failed_files.append(
                    {
                        "path": file_path,
                        "error": str(e),
                        "error_type": "processing_error",
                    }
                )
                logger.error(
                    f"Error processing file: {file_path} - {e} {type(e)} {traceback.format_exc()}"
                )
                # Removed "raise e" to allow processing of other files
        # Log summary
        logger.info(
            f"Processed {len(successful_files)} files successfully, {len(failed_files)} files failed"
        )

        return successful_files, failed_files

    def process_file(self, file_path: str) -> Optional[FileMetadata]:
        """
        Process and validate a file.

        Performs validation checks, generates metadata, and processes the file
        based on its type. Creates thumbnails for images and videos.

        Args:
            file_path: Path to the file to process

        Returns:
            Optional[FileMetadata]: Metadata if processing successful, None otherwise

        Raises:
            MediaValidationError: If file fails validation checks
            UnsupportedMediaTypeError: If file type is not supported
            MediaProcessingError: For general processing errors
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise MediaValidationError(f"File not found: {file_path}")

            file_type = FileConfig.get_file_type(str(file_path))
            if not file_type:
                raise UnsupportedMediaTypeError(f"Unsupported file type: {file_path}")

            file_size = file_path.stat().st_size
            config = FileConfig.SUPPORTED_FILE_TYPES[file_type]

            if file_size > config["max_size"]:
                raise MediaValidationError(
                    f"File too large: {file_size} bytes (max {config['max_size']} bytes)"
                )

            mime_type = (
                mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            )

            metadata = FileMetadata(
                path=str(file_path), type=file_type, size=file_size, mime_type=mime_type
            )

            # Process specific file types
            if file_type == "image":
                self._process_image(file_path, metadata)
            elif file_type == "code":
                self._process_code(file_path, metadata)
            elif file_type == "video":
                self._process_video(file_path, metadata)
            elif file_type == "text":
                self._process_text(file_path, metadata)

            return metadata

        except MediaProcessingError as e:
            logger.error(f"Media processing error for {file_path}: {e}")
            raise
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            raise MediaValidationError(f"File not found: {file_path}") from e
        except PermissionError as e:
            logger.error(f"Permission denied when accessing file: {file_path}")
            raise MediaProcessingError(
                f"Permission denied when accessing file: {file_path}"
            ) from e
        except OSError as e:
            logger.error(f"OS error when processing file {file_path}: {e}")
            raise MediaProcessingError(f"OS error when processing file: {e}") from e
        except MemoryError as e:
            logger.error(f"Out of memory when processing file {file_path}")
            raise MediaProcessingError(
                f"File too large to process in available memory: {file_path}"
            ) from e
        except Exception as e:
            logger.exception(
                f"Unexpected error processing file {file_path} {traceback.format_exc()}"
            )
            raise MediaProcessingError(f"Failed to process file {file_path}: {e}") from e

    def prepare_multiple_media_messages(
        self,
        file_metadatas: List[FileMetadata],
        conversation_context: str = "",
        role: str = "user",
    ) -> List[Dict[str, Any]]:
        """
        Prepare multiple media files for conversation inclusion.

        Creates structured messages containing the media content and metadata,
        suitable for inclusion in a conversation.

        Args:
            file_metadatas: List of FileMetadata objects
            conversation_context: Optional context string
            role: Message role (default: "user")

        Returns:
            List[Dict[str, Any]]: List of structured messages

        Raises:
            MediaProcessingError: If media processing fails
        """
        messages = []

        # Add context as a separate message if provided
        if conversation_context:
            messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": conversation_context}],
                    "metadata": {"type": "text"},
                }
            )

        # Process each file
        for metadata in file_metadatas:
            try:
                # Read file data
                try:
                    with open(metadata.path, "rb") as f:
                        file_data = f.read()
                except FileNotFoundError as e:
                    raise FileIOError(f"File not found: {metadata.path}") from e
                except PermissionError as e:
                    raise FileIOError(
                        f"Permission denied when accessing file: {metadata.path}"
                    ) from e
                except OSError as e:
                    raise FileIOError(f"Error reading file {metadata.path}: {e}") from e

                # Create message for this file
                message = {"role": role, "content": [], "metadata": metadata}

                # Add file content based on type
                if metadata.type == "image":
                    message["content"].append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "mime_type": metadata.mime_type,
                                "data": file_data,
                            },
                        }
                    )
                elif metadata.type == "video":
                    message["content"].append(
                        {
                            "type": "video",
                            "source": {
                                "type": "base64",
                                "mime_type": metadata.mime_type,
                                "data": file_data,
                            },
                        }
                    )

                messages.append(message)

            except FileIOError as e:
                logger.error(f"File I/O error: {e}")
                # Continue with other files
                continue
            except MediaProcessingError as e:
                logger.error(f"Media processing error: {e}")
                # Continue with other files
                continue
            except MemoryError:
                logger.error(
                    f"Out of memory when preparing media message for {metadata.path}"
                )
                # Continue with other files
                continue
            except Exception as e:
                logger.exception(f"Failed to prepare media message for {metadata.path}")
                # Continue with other files
                continue

        return messages

    def prepare_media_message(
        self, file_path: str, conversation_context: str = "", role: str = "user"
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare media for conversation inclusion.

        Creates a structured message containing the media content and metadata,
        suitable for inclusion in a conversation.

        Args:
            file_path: Path to the media file
            conversation_context: Optional context string
            role: Message role (default: "user")

        Returns:
            Optional[Dict[str, Any]]: Structured message if successful, None otherwise

        Raises:
            MediaProcessingError: If media processing fails
            MediaValidationError: If file validation fails
        """
        metadata = self.process_file(file_path)
        if not metadata:
            return None

        try:
            try:
                with open(file_path, "rb") as f:
                    file_data = f.read()
            except FileNotFoundError as e:
                raise FileIOError(f"File not found: {file_path}") from e
            except PermissionError as e:
                raise FileIOError(f"Permission denied when accessing file: {file_path}") from e
            except OSError as e:
                raise FileIOError(f"Error reading file {file_path}: {e}") from e

            # Base message structure
            message = {"role": role, "content": [], "metadata": metadata}

            # Add context if provided
            if conversation_context:
                message["content"].append(
                    {"type": "text", "text": conversation_context}
                )

            # Add media content based on type
            if metadata.type == "image":
                message["content"].append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "mime_type": metadata.mime_type,
                            "data": file_data,
                        },
                    }
                )
            elif metadata.type == "video":
                message["content"].append(
                    {
                        "type": "video",
                        "source": {
                            "type": "base64",
                            "mime_type": metadata.mime_type,
                            "data": file_data,
                        },
                    }
                )

            return message

        except FileIOError as e:
            logger.error(f"File I/O error: {e}")
            raise
        except MediaProcessingError as e:
            logger.error(f"Media processing error: {e}")
            raise
        except MemoryError as e:
            logger.error(f"Out of memory when preparing media message for {file_path}")
            raise MediaProcessingError(
                f"File too large to process in available memory: {file_path}"
            ) from e
        except Exception as e:
            logger.exception(f"Failed to prepare media message for {file_path}")
            raise MediaProcessingError(
                f"Failed to prepare media message for {file_path}: {e}"
            ) from e

    def create_media_prompt(
        self, metadata: FileMetadata, context: str = "", task: str = "analyze"
    ) -> str:
        """
        Create context-aware prompt for media analysis.

        Generates a structured prompt that guides the analysis of media content,
        incorporating conversation context and specific analysis tasks.

        Args:
            metadata: FileMetadata for the media
            context: Optional conversation context
            task: Analysis task type (default: "analyze")

        Returns:
            str: Formatted prompt string

        Raises:
            MediaProcessingConfigError: If prompt generation fails due to
                invalid configuration
        """
        logger.debug(f"Creating prompt for {metadata.type} with task: {task}")
        prompts = []

        # Add conversation context if provided
        if context:
            prompts.append(f"In the context of our discussion about {context}:")

        # Add media-specific prompts
        if metadata.type == "image":
            prompts.extend(
                [
                    f"This is a {metadata.mime_type} image",
                    (
                        f"Dimensions: {metadata.dimensions[0]}x{metadata.dimensions[1]} pixels"
                        if metadata.dimensions
                        else ""
                    ),
                ]
            )

            if task == "analyze":
                prompts.extend(
                    [
                        "Let's analyze this image together. Consider:",
                        "1. Visual content and composition",
                        "2. Any text or writing present",
                        "3. Relevance to our discussion",
                        "4. Key insights or implications",
                        "\nShare your initial analysis, but also:",
                        "- Ask your conversation partner what aspects stand out to them",
                        "- Identify any patterns or details that might need closer examination",
                        "- Suggest specific elements that could benefit from your partner's expertise",
                        "- Be open to alternative interpretations and encourage deeper exploration",
                        "\nRemember to:",
                        "- Connect your analysis to the ongoing discussion",
                        "- Challenge assumptions and invite different perspectives",
                        "- Build on your partner's insights to develop a richer understanding",
                        "- Identify potential areas for further investigation",
                    ]
                )
            elif task == "enhance":
                prompts.extend(
                    [
                        "Please suggest improvements for this image:",
                        "1. Quality enhancements",
                        "2. Composition adjustments",
                        "3. Relevant modifications based on our discussion",
                        "4. Ask your partner's opinion on these suggestions",
                    ]
                )

        elif metadata.type == "video":
            prompts.extend(
                [
                    f"This is a {metadata.mime_type} video",
                    (
                        f"Duration: {metadata.duration:.1f} seconds"
                        if metadata.duration
                        else ""
                    ),
                    (
                        f"Resolution: {metadata.dimensions[0]}x{metadata.dimensions[1]}"
                        if metadata.dimensions
                        else ""
                    ),
                ]
            )

            prompts.extend(
                [
                    "Let's analyze this video together. Consider:",
                    "1. Visual content and scene composition",
                    "2. Motion and temporal elements",
                    "3. Relevance to our discussion",
                    "4. Key moments or insights",
                    "\nAfter sharing your initial observations:",
                    "- Ask which scenes or moments caught your partner's attention",
                    "- Discuss any patterns or themes you notice",
                    "- Explore how different interpretations might affect our understanding",
                    "- Consider what aspects deserve deeper analysis",
                ]
            )

        return "\n".join(filter(None, prompts))

    def _process_image(self, file_path: Path, metadata: FileMetadata) -> None:
        """
        Process image files.

        Validates image dimensions, creates thumbnail, and updates metadata.

        Args:
            file_path: Path to image file
            metadata: FileMetadata to update

        Raises:
            MediaValidationError: If image validation fails
        """
        try:
            try:
                with Image.open(file_path) as img:
                    # Check dimensions
                    if (
                        img.size[0]
                        > FileConfig.SUPPORTED_FILE_TYPES["image"]["max_resolution"][0]
                        or img.size[1]
                        > FileConfig.SUPPORTED_FILE_TYPES["image"]["max_resolution"][1]
                    ):
                        logger.info(
                            f"Image dimensions {img.size} exceed maximum resolution "
                            + f"{FileConfig.SUPPORTED_FILE_TYPES['image']['max_resolution']}"
                        )
                        logger.warning(
                            f"Image dimensions {img.size} exceed maximum, will be resized"
                        )

                    metadata.dimensions = img.size

                    # Create thumbnail
                    logger.info(
                        f"Creating thumbnail for image {file_path} with original size {img.size}"
                    )

                    # Calculate thumbnail size maintaining aspect ratio with longest side = 768px
                    max_dimension = 768
                    width, height = img.size
                    if width > height:
                        thumb_size = (
                            max_dimension,
                            int(height * max_dimension / width),
                        )
                    else:
                        thumb_size = (
                            int(width * max_dimension / height),
                            max_dimension,
                        )

                    logger.info(
                        f"Calculated thumbnail size: {thumb_size} (maintaining aspect ratio)"
                    )
                    thumb_img = img.copy()
                    thumb_img.thumbnail(thumb_size)

                    # Save thumbnail to file
                    thumb_path = self.output_dir / f"thumb_{file_path.name}"
                    try:
                        thumb_img.save(thumb_path)
                        metadata.thumbnail_path = str(thumb_path)
                        logger.info(
                            f"Created thumbnail for {file_path}: {thumb_path} with size {thumb_img.size}"
                        )
                    except (OSError, IOError) as e:
                        logger.error(f"Failed to save thumbnail for {file_path}: {e}")
                        # Continue processing even if thumbnail creation fails

                    # Resize image if larger than max resolution
                    if (
                        img.size[0] > self.max_image_resolution[0]
                        or img.size[1] > self.max_image_resolution[1]
                    ):
                        logger.info(
                            f"Resizing image {file_path} from {img.size} to fit within {self.max_image_resolution}"
                        )
                        # Calculate new dimensions while maintaining aspect ratio
                        ratio = min(
                            self.max_image_resolution[0] / img.size[0],
                            self.max_image_resolution[1] / img.size[1],
                        )
                        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                        resized_img = img.resize(new_size, Image.LANCZOS)  # pylint: disable=no-member
                        logger.info(
                            f"Resized image from {metadata.dimensions} to {new_size}"
                        )
                        metadata.dimensions = new_size
                        resized_path = self.output_dir / f"resized_{file_path.name}"
                        logger.info(f"Saving resized image to {resized_path}")
                        resized_img.save(resized_path)
            except UnidentifiedImageError as e:
                raise ImageProcessingError(f"Cannot identify image file: {file_path}") from e
            except (IOError, OSError) as e:
                raise ImageProcessingError(
                    f"Error opening or processing image file {file_path}: {e}"
                ) from e
        except Exception as e:
            logger.exception(f"Unexpected error processing image {file_path}")
            raise ImageProcessingError(f"Failed to process image {file_path}: {e}") from e

    def _process_video(self, file_path: Path, metadata: FileMetadata) -> None:
        """
        Process video files.

        Processes the video at a lower framerate and resolution for model consumption.

        Extracts video properties, creates thumbnail from first frame,
        and updates metadata.

        Args:
            file_path: Path to video file
            metadata: FileMetadata to update

        Returns:
            None: Updates metadata in place

        Note: Requires OpenCV (cv2) package for video processing
        """
        try:
            import cv2  # pylint: disable=import-outside-toplevel

            try:
                video = cv2.VideoCapture(str(file_path))
                if not video.isOpened():
                    raise VideoProcessingError(
                        f"Failed to open video file: {file_path}"
                    )
            except Exception as e:
                raise VideoProcessingError(f"Error opening video file {file_path}: {e}") from e

            # Get video properties
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = video.get(cv2.CAP_PROP_FPS)
            duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)

            metadata.dimensions = (width, height)
            metadata.duration = duration

            # Process video at lower framerate and resolution
            target_fps = 3  # Configurable

            # Use max_dimension from metadata if available
            max_dimension = 768# Default to 512 if not specified
            if hasattr(metadata, "max_resolution") and metadata.max_resolution:
                try:
                    # Parse max_resolution string
                    resolution_parts = metadata.max_resolution.split("x")
                    if len(resolution_parts) == 2:
                        # Use the first dimension as max_dimension
                        max_dimension = int(resolution_parts[0])
                        logger.info(
                            f"Using max_dimension {max_dimension} from configuration"
                        )
                except (ValueError, AttributeError) as e:
                    logger.warning(
                        f"Could not parse max_resolution: {e}, using default {max_dimension}"
                    )

            logger.info(
                f"Processing video {file_path} with original dimensions {width}x{height}, fps: {original_fps}"
            )
            logger.info(
                f"Target processing parameters: max dimension {max_dimension}px, fps: {target_fps}"
            )

            # Calculate frame interval based on target FPS
            frame_interval = int(original_fps / target_fps)
            if frame_interval < 1:
                logger.info(
                    f"Original fps {original_fps} is less than target fps {target_fps}, using every frame"
                )
                logger.info(
                    f"Using frame interval of {frame_interval} (every {frame_interval}th frame)"
                )
                frame_interval = 1

            # Create directory for processed frames
            frames_dir = self.output_dir / f"frames_{file_path.stem}"
            try:
                frames_dir.mkdir(exist_ok=True)
            except (OSError, IOError) as e:
                logger.error(f"Failed to create frames directory {frames_dir}: {e}")

            # Extract frames at target FPS and resolution
            frame_count = 0
            frame_paths = []

            while True:
                # Set position to next frame
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)

                success, frame = video.read()
                if not success:
                    break

                # Calculate new dimensions maintaining aspect ratio with longest side = max_dimension
                if width > height:  # Landscape
                    new_size = (max_dimension, int(height * max_dimension / width))
                else:  # Portrait
                    new_size = (int(width * max_dimension / height), max_dimension)

                logger.info(
                    f"Resizing frame {frame_count} from {width}x{height} to {new_size} (maintaining aspect ratio)"
                )
                frame = cv2.resize(frame, new_size)

                # Save frame
                frame_path = frames_dir / f"frame_{frame_count:04d}.jpg"
                try:
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
                except Exception as e:
                    logger.error(
                        f"Failed to save frame {frame_count} to {frame_path}: {e}"
                    )
                    # Continue processing other frames
                    continue
                logger.info(f"Saved frame {frame_count} to {frame_path}")
                frame_count += 1
                # Create thumbnail from first frame
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
                success, frame = video.read()
                if success:
                    # Resize thumbnail maintaining aspect ratio
                    thumb_frame = cv2.resize(frame, new_size)
                    logger.info(
                        f"Creating thumbnail for video {file_path} with size {new_size}"
                    )
                    thumb_path = self.output_dir / f"thumb_{file_path.name}.jpg"
                    try:
                        cv2.imwrite(str(thumb_path), thumb_frame)
                        metadata.thumbnail_path = str(thumb_path)
                    except Exception as e:
                        logger.error(
                            f"Failed to save thumbnail for video {file_path}: {e}"
                        )
                    logger.debug(
                        f"Created thumbnail for video {file_path}: {thumb_path}"
                    )

                # Process video at lower resolution and save
                processed_video_path = self.output_dir / f"processed_{file_path.name}"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4

                # Check if the file is a .mov file and convert to .mp4 using ffmpeg
                if file_path.suffix.lower() == ".mov":
                    # Create a .mp4 output path
                    mp4_output_path = (
                        self.output_dir / f"processed_{file_path.stem}.mp4"
                    )

                    try:
                        # Use ffmpeg to convert .mov to .mp4 with proper encoding for Gemini
                        cmd = [
                            "ffmpeg",
                            "-i",
                            str(file_path),
                            "-c:v",
                            "libx264",  # H.264 video codec
                            "-preset",
                            "medium",  # Encoding speed/quality balance
                            "-crf",
                            "23",  # Quality level (lower is better)
                            "-c:a",
                            "aac",  # AAC audio codec
                            "-b:a",
                            "128k",  # Audio bitrate
                            "-vf",
                            f"scale={new_size[0]}:{new_size[1]}",  # Resize
                            "-r",
                            str(target_fps),  # Target framerate
                            "-y",  # Overwrite output file if it exists
                            str(mp4_output_path),
                        ]

                        logger.info(
                            f"Converting .mov to .mp4 using ffmpeg: {' '.join(cmd)}"
                        )
                        result = subprocess.run(cmd, capture_output=True, text=True)

                        if result.returncode != 0:
                            logger.error(f"ffmpeg conversion failed: {result.stderr}")
                            raise VideoProcessingError(
                                f"ffmpeg conversion failed: {result.stderr}"
                            )

                        logger.info(
                            f"Successfully converted {file_path} to {mp4_output_path}"
                        )
                        processed_video_path = mp4_output_path

                        # Get frame count from the converted video
                        converted_video = cv2.VideoCapture(str(mp4_output_path))
                        frame_count = int(converted_video.get(cv2.CAP_PROP_FRAME_COUNT))
                        converted_video.release()

                    except Exception as e:
                        logger.error(f"Error converting .mov to .mp4: {e}")
                        # Fall back to OpenCV processing
                        logger.info("Falling back to OpenCV processing")
                        out = cv2.VideoWriter(
                            str(processed_video_path), fourcc, target_fps, new_size
                        )
                        video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
                        frame_count = 0

                        while True:
                            success, frame = video.read()
                            if not success:
                                break

                            # Resize frame maintaining aspect ratio
                            resized_frame = cv2.resize(frame, new_size)
                            out.write(resized_frame)
                            frame_count += 1

                        out.release()
                else:
                    # For non-mov files, use the original OpenCV approach
                    out = cv2.VideoWriter(
                        str(processed_video_path), fourcc, target_fps, new_size
                    )
                    video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
                    frame_count = 0

                    while True:
                        success, frame = video.read()
                        if not success:
                            break

                        # Resize frame maintaining aspect ratio
                        resized_frame = cv2.resize(frame, new_size)
                        out.write(resized_frame)
                        frame_count += 1

                    out.release()

                video.release()

                # Store processed video information
                metadata.processed_video = {
                    "fps": target_fps,
                    "resolution": new_size,
                    "processed_video_path": str(processed_video_path),
                    "frame_count": frame_count,
                    "frame_paths": frame_paths,
                    "mime_type": "video/mp4",  # Always use mp4 mime type for better compatibility
                }
                logger.info(
                    f"Processed video saved to {processed_video_path} with {frame_count} frames at {target_fps} fps and resolution {new_size}"
                )

        except ImportError as e:
            logger.warning(
                "OpenCV not available for video processing. Install opencv-python package."
            )
            raise VideoProcessingError(
                "OpenCV library not available for video processing"
            ) from e
        except VideoProcessingError:
            raise  # Re-raise specific exceptions
        except Exception as e:
            logger.exception(f"Unexpected error processing video {file_path}")
            raise VideoProcessingError(f"Failed to process video {file_path}: {e}") from e

    def _process_text(self, file_path: Path, metadata: FileMetadata) -> None:
        """
        Process text files.

        Reads and stores text content in metadata.

        Args:
            file_path: Path to text file
            metadata: FileMetadata to update

        Raises:
            MediaValidationError: If text file cannot be decoded
        """
        try:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    metadata.text_content = content
            except FileNotFoundError as e:
                raise TextProcessingError(f"Text file not found: {file_path}") from e
            except PermissionError as e:
                raise TextProcessingError(
                    f"Permission denied when accessing text file: {file_path}"
                ) from e
        except UnicodeDecodeError as e:
            raise MediaValidationError(
                f"Could not decode text file {file_path} as UTF-8"
            ) from e
        except Exception as e:
            raise TextProcessingError(f"Error processing text file {file_path}: {e}") from e

    def _process_code(self, file_path: Path, metadata: FileMetadata) -> None:
        """
        Process code files with syntax highlighting and line numbers.

        Args:
            file_path: Path to code file
            metadata: FileMetadata to update

        Raises:
            MediaValidationError: If code file cannot be decoded
        """
        try:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except FileNotFoundError as e:
                raise CodeProcessingError(f"Code file not found: {file_path}") from e
            except PermissionError as e:
                raise CodeProcessingError(
                    f"Permission denied when accessing code file: {file_path}"
                ) from e
            except OSError as e:
                raise CodeProcessingError(f"Error reading code file {file_path}: {e}") from e

            # Get file extension for language detection
            ext = file_path.suffix.lower()
            language = ext[1:] if ext else "text"  # Remove the dot

            # Format with line numbers
            lines = content.split("\n")
            formatted_content = "\n".join(
                [f"{i+1} | {line}" for i, line in enumerate(lines)]
            )

            # Update metadata
            metadata.text_content = formatted_content
            metadata.dimensions = (
                len(lines),
                max(len(line) for line in lines) if lines else 0,
            )
            metadata.mime_type = f"text/x-{language}"
        except UnicodeDecodeError as e:
            raise MediaValidationError(
                f"Could not decode code file {file_path} as UTF-8"
            ) from e
        except Exception as e:
            raise CodeProcessingError(f"Error processing code file {file_path}: {e}") from e
