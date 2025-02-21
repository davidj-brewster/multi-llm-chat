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
import os
import io
import logging
import mimetypes
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from google.genai import types

logger = logging.getLogger(__name__)

class MediaProcessingError(Exception):
    """Base exception for media processing errors."""
    pass

class UnsupportedMediaTypeError(MediaProcessingError):
    """Raised when file type is not supported."""
    pass

class MediaValidationError(MediaProcessingError):
    """Raised when file fails validation checks."""
    pass

class MediaProcessingConfigError(MediaProcessingError):
    """Raised when there's a configuration-related error."""
    pass

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
    """
    path: str
    type: str
    size: int
    mime_type: str
    dimensions: Optional[Tuple[int, int]] = None
    duration: Optional[float] = None
    text_content: Optional[str] = None
    thumbnail_path: Optional[str] = None

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
            "max_size": 20 * 1024 * 1024,  # 20MB
            "max_resolution": (8192, 8192),
            "supported_models": {
                "gemini": ["gemini-pro-vision"],
                "claude": ["claude-3-sonnet", "claude-3-haiku"],
                "openai": ["gpt-4-vision", "dall-e-3"]
            }
        },
        "video": {
            "extensions": [".mp4", ".mov", ".avi", ".webm"],
            "max_size": 300 * 1024 * 1024,  # 300MB
            "max_resolution": (3840, 2160),  # 4K
            "supported_models": {
                "gemini": ["gemini-pro-vision"]
            }
        },
        "text": {
            "extensions": [".txt", ".md", ".py", ".js", ".html", ".csv", ".json", ".yaml", ".yml"],
            "max_size": 20 * 1024 * 1024  # 20MB
        }
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
            "o1": "openai"
        }
        
        # Get provider from model name
        provider = next((p for k, p in provider_map.items() if k in model_name.lower()), None)
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
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized ConversationMediaHandler with output directory: {output_dir}")

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
                raise MediaValidationError(f"File too large: {file_size} bytes (max {config['max_size']} bytes)")
                
            mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            
            metadata = FileMetadata(
                path=str(file_path),
                type=file_type,
                size=file_size,
                mime_type=mime_type
            )
            
            # Process specific file types
            if file_type == "image":
                self._process_image(file_path, metadata)
            elif file_type == "video":
                self._process_video(file_path, metadata)
            elif file_type == "text":
                self._process_text(file_path, metadata)
                
            return metadata
            
        except MediaProcessingError as e:
            logger.error(f"Media processing error for {file_path}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error processing file {file_path}")
            raise MediaProcessingError(f"Failed to process file: {e}")

    def prepare_media_message(self, 
                            file_path: str,
                            conversation_context: str = "",
                            role: str = "user") -> Optional[Dict[str, Any]]:
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
            with open(file_path, 'rb') as f:
                file_data = f.read()

            # Base message structure
            message = {
                "role": role,
                "content": [],
                "metadata": metadata
            }

            # Add context if provided
            if conversation_context:
                message["content"].append({
                    "type": "text",
                    "text": conversation_context
                })

            # Add media content based on type
            if metadata.type == "image":
                message["content"].append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "mime_type": metadata.mime_type,
                        "data": file_data
                    }
                })
            elif metadata.type == "video":
                message["content"].append({
                    "type": "video",
                    "source": {
                        "type": "base64",
                        "mime_type": metadata.mime_type,
                        "data": file_data
                    }
                })

            return message

        except Exception as e:
            logger.exception(f"Failed to prepare media message for {file_path}")
            raise MediaProcessingError(f"Failed to prepare media message: {e}")

    def create_media_prompt(self, 
                          metadata: FileMetadata,
                          context: str = "",
                          task: str = "analyze") -> str:
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
            prompts.extend([
                f"This is a {metadata.mime_type} image",
                f"Dimensions: {metadata.dimensions[0]}x{metadata.dimensions[1]} pixels" if metadata.dimensions else ""
            ])

            if task == "analyze":
                prompts.extend([
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
                    "- Identify potential areas for further investigation"
                ])
            elif task == "enhance":
                prompts.extend([
                    "Please suggest improvements for this image:",
                    "1. Quality enhancements",
                    "2. Composition adjustments",
                    "3. Relevant modifications based on our discussion",
                    "4. Ask your partner's opinion on these suggestions"
                ])

        elif metadata.type == "video":
            prompts.extend([
                f"This is a {metadata.mime_type} video",
                f"Duration: {metadata.duration:.1f} seconds" if metadata.duration else "",
                f"Resolution: {metadata.dimensions[0]}x{metadata.dimensions[1]}" if metadata.dimensions else ""
            ])

            prompts.extend([
                "Let's analyze this video together. Consider:",
                "1. Visual content and scene composition",
                "2. Motion and temporal elements",
                "3. Relevance to our discussion",
                "4. Key moments or insights",
                "\nAfter sharing your initial observations:",
                "- Ask which scenes or moments caught your partner's attention",
                "- Discuss any patterns or themes you notice",
                "- Explore how different interpretations might affect our understanding",
                "- Consider what aspects deserve deeper analysis"
            ])

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
        with Image.open(file_path) as img:
            # Check dimensions
            if img.size[0] > FileConfig.SUPPORTED_FILE_TYPES["image"]["max_resolution"][0] or \
               img.size[1] > FileConfig.SUPPORTED_FILE_TYPES["image"]["max_resolution"][1]:
                raise MediaValidationError(f"Image dimensions too large: {img.size}")
                
            metadata.dimensions = img.size
            
            # Create thumbnail
            thumb_size = (512, 512)
            img.thumbnail(thumb_size)
            thumb_path = self.output_dir / f"thumb_{file_path.name}"
            img.save(thumb_path)
            metadata.thumbnail_path = str(thumb_path)
            logger.debug(f"Created thumbnail for {file_path}: {thumb_path}")
            
    def _process_video(self, file_path: Path, metadata: FileMetadata) -> None:
        """
        Process video files.
        
        Extracts video properties, creates thumbnail from first frame,
        and updates metadata.
        
        Args:
            file_path: Path to video file
            metadata: FileMetadata to update
            
        Note: Requires OpenCV (cv2) package for video processing
        """
        try:
            import cv2
            video = cv2.VideoCapture(str(file_path))
            
            # Get video properties
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
            
            metadata.dimensions = (width, height)
            metadata.duration = duration
            
            # Create thumbnail from first frame
            success, frame = video.read()
            if success:
                thumb_path = self.output_dir / f"thumb_{file_path.name}.jpg"
                cv2.imwrite(str(thumb_path), frame)
                metadata.thumbnail_path = str(thumb_path)
                logger.debug(f"Created thumbnail for video {file_path}: {thumb_path}")
                
            video.release()
            
        except ImportError:
            logger.warning("OpenCV not available for video processing")
            
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
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                metadata.text_content = content
        except UnicodeDecodeError:
            raise MediaValidationError(f"Could not decode text file {file_path} as UTF-8")