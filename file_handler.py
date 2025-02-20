"""File and media handling integrated with conversation flow"""
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

@dataclass
class FileMetadata:
    """Metadata for processed files"""
    path: str
    type: str
    size: int
    mime_type: str
    dimensions: Optional[Tuple[int, int]] = None
    duration: Optional[float] = None
    text_content: Optional[str] = None
    thumbnail_path: Optional[str] = None

class FileConfig:
    """Configuration for file type handling"""
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
        """Determine file type from extension"""
        ext = os.path.splitext(file_path)[1].lower()
        for file_type, config in cls.SUPPORTED_FILE_TYPES.items():
            if ext in config["extensions"]:
                return file_type
        return None

    @classmethod
    def can_handle_media(cls, model_name: str, file_type: str) -> bool:
        """Check if model can handle media type"""
        if file_type not in cls.SUPPORTED_FILE_TYPES:
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
    """Handles media analysis within conversation context"""
    
    def __init__(self, output_dir: str = "processed_files"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def process_file(self, file_path: str) -> Optional[FileMetadata]:
        """Process and validate a file"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            file_type = FileConfig.get_file_type(str(file_path))
            if not file_type:
                raise ValueError(f"Unsupported file type: {file_path}")
                
            file_size = file_path.stat().st_size
            config = FileConfig.SUPPORTED_FILE_TYPES[file_type]
            
            if file_size > config["max_size"]:
                raise ValueError(f"File too large: {file_size} bytes (max {config['max_size']} bytes)")
                
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
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None

    def prepare_media_message(self, 
                            file_path: str,
                            conversation_context: str = "",
                            role: str = "user") -> Optional[Dict[str, Any]]:
        """Prepare media for conversation inclusion"""
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
            logger.error(f"Error preparing media message: {e}")
            return None

    def create_media_prompt(self, 
                          metadata: FileMetadata,
                          context: str = "",
                          task: str = "analyze") -> str:
        """Create context-aware prompt for media analysis"""
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
                    "Please analyze this image considering:",
                    "1. Visual content and composition",
                    "2. Any text or writing present",
                    "3. Relevance to our discussion",
                    "4. Key insights or implications"
                ])
            elif task == "enhance":
                prompts.extend([
                    "Please suggest improvements for this image:",
                    "1. Quality enhancements",
                    "2. Composition adjustments",
                    "3. Relevant modifications based on our discussion"
                ])

        elif metadata.type == "video":
            prompts.extend([
                f"This is a {metadata.mime_type} video",
                f"Duration: {metadata.duration:.1f} seconds" if metadata.duration else "",
                f"Resolution: {metadata.dimensions[0]}x{metadata.dimensions[1]}" if metadata.dimensions else ""
            ])

            prompts.extend([
                "Please analyze this video considering:",
                "1. Visual content and scene composition",
                "2. Motion and temporal elements",
                "3. Relevance to our discussion",
                "4. Key moments or insights"
            ])

        return "\n".join(filter(None, prompts))

    def _process_image(self, file_path: Path, metadata: FileMetadata) -> None:
        """Process image files"""
        with Image.open(file_path) as img:
            # Check dimensions
            if img.size[0] > FileConfig.SUPPORTED_FILE_TYPES["image"]["max_resolution"][0] or \
               img.size[1] > FileConfig.SUPPORTED_FILE_TYPES["image"]["max_resolution"][1]:
                raise ValueError(f"Image dimensions too large: {img.size}")
                
            metadata.dimensions = img.size
            
            # Create thumbnail
            thumb_size = (512, 512)
            img.thumbnail(thumb_size)
            thumb_path = self.output_dir / f"thumb_{file_path.name}"
            img.save(thumb_path)
            metadata.thumbnail_path = str(thumb_path)
            
    def _process_video(self, file_path: Path, metadata: FileMetadata) -> None:
        """Process video files"""
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
                
            video.release()
            
        except ImportError:
            logger.warning("OpenCV not available for video processing")
            
    def _process_text(self, file_path: Path, metadata: FileMetadata) -> None:
        """Process text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                metadata.text_content = content
        except UnicodeDecodeError:
            logger.warning(f"Could not decode text file {file_path} as UTF-8")