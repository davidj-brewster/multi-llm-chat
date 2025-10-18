"""Configuration integration module for AI Battle framework"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from logging import getLogger
import logging

# Import consolidated constants from constants.py
from constants import SUPPORTED_MODELS, SUPPORTED_FILE_TYPES

logger = getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class TimeoutConfig:
    """
    Configuration for request timeouts and retry behavior.
    (Docstring remains the same)
    """
    request: int = field(default=600)
    retry_count: int = field(default=1)
    notify_on: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.request < 30 or self.request > 600:
            raise ValueError("Request timeout must be between 30 and 600 seconds")
        if self.retry_count < 0 or self.retry_count > 5:
            raise ValueError("Retry count must be between 0 and 5")
        if not self.notify_on:
            self.notify_on = ["timeout", "retry", "error"]
        valid_events = ["timeout", "retry", "error"]
        invalid_events = [e for e in self.notify_on if e not in valid_events]
        if invalid_events:
            raise ValueError(f"Invalid notification events: {invalid_events}")


@dataclass
class FileConfig:
    """
    Configuration for file handling and validation.
    (Docstring remains the same)
    """
    path: str
    type: str
    max_resolution: Optional[str] = field(default=None)
    is_directory: bool = field(default=False)
    file_pattern: Optional[str] = field(default=None)

    @staticmethod
    def get_file_type(file_path_str: str) -> Optional[str]:
        """
        Determine file type from extension using imported SUPPORTED_FILE_TYPES.
        """
        ext = os.path.splitext(file_path_str)[1].lower()
        for f_type, config in SUPPORTED_FILE_TYPES.items():
            if ext in config.get("extensions", []):
                return f_type
        logger.warning(f"Could not determine file type for {file_path_str} based on extension {ext}.")
        return None

    def __post_init__(self):
        if not os.path.exists(self.path):
            raise ValueError(f"Path not found: {self.path}")

        if self.is_directory:
            if not os.path.isdir(self.path):
                raise ValueError(f"Path is not a directory: {self.path}")
            return

        # If type was None and get_file_type was used, it's now set.
        # If type is still None here, it means it wasn't provided and couldn't be derived.
        if not self.type:
             # Attempt to derive type one last time if not set (e.g. direct instantiation with path only)
             # This case is less likely if constructed via DiscussionConfig processing
             derived_type = FileConfig.get_file_type(self.path)
             if derived_type:
                 self.type = derived_type
             else:
                 raise ValueError(f"File type for {self.path} is not specified and could not be derived from extension.")

        file_size = os.path.getsize(self.path)
        extension = os.path.splitext(self.path)[1].lower()

        if self.type not in SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported file type: {self.type}")

        type_config = SUPPORTED_FILE_TYPES[self.type]

        if extension not in type_config["extensions"]:
            raise ValueError(
                f"Unsupported file extension {extension} for type {self.type}. "
                f"Supported extensions: {', '.join(type_config['extensions'])}"
            )

        if file_size > type_config["max_size"]:
            readable_size = f"{type_config['max_size'] / (1024 * 1024):.1f}MB"
            raise ValueError(
                f"File size {file_size / (1024 * 1024):.1f}MB exceeds maximum {readable_size} for type {self.type}"
            )
        
        if self.type in ["image", "video"] and self.max_resolution:
            if "max_resolution" not in type_config:
                logger.warning(f"Max resolution not defined for type {self.type} in SUPPORTED_FILE_TYPES")
                return

            max_width, max_height = type_config["max_resolution"]
            requested_res = self.max_resolution.upper()
            if requested_res == "4K":
                if self.type != "video":
                    raise ValueError("4K resolution only supported for video files")
            elif "X" in requested_res:
                try:
                    width, height = map(int, requested_res.split("X"))
                    if width > max_width or height > max_height:
                        raise ValueError(
                            f"Requested resolution {width}x{height} exceeds maximum {max_width}x{max_height}"
                        )
                except ValueError:
                    raise ValueError(f"Invalid resolution format: {requested_res}. Expected format: WIDTHxHEIGHT")


@dataclass
class MultiFileConfig:
    """
    Configuration for handling multiple files.
    (Docstring remains the same)
    """
    files: List[FileConfig] = field(default_factory=list)
    directory: Optional[str] = field(default=None)
    file_pattern: Optional[str] = field(default=None)
    max_files: int = field(default=10)
    max_resolution: Optional[str] = field(default=None)

    def __post_init__(self):
        if not self.files and not self.directory:
            raise ValueError("Either files or directory must be provided")

        if self.directory and not os.path.exists(self.directory):
            raise ValueError(f"Directory not found: {self.directory}")

        if self.directory and not os.path.isdir(self.directory):
            raise ValueError(f"Path is not a directory: {self.directory}")


@dataclass
class ModelConfig:
    """
    Configuration for AI model settings and behavior.
    (Docstring remains the same)
    """
    type: str
    role: str
    persona: Optional[str] = field(default=None)

    def __post_init__(self):
        provider = next((p for p in SUPPORTED_MODELS if self.type.startswith(p)), None)
        if not provider:
            provider_candidate = self.type.split(":")[0] if ":" in self.type else None
            if provider_candidate and SUPPORTED_MODELS.get(provider_candidate) == ["*"]:
                provider = provider_candidate
            else:
                raise ValueError(f"Unsupported model type or provider: {self.type}. Known providers: {', '.join(SUPPORTED_MODELS.keys())}")

        if SUPPORTED_MODELS[provider] != ["*"]:
            is_supported = self.type in SUPPORTED_MODELS[provider] or \
                           any(m.endswith('*') and self.type.startswith(m[:-1]) for m in SUPPORTED_MODELS[provider])
            if not is_supported:
                raise ValueError(f"Unsupported model variant: {self.type} for provider {provider}. "
                                 f"Supported variants: {', '.join(SUPPORTED_MODELS[provider])}")

        if self.role not in ["human", "assistant"]:
            raise ValueError(
                f"Invalid role: {self.role}. Must be 'human' or 'assistant'"
            )
        
        if self.persona is not None and not isinstance(self.persona, str):
            logger.warning(f"Persona for model {self.type} is not a string (type: {type(self.persona)}). Converting to string.")
            self.persona = str(self.persona)


@dataclass
class DiscussionConfig:
    """
    Configuration for AI-to-AI or human-to-AI discussions.
    (Docstring remains the same)
    """
    turns: int
    models: Dict[str, ModelConfig]
    goal: str
    input_file: Optional[FileConfig] = field(default=None)
    input_files: Optional[MultiFileConfig] = field(default=None)
    timeouts: Optional[TimeoutConfig] = field(default=None)

    def __post_init__(self):
        if self.turns < 1:
            raise ValueError("Turns must be greater than 0")

        if len(self.models) < 2:
            raise ValueError("At least two models must be configured")

        if not self.goal:
            raise ValueError("Goal must be provided")

        logger.info(
            f"Configuring discussion with {len(self.models)} models for {self.turns} turns"
        )
        if isinstance(self.models, dict):
            self.models = {
                name: ModelConfig(**config) if isinstance(config, dict) else config
                for name, config in self.models.items()
            }
        logger.debug(f"Models: {self.models}")

        if isinstance(self.timeouts, dict):
            self.timeouts = TimeoutConfig(**self.timeouts)
        elif self.timeouts is None:
            self.timeouts = TimeoutConfig()

        if isinstance(self.input_files, dict):
            if "files" in self.input_files and isinstance(
                self.input_files["files"], list
            ):
                self.input_files["files"] = [
                    (
                        FileConfig(**file_dict)
                        if isinstance(file_dict, dict)
                        else file_dict
                    )
                    for file_dict in self.input_files["files"]
                ]
            self.input_files = MultiFileConfig(**self.input_files)

        # Process self.input_file
        if self.input_file: # Only process if it's not None
            if isinstance(self.input_file, str): # Case 1: Path string
                derived_type = FileConfig.get_file_type(self.input_file)
                if derived_type:
                    self.input_file = FileConfig(path=self.input_file, type=derived_type)
                else:
                    raise ValueError(f"Cannot determine file type for input_file path: {self.input_file}. Please specify type or use a supported extension.")
            elif isinstance(self.input_file, dict): # Case 2: Dictionary
                self.input_file = FileConfig(**self.input_file)
            elif not isinstance(self.input_file, FileConfig): # Case 3: Neither string, dict, nor FileConfig object
                raise ValueError(f"input_file must be a path string, a dictionary, or a FileConfig object, not {type(self.input_file)}")
            # If self.input_file was already a FileConfig object, it remains unchanged, which is correct.
        
        logger.debug(f"Input file: {self.input_file}")
