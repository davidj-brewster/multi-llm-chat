"""Configuration integration module for AI Battle framework"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from logging import getLogger
import logging

# Constants referenced in the code but not defined in the snippet
# These would typically be defined in this file or imported
# Supported model configurations
SUPPORTED_MODELS = {
    "claude": [
        "claude",
        "sonnet",
        "haiku",
        "claude*",
        "claude-3-5-haiku",
        "claude-3-7",
        "claude-3-7-latest",
        "claude-3-7-sonnet-latest",
        "claude-3-5-sonnet-latest",
        "claude-3-7-sonnet",
        "claude-3-7-reasoning",
        "claude-3-7-reasoning-medium",
        "claude-3-7-reasoning-low",
        "claude-3-7-reasoning-none",
    ],
    "gemini": [
        "gemini*",
        "gemini-2-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.5-pro-exp",
        "gemini-2-pro",
        "gemini-2-reasoning",
        "gemini-2.0-flash-exp",
        "gemini",
    ],
    "openai": [
        "gpt-4-vision",
        "gpt-4o",
        "chatgpt-latest",
        "o1",
        "o3",
        "o1-reasoning-high",
        "o1-reasoning-medium",
        "o1-reasoning-low",
        "o3-reasoning-high",
        "o3-reasoning-medium",
        "o3-reasoning-low",
    ],
    "ollama": ["*"],  # All Ollama models supported
    "mlx": ["*"],  # All MLX models supported
}

# File type configurations
SUPPORTED_FILE_TYPES = {
    "image": {
        "extensions": [".jpg", ".jpeg", ".png", ".gif", ".webp"],
        "max_size": 200 * 1024 * 1024,  # 20MB
        "max_resolution": (8192, 8192),
    },
    "video": {
        "extensions": [".mp4", ".mov", ".avi", ".webm"],
        "max_size": 3000 * 1024 * 1024,  # 100MB
        "max_resolution": (3840, 2160),  # 4K
    },
    "text": {
        "extensions": [
            ".txt",
            ".md",
            ".py",
            ".js",
            ".html",
            ".css",
            ".json",
            ".yaml",
            ".yml",
        ],
        "max_size": 300 * 1024 * 1024,  # 10MB
    },
}
logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class TimeoutConfig:
    """
    Configuration for request timeouts and retry behavior.

    This class defines timeout settings for API requests, including the maximum
    request duration, retry behavior, and notification preferences. It includes
    validation to ensure the settings are within acceptable ranges.

    Attributes:
        request (int): Maximum request duration in seconds. Must be between
            30 and 600 seconds (5 minutes). Defaults to 600 seconds.

        retry_count (int): Number of retry attempts for failed requests.
            Must be between 0 and 5. Defaults to 1.

        notify_on (List[str]): List of events that should trigger notifications.
            Valid values are "timeout", "retry", and "error". Defaults to all three.

    Examples:
        Default configuration:
        >>> timeout_config = TimeoutConfig()
        >>> timeout_config.request
        600
        >>> timeout_config.retry_count
        1
        >>> timeout_config.notify_on
        ['timeout', 'retry', 'error']

        Custom configuration:
        >>> timeout_config = TimeoutConfig(
        ...     request=300,
        ...     retry_count=2,
        ...     notify_on=["timeout", "error"]
        ... )

        Invalid configuration (will raise ValueError):
        >>> try:
        ...     TimeoutConfig(request=1000)  # Exceeds maximum
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        'Error: Request timeout must be between 30 and 600 seconds'
    """

    request: int = field(default=600)  # Default 5 minutes
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

    This class defines settings for handling different types of files (images, videos, text)
    in the AI Battle framework. It includes validation logic to ensure files meet
    the requirements for size, format, and resolution.

    Attributes:
        path (str): Path to the file or directory on the filesystem. Must exist and be accessible.

        type (str): Type of file. Must be one of the supported file types defined in
            SUPPORTED_FILE_TYPES (e.g., "image", "video", "text").

        max_resolution (Optional[str], optional): Maximum resolution for image or video files.
            Can be specified as "WIDTHxHEIGHT" (e.g., "1920x1080") or "4K" for videos.
            Defaults to None, which uses the default maximum resolution for the file type.

        is_directory (bool, optional): Indicates if the path is a directory rather than a single file.
            When True, the directory will be scanned for files matching the file_pattern.
            Defaults to False.

        file_pattern (Optional[str], optional): Glob pattern for filtering files when is_directory is True.
            For example, "*.jpg" to include only JPEG files. Defaults to None, which includes all files.

    Implementation Notes:
        The __post_init__ method performs extensive validation including:
        - Checking if the file or directory exists
        - For single files:
          - Validating the file type against supported types
          - Checking the file extension against allowed extensions for the type
          - Validating file size against maximum allowed size
          - For images and videos, validating resolution if specified
        - For directories:
          - Checking if the directory exists and is accessible
          - Validating the file_pattern if provided

    Examples:
        Image file configuration:
        >>> file_config = FileConfig(
        ...     path="/path/to/image.jpg",
        ...     type="image",
        ...     max_resolution="1920x1080"
        ... )

        Video file configuration:
        >>> file_config = FileConfig(
        ...     path="/path/to/video.mp4",
        ...     type="video",
        ...     max_resolution="4K"  # Only valid for videos
        ... )

        Text file configuration:
        >>> file_config = FileConfig(
        ...     path="/path/to/document.txt",
        ...     type="text"
        ... )

        Directory configuration:
        >>> file_config = FileConfig(
        ...     path="/path/to/images",
        ...     type="image",
        ...     is_directory=True,
        ...     file_pattern="*.jpg"
        ... )
    """

    path: str
    type: str
    max_resolution: Optional[str] = field(default=None)
    is_directory: bool = field(default=False)
    file_pattern: Optional[str] = field(default=None)

    def __post_init__(self):
        if not os.path.exists(self.path):
            raise ValueError(f"Path not found: {self.path}")

        # Handle directory case
        if self.is_directory:
            if not os.path.isdir(self.path):
                raise ValueError(f"Path is not a directory: {self.path}")
            return

        # Handle single file case
        file_size = os.path.getsize(self.path)
        extension = os.path.splitext(self.path)[1].lower()

        # Validate file type
        if self.type not in SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported file type: {self.type}")

        type_config = SUPPORTED_FILE_TYPES[self.type]

        # Check extension
        if extension not in type_config["extensions"]:
            raise ValueError(
                f"Unsupported file extension {extension} for type {self.type}"
            )

        # Check file size
        if file_size > type_config["max_size"]:
            raise ValueError(
                f"File size {file_size} exceeds maximum {type_config['max_size']} for type {self.type}"
            )

        # Validate resolution for image/video
        if self.type in ["image", "video"] and self.max_resolution:
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
                    raise ValueError(f"Invalid resolution format: {requested_res}")


# @dataclass
@dataclass
class MultiFileConfig:
    """
    Configuration for handling multiple files.

    This class defines settings for handling multiple files or directories in the AI Battle framework.
    It provides options for specifying individual files, a directory to scan, or both.

    Attributes:
        files (List[FileConfig], optional): List of individual file configurations.
            Each item in the list should be a FileConfig object. Defaults to an empty list.

        directory (Optional[str], optional): Directory path to scan for files.
            If provided, all files in the directory matching the file_pattern will be included.
            Defaults to None.

        file_pattern (Optional[str], optional): Glob pattern for filtering files when directory is provided.
            For example, "*.jpg" to include only JPEG files. Defaults to None, which includes all files.

        max_files (int, optional): Maximum number of files to process from the directory.
            This limit helps prevent processing too many files which could cause performance issues.
            Defaults to 10.

        max_resolution (Optional[str], optional): Default maximum resolution for all files.
            This will be applied to all files unless overridden in individual FileConfig objects.
            Defaults to None.

    Implementation Notes:
        The __post_init__ method performs validation including:
        - Ensuring that either files or directory is provided
        - Initializing the files list if it's None
        - Validating the directory path if provided

    Examples:
        Multiple individual files:
        >>> multi_file_config = MultiFileConfig(
        ...     files=[
        ...         FileConfig(path="/path/to/image1.jpg", type="image"),
        ...         FileConfig(path="/path/to/image2.png", type="image")
        ...     ]
        ... )

        Directory with pattern:
        >>> multi_file_config = MultiFileConfig(
        ...     directory="/path/to/images",
        ...     file_pattern="*.jpg",
        ...     max_files=5,
        ...     max_resolution="1024x1024"
        ... )

        Combined approach:
        >>> multi_file_config = MultiFileConfig(
        ...     files=[FileConfig(path="/path/to/important.jpg", type="image")],
        ...     directory="/path/to/images",
        ...     file_pattern="*.jpg"
        ... )
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

    This class defines the configuration for an AI model in the AI Battle framework,
    including the model type, role, and optional persona. It includes validation
    to ensure the model type is supported and the role is valid.

    Attributes:
        type (str): The model type identifier, which should be one of the supported
            model types defined in SUPPORTED_MODELS. This typically includes the
            provider prefix (e.g., "claude-3-sonnet", "gemini-pro", "gpt-4").

        role (str): The role this model should play in the conversation. Must be
            either "human" or "assistant".

        persona (Optional[str], optional): Optional persona instructions for the model.
            This can be used to give the model a specific personality or behavior.
            Defaults to None.

    Implementation Notes:
        The __post_init__ method performs validation including:
        - Checking if the model type is supported by a known provider
        - For non-local models, validating the specific model variant
        - Ensuring the role is either "human" or "assistant"
        - Validating that the persona is a string if provided

        Local models (ollama, mlx) support any variant, while cloud models
        (claude, gemini, openai) must use specific supported variants.

    Examples:
        Assistant model configuration:
        >>> model_config = ModelConfig(
        ...     type="claude-3-sonnet",
        ...     role="assistant"
        ... )

        Human model configuration with persona:
        >>> model_config = ModelConfig(
        ...     type="gemini-pro",
        ...     role="human",
        ...     persona="You are a skeptical scientist who asks probing questions."
        ... )

        Local model configuration:
        >>> model_config = ModelConfig(
        ...     type="ollama:llama2",
        ...     role="assistant"
        ... )
    """

    type: str
    role: str
    persona: Optional[str] = field(default=None)

    def __post_init__(self):
        # Validate model type
        provider = next((p for p in SUPPORTED_MODELS if self.type.startswith(p)), None)
        if not provider:
            raise ValueError(f"Unsupported model type: {self.type}")

        if provider not in ["ollama", "mlx"]:  # Local models support any variant
            if self.type not in SUPPORTED_MODELS[provider]:
                raise ValueError(f"Unsupported model variant: {self.type}")

        # Validate role
        if self.role not in ["human", "assistant"]:
            raise ValueError(
                f"Invalid role: {self.role}. Must be 'human' or 'assistant'"
            )

        # Validate persona if provided
        if self.persona and not isinstance(self.persona, str):
            raise ValueError("Persona must be a string")


@dataclass
class DiscussionConfig:
    """
    Configuration for AI-to-AI or human-to-AI discussions.

    This class defines the overall configuration for a discussion between AI models
    in the AI Battle framework. It includes settings for the number of turns,
    participating models, discussion goal, optional input file, and timeout settings.

    Attributes:
        turns (int): Number of conversation turns to execute. Must be greater than 0.

        models (Dict[str, ModelConfig]): Dictionary mapping model names to their
            configurations. Must include at least two models, typically one with
            role="human" and one with role="assistant".

        goal (str): The discussion topic or objective. This is used to guide the
            conversation and provide context to the models.

        input_file (Optional[FileConfig], optional): Optional file to include as
            context for the discussion (e.g., an image to discuss or a text document
            to analyze). Defaults to None. This is kept for backward compatibility.

        input_files (Optional[MultiFileConfig], optional): Optional configuration for
            multiple files to include as context for the discussion. This allows for
            specifying multiple files or a directory of files to process. Defaults to None.
            When both input_file and input_files are provided, input_files takes precedence.

        timeouts (Optional[TimeoutConfig], optional): Timeout configuration for
            API requests during the discussion. Defaults to None, which uses
            the default TimeoutConfig.

    Implementation Notes:
        The __post_init__ method performs validation and conversion including:
        - Ensuring the number of turns is positive
        - Verifying at least two models are configured
        - Checking that a goal is provided
        - Converting dictionary configurations to proper ModelConfig objects
        - Converting dictionary timeout settings to a TimeoutConfig object

    Examples:
        Basic discussion configuration:
        >>> discussion_config = DiscussionConfig(
        ...     turns=5,
        ...     models={
        ...         "model1": ModelConfig(type="claude-3-sonnet", role="human"),
        ...         "model2": ModelConfig(type="gemini-pro", role="assistant")
        ...     },
        ...     goal="Discuss the implications of quantum computing on cryptography"
        ... )

        Configuration with input file and custom timeouts:
        >>> discussion_config = DiscussionConfig(
        ...     turns=10,
        ...     models={
        ...         "human": ModelConfig(type="gpt-4", role="human"),
        ...         "assistant": ModelConfig(type="claude-3-opus", role="assistant")
        ...     },
        ...     goal="Analyze the provided image and discuss its artistic elements",
        ...     input_file=FileConfig(path="/path/to/artwork.jpg", type="image"),
        ...     timeouts=TimeoutConfig(request=300, retry_count=2)
        ... )

        Configuration with multiple input files:
        >>> discussion_config = DiscussionConfig(
        ...     turns=8,
        ...     models={
        ...         "human": ModelConfig(type="gemini-pro", role="human"),
        ...         "assistant": ModelConfig(type="ollama-gemma3-12b", role="assistant")
        ...     },
        ...     goal="Compare and contrast the provided medical images",
        ...     input_files=MultiFileConfig(
        ...         files=[
        ...             FileConfig(path="/path/to/scan1.jpg", type="image"),
        ...             FileConfig(path="/path/to/scan2.jpg", type="image")
        ...         ],
        ...         max_resolution="1024x1024"
        ...     )
        ... )

        Configuration with directory of input files:
        >>> discussion_config = DiscussionConfig(
        ...     turns=5,
        ...     models={
        ...         "human": ModelConfig(type="claude-3-sonnet", role="human"),
        ...         "assistant": ModelConfig(type="gpt-4o", role="assistant")
        ...     },
        ...     goal="Analyze the collection of X-ray images",
        ...     input_files=MultiFileConfig(
        ...         directory="/path/to/xrays",
        ...         file_pattern="*.jpg",
        ...         max_files=10
        ...     )
        ... )
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
        # Convert dict models to ModelConfig objects
        if isinstance(self.models, dict):
            self.models = {
                name: ModelConfig(**config) if isinstance(config, dict) else config
                for name, config in self.models.items()
            }
        logger.info(f"Models: {self.models}")

        # Convert timeouts dict to TimeoutConfig
        if isinstance(self.timeouts, dict):
            self.timeouts = TimeoutConfig(**self.timeouts)
        elif self.timeouts is None:
            self.timeouts = TimeoutConfig()

        # Convert input_files dict to MultiFileConfig
        if isinstance(self.input_files, dict):
            # If the input_files dict has a 'files' key that contains a list of dicts,
            # convert each dict to a FileConfig object
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

        # Convert input_file dict to FileConfig
        if isinstance(self.input_file, dict):
            self.input_file = FileConfig(**self.input_file)
