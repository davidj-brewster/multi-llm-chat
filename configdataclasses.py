"""Configuration integration module for AI Battle framework"""
import os
from typing import Dict, List, Optional

# Constants referenced in the code but not defined in the snippet
# These would typically be defined in this file or imported
SUPPORTED_FILE_TYPES = {}  # Placeholder
SUPPORTED_MODELS = {}      # Placeholder

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
    request: int = 600  # Default 5 minutes
    retry_count: int = 1
    notify_on: List[str] = None

    def __post_init__(self):
        if self.request < 30 or self.request > 600:
            raise ValueError("Request timeout must be between 30 and 600 seconds")
        if self.retry_count < 0 or self.retry_count > 5:
            raise ValueError("Retry count must be between 0 and 5")
        if self.notify_on is None:
            self.notify_on = ["timeout", "retry", "error"]
        valid_events = ["timeout", "retry", "error"]
        invalid_events = [e for e in self.notify_on if e not in valid_events]
        if invalid_events:
            raise ValueError(f"Invalid notification events: {invalid_events}")

class FileConfig:
    """
    Configuration for file handling and validation.
    
    This class defines settings for handling different types of files (images, videos, text)
    in the AI Battle framework. It includes validation logic to ensure files meet
    the requirements for size, format, and resolution.
    
    Attributes:
        path (str): Path to the file on the filesystem. Must exist and be accessible.
        
        type (str): Type of file. Must be one of the supported file types defined in
            SUPPORTED_FILE_TYPES (e.g., "image", "video", "text").
        
        max_resolution (Optional[str], optional): Maximum resolution for image or video files.
            Can be specified as "WIDTHxHEIGHT" (e.g., "1920x1080") or "4K" for videos.
            Defaults to None, which uses the default maximum resolution for the file type.
    
    Implementation Notes:
        The __post_init__ method performs extensive validation including:
        - Checking if the file exists
        - Validating the file type against supported types
        - Checking the file extension against allowed extensions for the type
        - Validating file size against maximum allowed size
        - For images and videos, validating resolution if specified
    
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
    """
    path: str
    type: str
    max_resolution: Optional[str] = None

    def __post_init__(self):
        if not os.path.exists(self.path):
            raise ValueError(f"File not found: {self.path}")
        
        file_size = os.path.getsize(self.path)
        extension = os.path.splitext(self.path)[1].lower()
        
        # Validate file type
        if self.type not in SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported file type: {self.type}")
        
        type_config = SUPPORTED_FILE_TYPES[self.type]
        
        # Check extension
        if extension not in type_config["extensions"]:
            raise ValueError(f"Unsupported file extension {extension} for type {self.type}")
        
        # Check file size
        if file_size > type_config["max_size"]:
            raise ValueError(f"File size {file_size} exceeds maximum {type_config['max_size']} for type {self.type}")
        
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
                        raise ValueError(f"Requested resolution {width}x{height} exceeds maximum {max_width}x{max_height}")
                except ValueError:
                    raise ValueError(f"Invalid resolution format: {requested_res}")

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
    persona: Optional[str] = None

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
            raise ValueError(f"Invalid role: {self.role}. Must be 'human' or 'assistant'")
        
        # Validate persona if provided
        if self.persona and not isinstance(self.persona, str):
            raise ValueError("Persona must be a string")

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
            to analyze). Defaults to None.
        
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
    """
    turns: int
    models: Dict[str, ModelConfig]
    goal: str
    input_file: Optional[FileConfig] = None
    timeouts: Optional[TimeoutConfig] = None

    def __post_init__(self):
        if self.turns < 1:
            raise ValueError("Turns must be greater than 0")
        
        if len(self.models) < 2:
            raise ValueError("At least two models must be configured")
        
        if not self.goal:
            raise ValueError("Goal must be provided")
        
        # Convert dict models to ModelConfig objects
        if isinstance(self.models, dict):
            self.models = {
                name: ModelConfig(**config) if isinstance(config, dict) else config
                for name, config in self.models.items()
            }
        
        # Convert timeouts dict to TimeoutConfig
        if isinstance(self.timeouts, dict):
            self.timeouts = TimeoutConfig(**self.timeouts)
        elif self.timeouts is None:
            self.timeouts = TimeoutConfig()


class DiscussionConfig:
    turns: int
    models: Dict[str, ModelConfig]
    goal: str
    input_file: Optional[FileConfig] = None
    timeouts: Optional[TimeoutConfig] = None

    def __post_init__(self):
        if self.turns < 1:
            raise ValueError("Turns must be greater than 0")
        
        if len(self.models) < 2:
            raise ValueError("At least two models must be configured")
        
        if not self.goal:
            raise ValueError("Goal must be provided")
        
        # Convert dict models to ModelConfig objects
        if isinstance(self.models, dict):
            self.models = {
                name: ModelConfig(**config) if isinstance(config, dict) else config
                for name, config in self.models.items()
            }
        
        # Convert timeouts dict to TimeoutConfig
        if isinstance(self.timeouts, dict):
            self.timeouts = TimeoutConfig(**self.timeouts)
        elif self.timeouts is None:
            self.timeouts = TimeoutConfig()
