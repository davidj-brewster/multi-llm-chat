"""Configuration integration module for AI Battle framework"""
import os
import yaml
import logging
import asyncio
import traceback
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path
import json

# Custom exception classes for better error handling
class ConfigIntegrationError(Exception):
    """Base exception for configuration integration errors."""
    pass

class ConfigFileNotFoundError(ConfigIntegrationError):
    """Raised when a configuration file is not found."""
    pass

class InvalidConfigFormatError(ConfigIntegrationError):
    """Raised when the configuration format is invalid."""
    pass

class ModelConfigurationError(ConfigIntegrationError):
    """Raised when there's an error in model configuration."""
    pass

class SystemInstructionsError(ConfigIntegrationError):
    """Raised when there's an error loading system instructions."""
    pass

logger = logging.getLogger(__name__)

# Supported model configurations
SUPPORTED_MODELS = {
    "claude": ["claude-3-sonnet", "claude-3-haiku"],
    "gemini": ["gemini-pro", "gemini-pro-vision"],
    "openai": ["gpt-4-vision", "gpt-4"],
    "ollama": ["*"],  # All Ollama models supported
    "mlx": ["*"]      # All MLX models supported
}

# File type configurations
SUPPORTED_FILE_TYPES = {
    "image": {
        "extensions": [".jpg", ".jpeg", ".png", ".gif", ".webp"],
        "max_size": 20 * 1024 * 1024,  # 20MB
        "max_resolution": (8192, 8192)
    },
    "video": {
        "extensions": [".mp4", ".mov", ".avi", ".webm", ".mkv"],
        "max_size": 200 * 1024 * 1024,  # 200MB
        "max_resolution": (3840, 2160)  # 4K
    },
    "text": {
        "extensions": [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".yaml", ".yml"],
        "max_size": 100 * 1024 * 1024  # 10M0B
    }
}

@dataclass
class TimeoutConfig:
    """
    Configuration for request timeouts and retry behavior.

    This dataclass defines timeout settings for API requests, including the maximum
    request duration, retry behavior, and notification preferences. It includes
    validation to ensure the settings are within acceptable ranges.

    Attributes:
        request (int): Maximum request duration in seconds. Must be between
            30 and 600 seconds (10 minutes). Defaults to 600 seconds.

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
    """
    request: int = 600  # Default 10 minutes
    retry_count: int = 1
    notify_on: List[str] = None

    def __post_init__(self):
        try:
            if self.request < 30 or self.request > 600:
                raise InvalidConfigFormatError(
                    f"Request timeout must be between 30 and 600 seconds, got {self.request}"
                )
            if self.retry_count < 0 or self.retry_count > 5:
                raise InvalidConfigFormatError(
                    f"Retry count must be between 0 and 5, got {self.retry_count}"
                )
            if self.notify_on is None:
                self.notify_on = ["timeout", "retry", "error"]
            valid_events = ["timeout", "retry", "error"]
            invalid_events = [e for e in self.notify_on if e not in valid_events]
            if invalid_events:
                raise InvalidConfigFormatError(f"Invalid notification events: {invalid_events}. Valid events are: {', '.join(valid_events)}")
        except Exception as e:
            logger.error(f"Error validating TimeoutConfig: {e}")
            raise

@dataclass
class FileConfig:
    """
    Configuration for file handling and validation.

    This dataclass defines settings for handling different types of files (images, videos, text)
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
        try:
            if not os.path.exists(self.path):
                raise ConfigFileNotFoundError(f"File not found: {self.path}")

            try:
                file_size = os.path.getsize(self.path)
            except (OSError, IOError) as e:
                raise ConfigIntegrationError(f"Error accessing file {self.path}: {e}")

            extension = os.path.splitext(self.path)[1].lower()

            # Validate file type
            if self.type not in SUPPORTED_FILE_TYPES:
                supported_types = ", ".join(SUPPORTED_FILE_TYPES.keys())
                raise InvalidConfigFormatError(f"Unsupported file type: {self.type}. Supported types: {supported_types}")

            type_config = SUPPORTED_FILE_TYPES[self.type]

            # Check extension
            if extension not in type_config["extensions"]:
                supported_extensions = ", ".join(type_config["extensions"])
                raise InvalidConfigFormatError(
                    f"Unsupported file extension {extension} for type {self.type}. "
                    f"Supported extensions: {supported_extensions}"
                )

            # Check file size
            if file_size > type_config["max_size"]:
                readable_size = f"{type_config['max_size'] / (1024 * 1024):.1f}MB"
                raise InvalidConfigFormatError(
                    f"File size {file_size / (1024 * 1024):.1f}MB exceeds maximum {readable_size} for type {self.type}"
                )

            # Validate resolution for image/video
            if self.type in ["image", "video"] and self.max_resolution:
                max_width, max_height = type_config["max_resolution"]
                requested_res = self.max_resolution.upper()
                if requested_res == "4K":
                    if self.type != "video":
                        raise InvalidConfigFormatError("4K resolution only supported for video files")
                elif "X" in requested_res:
                    try:
                        width, height = map(int, requested_res.split("X"))
                        if width > max_width or height > max_height:
                            raise InvalidConfigFormatError(f"Requested resolution {width}x{height} exceeds maximum {max_width}x{max_height}")
                    except ValueError:
                        raise InvalidConfigFormatError(f"Invalid resolution format: {requested_res}. Expected format: WIDTHxHEIGHT")
        except ConfigIntegrationError:
            # Re-raise specific configuration errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error validating file config: {e}")
            logger.debug(traceback.format_exc())
            raise ConfigIntegrationError(f"Unexpected error validating file configuration: {e}")

@dataclass
class ModelConfig:
    """
    Configuration for AI model settings and behavior.

    This dataclass defines the configuration for an AI model in the AI Battle framework,
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
        if not provider and self.type:
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

@dataclass
class DiscussionConfig:
    """
    Configuration for AI-to-AI or human-to-AI discussions.

    This dataclass defines the overall configuration for a discussion between AI models
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

def load_system_instructions() -> Dict:
    """
    Load system instructions from docs/system_instructions.md.

    This function reads the system instructions markdown file and extracts YAML blocks
    that define instruction templates. These templates can be referenced in model
    configurations to provide standardized system instructions.

    The function parses all YAML code blocks in the markdown file and combines them
    into a single dictionary of instruction templates.

    Returns:
        Dict: A dictionary mapping template names to their instruction content.
            For example:
            {
                "human_persona": {"role": "human", "instructions": "..."},
                "assistant_persona": {"role": "assistant", "instructions": "..."}
            }

    Raises:
        FileNotFoundError: If the system instructions file is not found.

    Examples:
        >>> instructions = load_system_instructions()
        >>> "human_persona" in instructions
        True
        >>> isinstance(instructions["human_persona"], dict)
        True

    Implementation Notes:
        - The function looks for ```yaml blocks in the markdown file
        - Each block should contain a valid YAML dictionary
        - All dictionaries are merged into a single result dictionary
        - YAML parsing errors are logged but don't stop processing
    """
    instructions_path = Path("docs/system_instructions.md")
    if not instructions_path.exists():
        raise SystemInstructionsError("System instructions file not found at docs/system_instructions.md")

    content = instructions_path.read_text()

    # Extract YAML blocks
    yaml_blocks = []
    in_yaml = False
    current_block = []

    for line in content.split("\n"):
        if line.strip() == "```yaml":
            in_yaml = True
            current_block = []
        elif line.strip() == "```" and in_yaml:
            in_yaml = False
            if current_block:
                yaml_blocks.append("\n".join(current_block))
        elif in_yaml:
            current_block.append(line)

    # Parse YAML blocks
    instructions = {}
    for block in yaml_blocks:
        try:
            data = yaml.safe_load(block)
            if isinstance(data, dict):
                instructions.update(data)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing system instructions YAML block: {e}")
            logger.debug(f"Problematic YAML block: {block}")
            continue

    return instructions

def load_config(path: str) -> DiscussionConfig:
    """
    Load and validate a YAML configuration file for AI discussions.

    This function reads a YAML configuration file, validates its structure, processes
    any instruction templates, and creates a DiscussionConfig object. It handles
    template substitution for model instructions, allowing for parameterized system
    prompts.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        DiscussionConfig: A fully initialized and validated discussion configuration
            object that can be used to run AI conversations.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If the YAML format is invalid or the configuration structure
            doesn't match the expected format.

    Examples:
        >>> config = load_config("examples/configs/debate_config.yaml")
        >>> config.turns
        5
        >>> len(config.models) >= 2
        True
        >>> config.goal
        'Discuss the ethical implications of AI'

    Implementation Notes:
        - The configuration file must contain a 'discussion' section
        - Model configurations can reference instruction templates
        - Template parameters are substituted using {parameter_name} syntax
        - The function loads system instructions to resolve template references
    """
    if not os.path.exists(path):
        raise ConfigFileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path) as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise InvalidConfigFormatError(f"Invalid YAML format in {path}: {e}")

    if not isinstance(config_dict, dict) or "discussion" not in config_dict:
        raise ValueError("Configuration must contain a 'discussion' section")

    try:
        # Load system instructions
        system_instructions = load_system_instructions()

        # Process model configurations
        for model_name, model_config in config_dict["discussion"]["models"].items():
            # Replace template references with actual instructions
            if "instructions" in model_config:
                template_name = model_config["instructions"].get("template")
                if template_name in system_instructions:
                    template = system_instructions[template_name]
                    params = model_config["instructions"].get("params", {})

                    # Replace template parameters
                    instruction_text = json.dumps(template)
                    for key, value in params.items():
                        instruction_text = instruction_text.replace(f"{{{key}}}", str(value))

                    model_config["persona"] = json.loads(instruction_text)

        return DiscussionConfig(**config_dict["discussion"])
    except (TypeError, ValueError, KeyError) as e:
        logger.error(f"Error processing configuration: {e}")
        raise InvalidConfigFormatError(f"Invalid configuration format in {path}: {e}")

def detect_model_capabilities(model_config: ModelConfig) -> Dict[str, bool]:
    """
    Detect and return capabilities of an AI model based on its type.

    This function analyzes the model type to determine which capabilities
    (vision, streaming, function calling) are supported by the model. It uses
    a rule-based approach based on known model capabilities.

    Args:
        model_config (ModelConfig): ModelConfig object or string containing the model type.
            If a string is provided, it's treated as the model type directly.

    Returns:
        Dict[str, bool]: Dictionary mapping capability names to boolean values
            indicating whether each capability is supported. The capabilities include:
            - "vision": Support for image/video processing
            - "streaming": Support for streaming responses
            - "function_calling": Support for function/tool calling

    Examples:
        Checking capabilities of a vision model:
        >>> model_config = ModelConfig(type="gemini-pro-vision", role="assistant")
        >>> capabilities = detect_model_capabilities(model_config)
        >>> capabilities["vision"]
        True
        >>> capabilities["streaming"]
        False

        Checking capabilities of a text-only model:
        >>> capabilities = detect_model_capabilities(ModelConfig(type="claude-3-sonnet", role="assistant"))
        >>> capabilities["vision"]
        False
        >>> capabilities["streaming"]
        True
        >>> capabilities["function_calling"]
        True

    Implementation Notes:
        - Vision capability is determined by model name containing "vision"
        - Streaming capability is determined by model provider (Claude, GPT)
        - Function calling capability is limited to specific models
    """
    capabilities = {
        "vision": False,
        "streaming": False,
        "function_calling": False
    }

    try:
        if not hasattr(model_config, 'type') or not model_config.type:
            logger.warning("Invalid model_config provided to detect_model_capabilities")
            return capabilities

        if model_config.type in ["gemini-pro-vision", "gpt-4-vision"]:
            capabilities["vision"] = True

        if model_config.type.startswith(("claude", "gpt")):
            capabilities["streaming"] = True

        if model_config.type in ["gpt-4", "claude-3-sonnet"]:
            capabilities["function_calling"] = True
    except Exception as e:
        logger.error(f"Error detecting model capabilities: {e}")
        # Return default capabilities on error
    return capabilities

def run_from_config(config_path: str) -> None:
    """
    Run an AI discussion based on settings in a configuration file.

    This function loads a configuration file, initializes a ConversationManager
    with the specified settings, and runs a conversation between the configured
    models. The conversation is then saved to an HTML file with a timestamped
    filename.

    Args:
        config_path (str): Path to the YAML configuration file containing the
            discussion configuration.

    Returns:
        None: The function doesn't return a value, but saves the conversation
            to an HTML file as a side effect.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If the configuration is invalid.
        ImportError: If there are issues importing required modules.

    Examples:
        Running a discussion from a configuration file:
        >>> run_from_config("examples/configs/debate_config.yaml")
        # Creates a file like: conversation-config_debate_0323-0337.html

        Running with a custom configuration:
        >>> # First create a config file
        >>> with open("custom_config.yaml", "w") as f:
        ...     f.write('''
        ...     discussion:
        ...       turns: 5
        ...       goal: "Discuss the future of AI"
        ...       models:
        ...         human_model:
        ...           type: "gemini-pro"
        ...           role: "human"
        ...         ai_model:
        ...           type: "claude-3-sonnet"
        ...           role: "assistant"
        ...     ''')
        >>> run_from_config("custom_config.yaml")

    Implementation Notes:
        - Uses asyncio to run the conversation asynchronously
        - Automatically identifies human and assistant models from the config
        - Saves the conversation with a sanitized filename based on the goal
        - Uses the ConversationManager from ai_battle.py to manage the conversation
    """
    try:
        config = load_config(config_path)

        # Import here to avoid circular imports
        try:
            from ai_battle import ConversationManager
        except ImportError as e:
            logger.error(f"Failed to import ConversationManager: {e}")
            raise ImportError(f"Failed to import ConversationManager: {e}. Make sure ai_battle.py is in the current directory.")

        # Create manager
        manager = ConversationManager(
            domain=config.goal,
            mode="ai-ai",  # Default to ai-ai mode for config-based initialization
            human_delay=20.0,
            min_delay=10
        )

        # Get model names
        try:
            human_model = next(name for name, model in config.models.items()
                              if model.role == "human")
            ai_model = next(name for name, model in config.models.items()
                           if model.role == "assistant")
        except StopIteration:
            logger.error("Configuration must include both a human and an assistant model")
            raise InvalidConfigFormatError("Configuration must include both a human model (role='human') and an assistant model (role='assistant')")

        # Run conversation
        try:
            conversation = asyncio.run(manager.run_conversation(
                initial_prompt=config.goal,
                human_model=human_model,
                ai_model=ai_model,
                mode="ai-ai",
                human_system_instruction=config.models[human_model].persona,
                ai_system_instruction=config.models[ai_model].persona,
                rounds=config.turns
            ))

            # Save conversation
            safe_prompt = manager._sanitize_filename_part(config.goal)
            time_stamp = datetime.datetime.now().strftime("%m%d-%H%M")
            filename = f"conversation-config_{safe_prompt}_{time_stamp}.html"
            manager.save_conversation(conversation, filename=filename, human_model=human_model, ai_model=ai_model, mode="ai-ai")
        except Exception as e:
            logger.error(f"Error running conversation: {e}")
            logger.debug(traceback.format_exc())
            raise
    except Exception as e:
        logger.error(f"Failed to run from config {config_path}: {e}")
        raise
