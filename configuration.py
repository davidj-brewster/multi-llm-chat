import os
import yaml
import logging
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path
import json

# Custom exception classes for better error handling
class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""
    pass

class ConfigFileNotFoundError(ConfigurationError):
    """Raised when a configuration file is not found."""
    pass

class InvalidConfigFormatError(ConfigurationError):
    """Raised when the configuration format is invalid."""
    pass

class ModelConfigurationError(ConfigurationError):
    """Raised when there's an error in model configuration."""
    pass

class SystemInstructionsError(ConfigurationError):
    """Raised when there's an error loading system instructions."""
    pass

logger = logging.getLogger(__name__)

# Supported model configurations
SUPPORTED_MODELS = {
    "claude": ["claude", "haiku"],
    "gemini": ["gemini-2-flash-lite", "gemini-2-pro","gemini-2-reasoning","gemini-2.0-flash-exp", "gemini"],
    "openai": ["gpt-4-vision", "gpt-4o"],
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
        "extensions": [".mp4", ".mov", ".avi", ".webm"],
        "max_size": 100 * 1024 * 1024,  # 100MB
        "max_resolution": (3840, 2160)  # 4K
    },
    "text": {
        "extensions": [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".yaml", ".yml"],
        "max_size": 10 * 1024 * 1024  # 10MB
    }
}

@dataclass
class TimeoutConfig:
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

@dataclass
class FileConfig:
    path: str
    type: str
    max_resolution: Optional[str] = None

    def __init__(self, path: str, type: str, max_resolution: Optional[str] = None):
        """Initialize FileConfig with parameters."""
        self.path = path
        self.type = type
        self.max_resolution = max_resolution
        self.__post_init__()

    def __post_init__(self):
        try:
            if not os.path.exists(self.path):
                raise ConfigFileNotFoundError(f"File not found: {self.path}")

            try:
                file_size = os.path.getsize(self.path)
            except (OSError, IOError) as e:
                raise ConfigurationError(f"Error accessing file {self.path}: {e}")

            extension = os.path.splitext(self.path)[1].lower()

            # Validate file type
            if self.type not in SUPPORTED_FILE_TYPES:
                raise InvalidConfigFormatError(f"Unsupported file type: {self.type}")

            type_config = SUPPORTED_FILE_TYPES[self.type]

            # Check extension
            if extension not in type_config["extensions"]:
                raise InvalidConfigFormatError(
                    f"Unsupported file extension {extension} for type {self.type}. "
                    f"Supported extensions: {', '.join(type_config['extensions'])}"
                )

            # Check file size
            if file_size > type_config["max_size"]:
                readable_size = f"{type_config['max_size'] / (1024 * 1024):.1f}MB"
                raise InvalidConfigFormatError(
                    f"File size {file_size / (1024 * 1024):.1f}MB exceeds maximum {readable_size} for type {self.type}"
                )
        except ConfigurationError:
            # Re-raise specific configuration errors
            raise
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"Unexpected error validating file config: {e}")
            logger.debug(traceback.format_exc())
            raise ConfigurationError(f"Unexpected error validating file configuration: {e}")

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
                        raise InvalidConfigFormatError(
                            f"Requested resolution {width}x{height} exceeds maximum {max_width}x{max_height}"
                        )
                except ValueError:
                    raise InvalidConfigFormatError(f"Invalid resolution format: {requested_res}. Expected format: WIDTHxHEIGHT")

@dataclass
class ModelConfig:
    type: str
    role: str
    persona: Optional[str] = None

    def __post_init__(self):
        # Validate model type
        provider = next((p for p in SUPPORTED_MODELS if self.type.startswith(p)), None)
        if not provider:
            raise ModelConfigurationError(f"Unsupported model type: {self.type}. Supported providers: {', '.join(SUPPORTED_MODELS.keys())}")

        if provider not in ["ollama", "mlx"]:  # Local models support any variant
            if self.type not in SUPPORTED_MODELS[provider]:
                raise ModelConfigurationError(f"Unsupported model variant: {self.type}. Supported variants: {', '.join(SUPPORTED_MODELS[provider])}")

        # Validate role
        if self.role not in ["human", "assistant"]:
            raise ModelConfigurationError(f"Invalid role: {self.role}. Must be 'human' or 'assistant'")

        # Validate persona if provided
        if self.persona and not isinstance(self.persona, str):
            raise ValueError("Persona must be a string")

@dataclass
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

def load_system_instructions() -> Dict:
    """Load system instructions from docs/system_instructions.md"""
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
    """Load and validate YAML configuration file"""
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
    except (TypeError, ValueError) as e:
        raise InvalidConfigFormatError(f"Invalid configuration format in {path}: {e}")

def detect_model_capabilities(model_config: Union[ModelConfig, str]) -> Dict[str, bool]:
    """Detect model capabilities based on type"""
    capabilities = {
        "vision": True,
        "streaming": False,
        "function_calling": False,
        "code_understanding": False
    }

    # Extract model type from ModelConfig or use string directly
    try:
        model_type = model_config.type if isinstance(model_config, ModelConfig) else model_config

        if not model_type:
            logger.warning("Empty model type provided to detect_model_capabilities")
            return capabilities

        # Vision-capable cloud models
        vision_models = [
            "claude",
            "gpt-4o",
            "sonnet",
            "openai",
            "gemini-2-pro",
            "gemini-2-reasoning",
            "gemini-2-flash-lite",
            "gemini-2.0-flash-exp",
            "chatgpt-latest",
            "gemini"
        ]

        # Ollama vision-capable models
        ollama_vision_models = [
            "gemma3", "llava", "bakllava", "moondream", "llava-phi3", "gpt", "chatgpt"
        ]

        # Check if it's an Ollama model with vision support
        if "ollama" in model_type.lower():
            for vision_model in ollama_vision_models:
                if vision_model in model_type.lower():
                    capabilities["vision"] = True
                    break
        else:
            # Check cloud model vision capabilities
            for prefix in vision_models:
                if model_type.startswith(prefix):
                    capabilities["vision"] = True
                    break

        # Streaming capability
        if model_type.startswith(("claude", "gpt", "chatgpt", "gemini", "gemma","gemini-2.0-flash-exp")):
            capabilities["vision"] = True
            capabilities["streaming"] = True

        # Function calling capability
        if "gpt-4" in model_type or "claude" in model_type:
            capabilities["function_calling"] = True

        return capabilities
    except Exception as e:
        logger.error(f"Error detecting model capabilities: {e}")
        return capabilities  # Return default capabilities on error
