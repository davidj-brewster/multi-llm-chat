import os
import yaml
import logging
import traceback
from dataclasses import dataclass
from typing import Dict, List, Union
# Ensure MultiFileConfig and DiscussionConfig are imported correctly if they also move
# or if configuration.py is meant to be the source for them too.
# For now, assuming they are correctly resolved from their original location or this one.
from configdataclasses import DiscussionConfig
from pathlib import Path
import json

# Custom exception classes for better error handling
class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""

class ConfigFileNotFoundError(ConfigurationError):
    """Raised when a configuration file is not found."""

class InvalidConfigFormatError(ConfigurationError):
    """Raised when the configuration format is invalid."""

class ModelConfigurationError(ConfigurationError):
    """Raised when there's an error in model configuration."""

class SystemInstructionsError(ConfigurationError):
    """Raised when there's an error loading system instructions."""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from configdataclasses import ModelConfig

@dataclass
class TimeoutConfig:
    """Configuration for request timeouts, retries, and notification events."""
    request: int = 600
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

# Removed local FileConfig definition
# Removed local ModelConfig definition

# This class seems to be an old or alternative version.
# DiscussionConfig from configdataclasses.py is used by load_config.
# If DiscussionConfigOld is to be kept and used, it must be updated to use the
# imported FileConfig and ModelConfig from .configdataclasses
# For now, assuming it's potentially deprecated or will be handled separately if still needed.
# If it were to be updated:
# class DiscussionConfigOld:
#     turns: int
#     models: Dict[str, ModelConfig] # This will now refer to the imported ModelConfig
#     goal: str
#     input_file: Optional[FileConfig] = None # This will now refer to the imported FileConfig
#     input_files: Optional[MultiFileConfig] # This already refers to configdataclasses.MultiFileConfig
#     timeouts: Optional[TimeoutConfig] = None

#     def __post_init__(self):
#         if self.turns < 1:
#             raise ValueError("Turns must be greater than 0")
#         if len(self.models) < 2:
#             raise ValueError("At least two models must be configured")
#         if not self.goal:
#             raise ValueError("Goal must be provided")
#         if isinstance(self.models, dict):
#             self.models = {
#                 # Ensure this uses the imported ModelConfig for instantiation
#                 name: ModelConfig(**config) if isinstance(config, dict) else config
#                 for name, config in self.models.items()
#             }
#         if isinstance(self.timeouts, dict):
#             self.timeouts = TimeoutConfig(**self.timeouts)
#         elif self.timeouts is None:
#             self.timeouts = TimeoutConfig()

# Marking DiscussionConfigOld as potentially deprecated by commenting out for now.
# If it's needed, it should be reviewed and updated.
class DiscussionConfigOld:
    """Deprecated placeholder for the legacy discussion configuration."""
    pass

def load_system_instructions() -> Dict:
    """Load system instructions from the markdown file and parse embedded YAML blocks."""
    instructions_path = Path("docs/system_instructions.md")
    if not instructions_path.exists():
        raise SystemInstructionsError(
            "System instructions file not found at docs/system_instructions.md"
        )
    content = instructions_path.read_text(encoding="utf-8")
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

def load_config(path: str) -> DiscussionConfig: # Returns DiscussionConfig from configdataclasses.py
    """Load and validate a discussion configuration from a YAML file."""
    if not os.path.exists(path):
        raise ConfigFileNotFoundError(f"Configuration file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise InvalidConfigFormatError(f"Invalid YAML format in {path}: {e}") from e
    if not isinstance(config_dict, dict) or "discussion" not in config_dict:
        raise ValueError("Configuration must contain a 'discussion' section")
    try:
        system_instructions = load_system_instructions()
        for _model_name, model_config_dict in config_dict["discussion"]["models"].items():
            if "instructions" in model_config_dict:
                template_name = model_config_dict["instructions"].get("template")
                if template_name in system_instructions:
                    template = system_instructions[template_name]
                    params = model_config_dict["instructions"].get("params", {})

                    # instruction_text = json.dumps(template)
                    # Ensure template is a string before replacing
                    if isinstance(template, str):
                        instruction_text = template
                        for key, value in params.items():
                            instruction_text = instruction_text.replace(f"{{{key}}}", str(value))
                        model_config_dict["persona"] = instruction_text
                    elif isinstance(template, (dict, list)):
                        # If template is complex, dump to JSON string, replace, then parse back.
                        # This is kept for compatibility but string templates are preferred.
                        json_template_str = json.dumps(template)
                        for key, value in params.items():
                            json_template_str = json_template_str.replace(f"{{{key}}}", str(value))
                        # The persona in ModelConfig (from configdataclasses) expects a string.
                        # If the result of this is a complex structure, it will fail ModelConfig validation.
                        # Forcing it to string here, or ensuring templates resolve to strings.
                        # This resolved structure will be passed to ModelConfig's persona.
                        resolved_persona_structure = json.loads(json_template_str)
                        # If persona must be string, convert complex structure to string (e.g., JSON string)
                        # This addresses the persona type mismatch.
                        if isinstance(resolved_persona_structure, (dict, list)):
                            model_config_dict["persona"] = json.dumps(resolved_persona_structure)
                        else:
                            model_config_dict["persona"] = str(resolved_persona_structure)

        # Ensure that DiscussionConfig (from .configdataclasses) is used for instantiation
        return DiscussionConfig(**config_dict["discussion"])
    except (TypeError, ValueError) as e:
        raise InvalidConfigFormatError(
            f"Invalid configuration format in {path}: {e} {traceback.format_exc()}"
        ) from e

def detect_model_capabilities(model_config_input: Union[ModelConfig, str]) -> Dict[str, bool]:
    """Detect supported capabilities for a given model based on its type string."""
    capabilities = {
        "vision": False,  # Default to False, enable specifically
        "streaming": False,
        "function_calling": False,
        "code_understanding": False, # Default, can be inferred if needed
        "advanced_reasoning": False,
    }
    try:
        model_type_str = (
            model_config_input.type if isinstance(model_config_input, ModelConfig) else model_config_input
        )
        if not model_type_str:
            logger.warning("Empty model type provided to detect_model_capabilities")
            return capabilities

        mt_lower = model_type_str.lower()

        # Vision Capabilities
        # General keywords for cloud providers known for vision
        cloud_vision_keywords = ["claude", "gpt-4o", "gemini", "o1", "o3", "vision", "gpt-4.1"]
        # Specific Ollama vision models (keywords within their names)
        ollama_vision_keywords = ["llava", "bakllava", "moondream", "gemma3", "llava-phi3", "vision", "mistral-medium"]

        if any(p in mt_lower for p in cloud_vision_keywords):
            capabilities["vision"] = True
        elif "ollama" in mt_lower:
            if any(vm_keyword in mt_lower for vm_keyword in ollama_vision_keywords):
                capabilities["vision"] = True
        # Add other specific checks if a model provider/name doesn't fit above general rules

        # Streaming Capability (based on common prefixes)
        # Assuming most modern models from these providers support streaming.
        if mt_lower.startswith(("claude", "gpt", "chatgpt", "gemini", "o1", "o3", "o4", "ollama-")): # Added ollama
            capabilities["streaming"] = True

        # Function Calling Capability
        # Gemini 1.5 models support function calling.
        # Claude 3 models support function calling.
        # GPT-4+ models support function calling.
        if "gpt-4" in mt_lower or "claude-3" in mt_lower or "gemini-1.5" in mt_lower or "gemini-2.5" in mt_lower:
            capabilities["function_calling"] = True

        # Advanced Reasoning Capability (for models with explicit reasoning/thinking parameters)
        # Claude 3.x, OpenAI o1/o3/o4-mini, and Ollama thinking-capable models
        ollama_thinking_keywords = ["qwen3", "deepseek-r1", "gpt-oss", "phi4-reasoning", "granite-reasoning"]
        if "claude-3" in mt_lower or \
           any(variant in mt_lower for variant in ["o1", "o3", "o4-mini"]) or \
           ("ollama" in mt_lower and any(kw in mt_lower for kw in ollama_thinking_keywords)):
            capabilities["advanced_reasoning"] = True

        # Code Understanding (Most LLMs have some level, this flag is for notable proficiency or specific features)
        # This can be highly model-specific and might require more granular checks if used to gate features.
        # For now, we'll assume general capability unless specific models are known to excel or lack it.
        if any(p in mt_lower for p in ["claude", "gemini", "gpt-4", "o1", "o3", "o4"]): # General high-tier models
            capabilities["code_understanding"] = True

        return capabilities
    except Exception as e:
        logger.error(f"Error detecting model capabilities for '{model_config_input}': {e}")
        return capabilities # Return default capabilities on error
