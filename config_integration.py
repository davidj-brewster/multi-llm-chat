"""Configuration integration module for AI Battle framework"""
import os
import yaml
import logging
import asyncio
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

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
    request: int = 600  # Default 10 minutes
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

@dataclass
class ModelConfig:
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
        raise FileNotFoundError("System instructions file not found")
    
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
            logger.error(f"Error parsing system instructions YAML: {e}")
            continue
    
    return instructions

def load_config(path: str) -> DiscussionConfig:
    """Load and validate YAML configuration file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    try:
        with open(path) as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {e}")
    
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
        raise ValueError(f"Invalid configuration format: {e}")

def detect_model_capabilities(model_config: ModelConfig) -> Dict[str, bool]:
    """Detect model capabilities based on type"""
    capabilities = {
        "vision": False,
        "streaming": False,
        "function_calling": False
    }
    
    if model_config.type in ["gemini-pro-vision", "gpt-4-vision"]:
        capabilities["vision"] = True
    
    if model_config.type.startswith(("claude", "gpt")):
        capabilities["streaming"] = True
    
    if model_config.type in ["gpt-4", "claude-3-sonnet"]:
        capabilities["function_calling"] = True
    
    return capabilities

def run_from_config(config_path: str) -> None:
    """Run discussion from configuration file"""
    config = load_config(config_path)
    
    # Import here to avoid circular imports
    from ai_battle import ConversationManager
    
    # Create manager
    manager = ConversationManager(
        domain=config.goal,
        mode="ai-ai",  # Default to ai-ai mode for config-based initialization
        human_delay=20.0,
        min_delay=10
    )
    
    # Get model names
    human_model = next(name for name, model in config.models.items() 
                      if model.role == "human")
    ai_model = next(name for name, model in config.models.items() 
                   if model.role == "assistant")
    
    # Run conversation
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