"""Configuration integration module for AI Battle framework"""
import os
import yaml
import logging
import asyncio
import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path
import json

class TimeoutConfig:
    request: int = 300  # Default 5 minutes
    retry_count: int = 3
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
