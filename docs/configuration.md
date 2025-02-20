# Configuration System Documentation

## Overview

The configuration system uses YAML to define discussion parameters, model settings, and file inputs. This document outlines the schema, validation rules, and implementation details.

## Schema

```yaml
discussion:
  # Number of back-and-forth exchanges
  turns: 3  
  
  # Model configurations
  models:
    model1:
      type: "claude-3-sonnet"  # Model identifier
      role: "human"            # Role in conversation (human/assistant)
      persona: |               # Persona-based system instructions
        You are a neurological radiologist with the following characteristics:
        - 15 years of clinical experience
        - Specialization in advanced imaging techniques
        - Research focus on early detection patterns
        - Known for innovative diagnostic approaches
        [Additional persona details integrated into role prompt]
    
    model2:
      type: "gemini-pro"
      role: "assistant"
      persona: |
        You are an AI assistant with the following characteristics:
        - Deep expertise in medical imaging analysis
        - Collaborative approach to diagnosis
        - Evidence-based reasoning methodology
        [Additional persona details integrated into role prompt]
  
  # Timeout settings
  timeouts:
    request: 300             # Request timeout in seconds
    retry_count: 3           # Number of retries
    notify_on:
      - timeout             # Notify on timeout
      - retry              # Notify on retry
      - error              # Notify on error
  
  # Input file configuration
  input_file:
    path: "./scan.mp4"      # Path to input file
    type: "video"           # image, video, or text
    max_resolution: "4K"    # Maximum resolution to maintain
    
  # Discussion objective
  goal: |
    Analyze the provided brain scan video sequence and discuss potential abnormalities,
    focusing on regions of concern and possible diagnostic implications.
```

## Validation Rules

### Required Fields
- `discussion.turns` (integer, > 0)
- `discussion.models` (object with at least 2 models)
- `discussion.goal` (string, non-empty)

### Model Configuration
- `type`: Must be one of the supported models:
  - Claude: claude-3-sonnet, claude-3-haiku
  - Gemini: gemini-pro, gemini-pro-vision
  - OpenAI: gpt-4-vision, gpt-4
  - Local: ollama-*, mlx-*
- `role`: Must be either "human" or "assistant"
- `persona`: Optional, but if provided must be non-empty string

### File Configuration
- Supported types:
  - Images: jpg, jpeg, png, gif, webp
  - Videos: mp4, mov, avi, webm
  - Text: txt, md, py, js, etc.
- Resolution settings:
  - Images: up to 8192x8192
  - Videos: up to 4K (3840x2160)
- File size limits:
  - Images: 20MB
  - Videos: 100MB
  - Text: 10MB

### Timeout Configuration
- `request`: 30-600 seconds
- `retry_count`: 0-5 attempts
- `notify_on`: Array of event types

## Implementation Details

### Configuration Loading
```python
def load_config(path: str) -> Dict:
    """Load and validate YAML configuration file"""
    with open(path) as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config
```

### Validation Pipeline
1. Schema validation
2. Model capability checking
3. File validation
4. Timeout configuration validation

### Error Handling
- Configuration errors include:
  - Schema validation failures
  - Unsupported model types
  - Invalid file types/sizes
  - Invalid timeout settings

### Model Capability Detection
- Vision support for image/video
- Text processing capabilities
- Token limits
- API restrictions

## Usage Example

```python
from ai_battle import ConversationManager

# Initialize with config
manager = ConversationManager.from_config("discussion_config.yaml")

# Run discussion
result = await manager.run_discussion()
```

## Integration Points

### BaseClient Extensions
- File content handling
- Vision capabilities
- Timeout management

### ConversationManager Updates
- Configuration parsing
- File context management
- Enhanced error handling

## Status

Phase 1 Implementation Progress:
- [x] Configuration schema defined
- [x] Validation rules documented
- [ ] YAML parser implementation
- [ ] Model capability detection
- [ ] File handling setup
- [ ] Timeout management
- [ ] Integration with existing system

Next Steps:
1. Implement YAML configuration parser
2. Add validation functions
3. Extend BaseClient for file handling
4. Update ConversationManager