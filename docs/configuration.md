# Configuration System Documentation

The AI Battle framework now supports a comprehensive configuration system that allows you to define complex discussions with multiple models and file inputs using YAML configuration files.

## Configuration File Structure

A complete configuration file has the following structure:

```yaml
discussion:
  turns: 3  # Number of back-and-forth exchanges
  models:
    model1:
      type: "claude-3-sonnet"  # Model identifier
      role: "human"            # Role in conversation
      persona: |               # Persona-based system instructions
        You are an expert with the following characteristics:
         - Specific expertise details
         - Behavioral traits
         - Knowledge areas
    model2:
      type: "gemini-pro"
      role: "assistant"
      persona: |
        You are an AI assistant with the following characteristics:
         - Expertise areas
         - Interaction style
         - Reasoning approach
   
  timeouts:
    request: 300             # Request timeout in seconds
    retry_count: 3           # Number of retries
    notify_on:
      - timeout              # Notify on timeout
      - retry                # Notify on retry
      - error                # Notify on error
  
  input_file:
    path: "./path/to/file"   # Path to input file
    type: "image"            # image, video, text, or code
    max_resolution: "1024x1024"  # For images/videos
    
  goal: |
    Detailed description of the discussion objective.
    This can be multi-line and include specific instructions.
```

## Configuration Components

### Models

The `models` section defines the participants in the conversation:

- **type**: The model identifier (e.g., "claude-3-sonnet", "gemini-pro", "gpt-4o")
- **role**: Either "human" or "assistant"
- **persona**: Detailed instructions that define the model's behavior, expertise, and interaction style

You must define at least two models, typically one with the "human" role and one with the "assistant" role.

### Timeouts

The `timeouts` section controls request handling:

- **request**: Maximum time in seconds to wait for a model response
- **retry_count**: Number of times to retry a failed request
- **notify_on**: Events that trigger notifications (timeout, retry, error)

### Input File

The `input_file` section defines a file to include in the discussion:

- **path**: Path to the file (relative to the project root)
- **type**: File type ("image", "video", "text", or "code")
- **max_resolution**: For images and videos, the maximum resolution to maintain

### Goal

The `goal` field defines the objective of the discussion. This should be a clear, detailed description of what you want the models to discuss or accomplish.

## Supported Model Types

The framework supports various model types:

### Cloud Models

- **Claude**: "claude-3-sonnet", "claude-3-opus", "claude-3-haiku", "claude-3-5-sonnet"
- **OpenAI**: "gpt-4", "gpt-4o", "gpt-4-vision"
- **Gemini**: "gemini-pro", "gemini-pro-vision", "gemini-2.0-pro"

### Local Models

- **Ollama**: Various models including vision-capable ones like "llava", "bakllava", "gemma3"
- **MLX**: Local models running through MLX

## Vision Capabilities

Not all models support vision. The framework automatically detects vision capabilities based on the model type:

- **Vision-capable cloud models**: Claude-3 series, GPT-4o, GPT-4-vision, Gemini Pro Vision
- **Vision-capable Ollama models**: llava, bakllava, moondream, llava-phi3, gemma3

When using a file input with a model that doesn't support vision, the framework will attempt to convert the file content to a text representation.

## File Type Support

The framework supports various file types:

### Images
- Extensions: .jpg, .jpeg, .png, .gif, .webp
- Automatically resized to a maximum resolution (default: 1024x1024)
- Converted to base64 for API transmission

### Videos
- Extensions: .mp4, .mov, .avi, .webm
- Key frames extracted for analysis
- First frame used for models that only support single images

### Text Files
- Extensions: .txt, .md, .csv, .json, .yaml, .yml
- Automatically chunked for large files
- Included directly in the conversation context

### Code Files
- Extensions: .py, .js, .html, .css, .java, etc.
- Displayed with line numbers
- Language detection based on file extension

## Using the Configuration System

### Loading a Configuration

```python
from ai_battle import ConversationManager

# Initialize with config
manager = ConversationManager.from_config("path/to/config.yaml")

# Run discussion
result = await manager.run_discussion()
```

### Programmatic Configuration

You can also create configuration objects programmatically:

```python
from configdataclasses import DiscussionConfig, ModelConfig, FileConfig, TimeoutConfig

config = DiscussionConfig(
    turns=3,
    models={
        "model1": ModelConfig(type="claude-3-sonnet", role="human", persona="..."),
        "model2": ModelConfig(type="gemini-pro", role="assistant", persona="...")
    },
    input_file=FileConfig(path="./image.jpg", type="image"),
    goal="Analyze this image and discuss its key elements."
)

manager = ConversationManager(config=config)
```

## Best Practices

1. **Clear Goals**: Define specific, clear goals for the discussion
2. **Complementary Personas**: Create personas that complement each other for richer discussions
3. **Appropriate Models**: Choose models with capabilities that match your file type
4. **File Optimization**: Optimize large files before including them in discussions
5. **Testing**: Test your configuration with smaller discussions before running longer ones