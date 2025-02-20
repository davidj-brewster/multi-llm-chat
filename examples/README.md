# AI Battle Configuration Examples

This directory contains example configurations demonstrating how to use the AI Battle framework with different scenarios and file types.

## Basic Usage

1. Create a YAML configuration file (e.g., `discussion_config.yaml`)
2. Define your models, file inputs, and discussion parameters
3. Run the discussion:

```python
from ai_battle import ConversationManager

# Initialize with config
manager = ConversationManager.from_config("discussion_config.yaml")

# Run discussion
result = await manager.run_discussion()
```

## Configuration Structure

### Models
```yaml
models:
  model1:
    type: "claude-3-sonnet"    # Model identifier
    role: "human"              # Role in conversation
    instructions:
      template: "human_system_instructions"
      params:
        domain: "Your Domain"  # Template parameters
```

Supported models:
- Claude: claude-3-sonnet, claude-3-haiku
- Gemini: gemini-pro, gemini-pro-vision
- OpenAI: gpt-4-vision, gpt-4
- Local: ollama-*, mlx-*

### File Inputs

```yaml
input_file:
  path: "./path/to/file"
  type: "image"              # image, video, or text
  max_resolution: "4K"       # For image/video
```

Supported file types:
- Images (jpg, jpeg, png, gif, webp)
  * Max size: 20MB
  * Max resolution: 8192x8192
- Videos (mp4, mov, avi, webm)
  * Max size: 100MB
  * Max resolution: 4K (3840x2160)
- Text (txt, md, py, js, etc.)
  * Max size: 10MB

### Timeouts

```yaml
timeouts:
  request: 300              # Request timeout in seconds
  retry_count: 3            # Number of retries
  notify_on:               # Notification events
    - timeout
    - retry
    - error
```

## Example Configurations

1. `discussion_config.yaml` - Medical image analysis with video input
2. `text_review.yaml` - Code review discussion (see examples/text_review.yaml)
3. `image_analysis.yaml` - Image analysis with multiple models (see examples/image_analysis.yaml)

## System Instructions

The framework uses template-based system instructions defined in `docs/system_instructions.md`. You can reference these templates in your configuration:

```yaml
models:
  model1:
    instructions:
      template: "human_system_instructions"  # Template name
      params:                               # Template parameters
        domain: "Your Domain"
        role: "Expert Reviewer"
```

Available templates:
- `human_system_instructions` - Human role simulation
- `ai_assistant_instructions` - AI assistant role
- `goal_oriented_instructions` - Task-focused discussion

## Best Practices

1. **File Handling**
   - Ensure files are accessible at the specified paths
   - Use appropriate resolution settings for image/video
   - Consider file size limits when selecting inputs

2. **Model Selection**
   - Match model capabilities to task requirements
   - Use vision-capable models for image/video inputs
   - Consider local models for faster processing

3. **Timeout Configuration**
   - Adjust timeouts based on input size and complexity
   - Set appropriate retry counts for reliability
   - Enable notifications for important events

4. **System Instructions**
   - Use templates to maintain consistent behavior
   - Customize instructions through parameters
   - Keep personas focused and relevant to the task

## Error Handling

The configuration system validates:
- File existence and format
- Model compatibility
- Parameter ranges
- Template validity

Common errors and solutions:
- `FileNotFoundError`: Check file paths
- `ValueError`: Review parameter values
- `TypeError`: Ensure correct data types
- `ValidationError`: Check configuration structure