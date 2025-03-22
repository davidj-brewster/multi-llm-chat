# AI Battle Framework Examples

This directory contains example configurations and scripts for the AI Battle framework.

## Configuration-Based Discussions

The AI Battle framework now supports configuration-driven setup via YAML files. This allows you to:

1. Define multiple models with specific roles and personas
2. Include file-based inputs (images, videos, text)
3. Set conversation parameters like number of turns
4. Define specific goals for the discussion

## Example Configurations

### Vision Discussion

The `configs/vision_discussion.yaml` file demonstrates a configuration for a vision-based discussion:

```yaml
discussion:
  turns: 3
  models:
    model1:
      type: "claude-3-sonnet"
      role: "human"
      persona: "Visual analysis expert..."
    model2:
      type: "gemini-pro-vision"
      role: "assistant"
      persona: "AI assistant with visual expertise..."
  input_file:
    path: "./examples/sample_image.jpg"
    type: "image"
    max_resolution: "1024x1024"
  goal: "Analyze the provided image and discuss its key elements..."
```

### Running Examples

To run the vision discussion example:

```bash
python examples/run_vision_discussion.py
```

Or specify a custom configuration file:

```bash
python examples/run_vision_discussion.py path/to/your/config.yaml
```

Or specify a file directly (overriding the one in the config):

```bash
python examples/run_vision_discussion.py path/to/your/config.yaml path/to/your/image.jpg
```

## Supported File Types

The framework supports the following file types:

1. **Images**: JPG, PNG, GIF, WebP, BMP
   - Automatically resized to a maximum of 1024x1024 pixels
   - Supported by vision-capable models (Claude-3, GPT-4o, Gemini Pro Vision)

2. **Videos**: MP4, MOV, AVI, WebM
   - Key frames are extracted for analysis
   - Currently best supported by Gemini Pro Vision

3. **Text Files**: TXT, MD, CSV, JSON, YAML, YML
   - Automatically chunked for large files

4. **Code Files**: PY, JS, HTML, CSS, Java, etc.
   - Displayed with line numbers
   - Language detection based on file extension

## Creating Your Own Configurations

To create your own configuration:

1. Create a YAML file following the structure in the examples
2. Specify at least two models with different roles
3. Optionally include an input file
4. Define a clear goal for the discussion

## Model Compatibility

Not all models support all file types. The framework will automatically check model capabilities and adapt accordingly:

- **Vision Support**: Claude-3, GPT-4o, Gemini Pro Vision, and some Ollama models (llava, bakllava, gemma3)
- **Text/Code**: All models
- **Video**: Currently best supported by Gemini Pro Vision

## Advanced Usage

For advanced usage, you can:

1. Customize model parameters in the configuration
2. Define complex personas for more specialized discussions
3. Combine multiple file types in a single discussion
4. Implement custom file processors for specialized file types