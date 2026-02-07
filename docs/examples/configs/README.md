# Configuration Examples

This directory contains example configurations for different AI discussion scenarios.

## Available Configurations

### 1. Code Review (`code_review.yaml`)
- Mode: ai-ai
- Models: claude-3-sonnet (human) + o3-mini (assistant)
- Focus: Code quality, performance, security, architecture
- Input: Python source code file
- Features:
  * Collaborative code analysis
  * Best practices review
  * Performance optimization
  * Security assessment

### 2. Medical Analysis (`medical_analysis.yaml`)
- Mode: human-ai
- Models: o1 (human) + gemini-2-pro (assistant)
- Focus: Medical image analysis
- Input: Brain scan image
- Features:
  * High-resolution image analysis
  * Clinical pattern recognition
  * Diagnostic collaboration
  * Extended conversation turns

### 3. Architecture Review (`architecture_review.yaml`)
- Mode: ai-ai
- Models: claude-3-sonnet (human) + o3-mini (assistant)
- Focus: System architecture evaluation
- Input: Markdown documentation
- Features:
  * Scalability analysis
  * Component interaction review
  * Security architecture assessment
  * Implementation feasibility

## Configuration Structure

Each configuration includes:

```yaml
discussion:
  mode: "ai-ai" | "human-ai"  # Conversation mode
  turns: number               # Number of conversation turns
  
  models:                     # Model configurations
    [model_name]:
      type: string           # Model type/version
      role: "human" | "assistant"
      parameters:            # Model-specific parameters
        temperature: number
        max_tokens: number
        top_p: number
      instructions:          # System instructions
        template: string
        params: object
  
  input_file:                # Input file configuration
    path: string
    type: "text" | "image" | "video"
    max_resolution?: string  # For image/video
  
  timeouts:                  # Timeout settings
    request: number
    retry_count: number
    notify_on: string[]
  
  execution:                 # Execution settings
    parallel: boolean
    delay_between_turns: number
    max_conversation_tokens: number
  
  goal: string              # Discussion objective
```

## Usage

1. Copy the appropriate example configuration
2. Modify parameters as needed:
   - Adjust model selections
   - Update file paths
   - Customize instructions
   - Tune execution parameters

3. Run with the configuration:
```bash
python ai-battle.py examples/configs/your-config.yaml
```

## Best Practices

1. Model Selection
   - Match model capabilities to task requirements
   - Consider token limits and pricing
   - Use appropriate temperature settings

2. System Instructions
   - Customize templates for specific domains
   - Provide clear role definitions
   - Include domain-specific terminology

3. Execution Settings
   - Adjust timeouts based on model response times
   - Set appropriate retry counts
   - Configure token limits based on discussion complexity

4. Input Files
   - Verify file paths and permissions
   - Check file type compatibility
   - Consider resolution/size limits

## Extending Configurations

To create new configurations:

1. Copy the most relevant example
2. Update the mode and models
3. Customize the goal and instructions
4. Adjust parameters for your use case
5. Test with small conversations first
6. Iterate based on results

## Notes

- Token limits vary by model
- Some models have specific temperature ranges
- File type support depends on model capabilities
- Response times may vary significantly
- Consider rate limits and API quotas