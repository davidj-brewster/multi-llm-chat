# Claude 3.7 with Reasoning Levels

This document explains the implementation of Claude 3.7 models with various reasoning level settings in the AI-Battle framework.

> **Important**: The reasoning parameter is only available for Claude 3.7 models. It will not work with Claude 3.5 or earlier models. The implementation automatically detects the model version and only applies reasoning settings to Claude 3.7.

> **Compatibility Note**: The reasoning parameter is a new feature in Claude 3.7. Your version of the Anthropic Python client library may not yet support this parameter if you're seeing errors like `"Messages.create() got an unexpected keyword argument 'reasoning'"`. Our implementation is designed to gracefully handle this by automatically detecting and removing the parameter at request time for unsupported client versions.

## Overview

Claude 3.7 introduces a new `reasoning` parameter that controls how explicitly Claude shows its reasoning process in responses. This parameter allows you to customize the level of detailed thinking Claude includes in its responses.

## Available Reasoning Levels

The following reasoning levels are supported:

- **auto**: (Default) Claude automatically determines the appropriate level of reasoning based on the task
- **none**: Claude provides answers with minimal explanation of its reasoning
- **low**: Claude provides brief explanations of its reasoning
- **medium**: Claude provides moderate explanations of its reasoning
- **high**: Claude provides detailed step-by-step explanations of its reasoning

## Model Variants

The AI-Battle framework now supports the following Claude 3.7 models with reasoning:

- `claude-3-7`: Uses Claude 3.7 Sonnet with default `auto` reasoning level
- `claude-3-7-reasoning`: Uses Claude 3.7 Sonnet with `high` reasoning level
- `claude-3-7-reasoning-medium`: Uses Claude 3.7 Sonnet with `medium` reasoning level
- `claude-3-7-reasoning-low`: Uses Claude 3.7 Sonnet with `low` reasoning level
- `claude-3-7-reasoning-none`: Uses Claude 3.7 Sonnet with `none` reasoning level (explicitly disabled)

## Implementation Details

The reasoning level is set through the `reasoning_level` property of the ClaudeClient class in the `model_clients.py` file. The client automatically detects Claude 3.7 models and enables the advanced reasoning capability.

When a model with advanced reasoning capability is used, the reasoning parameter is included in the API request to Anthropic.

## Example Usage

### In Configuration Files

```yaml
discussion:
  models:
    model1:
      type: "claude-3-7-reasoning-high"
      role: "human"
      # ...other configuration...
    model2:
      type: "claude-3-7"
      role: "assistant"
      # ...other configuration...
```

### In Code

```python
from model_clients import ClaudeClient

# Create a Claude 3.7 client with custom reasoning level
client = ClaudeClient(
    role="user",
    api_key=api_key, 
    mode="ai-ai",
    domain="testing",
    model="claude-3-7-sonnet"
)

# Set the reasoning level
client.reasoning_level = "high"

# Generate a response with detailed reasoning
response = client.generate_response(
    prompt="Explain the implications of quantum computing on cryptography",
    system_instruction="You are a quantum computing expert",
)
```

## When to Use Different Reasoning Levels

- **high**: When you want detailed step-by-step explanations that show Claude's full thinking process. Good for educational content, complex problem solving, and when understanding the reasoning is as important as the answer.

- **medium**: When you want moderate explanations that give insight into Claude's thinking but don't need exhaustive details. Good for most general use cases.

- **low**: When you want brief explanations along with answers. Good for situations where some rationale is helpful but brevity is preferred.

- **none**: When you only want direct answers without explanations. Good for generating concise content or when the reasoning process isn't relevant.

- **auto**: When you want Claude to determine the appropriate level of reasoning based on the task. This is the default and works well for most scenarios.

## Testing

A test suite is available in `test_claude_reasoning.py` to verify that all Claude 3.7 reasoning variants are working correctly.

## Technical Implementation

The different reasoning levels are implemented by:

1. Adding variants in the `_get_client` method in `ai-battle.py`
2. Setting the appropriate `reasoning_level` property on the ClaudeClient
3. Updating the supported models list in configuration files
4. Adding capability detection for advanced reasoning

The reasoning parameter is passed to the Anthropic API when making a request with a Claude 3.7 model that has the advanced_reasoning capability.