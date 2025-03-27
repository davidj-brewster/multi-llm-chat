# Reasoning Level Control in Claude and OpenAI Models

This document explains how to use reasoning level controls in Claude 3.7 and OpenAI O1/O3 models.

## Overview

Reasoning controls allow you to adjust how explicitly the model shows its reasoning process in responses. This is useful for:

- Making models show their work in step-by-step explanations
- Controlling the balance between detailed explanations vs. concise answers
- Emphasizing logical chains of thought in complex problem-solving scenarios

## Supported Models

Currently, reasoning parameters are supported in:

### Claude Models
- claude-3-7
- claude-3-7-sonnet
- claude-3-7-reasoning
- claude-3-7-reasoning-medium
- claude-3-7-reasoning-low
- claude-3-7-reasoning-none

### OpenAI Models
- o1-reasoning-high
- o1-reasoning-medium
- o1-reasoning-low
- o3-reasoning-high
- o3-reasoning-medium
- o3-reasoning-low

## Reasoning Levels

The reasoning levels control how explicitly the model shows its work:

| Level   | Description | Best for |
|---------|-------------|---------|
| `high`  | Extremely detailed explanations with comprehensive rationale | Educational contexts, complex problem solving, showing all work |
| `medium`| Balanced explanations with key reasoning steps | General explanations, moderate detail |
| `low`   | Brief explanations with minimal reasoning shown | Quick answers, summaries |
| `none`  | Direct answers with no explicit reasoning (Claude only) | Simple factual responses |
| `auto`  | Let the model decide based on the context (Claude only) | Adaptive response style |

## Implementation Details

### Claude Implementation

For Claude 3.7 models, the reasoning level is set using the `reasoning` parameter:

```python
client = ClaudeClient(
    role="user",
    api_key=api_key,
    mode="ai-ai",
    domain="testing",
    model="claude-3-7-sonnet"
)
client.reasoning_level = "medium"  # Set to high, medium, low, none, or auto

response = client.generate_response(
    prompt="Explain how backpropagation works in neural networks.",
    system_instruction="You are a machine learning expert."
)
```

### OpenAI Implementation

For OpenAI O1/O3 models, the reasoning level is mapped to OpenAI's `reasoning_effort` parameter internally:

```python
client = OpenAIClient(
    role="user",
    api_key=api_key,
    mode="ai-ai",
    domain="testing",
    model="o1-reasoning-medium"  # Or use o1/o3 and set reasoning_level
)
client.reasoning_level = "medium"  # Set to high, medium, or low

response = client.generate_response(
    prompt="Explain how backpropagation works in neural networks.",
    system_instruction="You are a machine learning expert."
)
```

## Mapping Between Internal and API Parameters

### Claude API Mapping
- `high` → `"high"`
- `medium` → `"medium"`
- `low` → `"low"`
- `none` → `"none"`
- `auto` → `"auto"`

### OpenAI API Mapping
- `high` → `"high"`
- `medium` → `"medium"`
- `low` → `"low"`
- `none` → `"low"` (OpenAI doesn't support "none", falls back to "low")
- `auto` → `"high"` (OpenAI doesn't support "auto", falls back to "high")

## Multimodal Support and Limitations

Important notes on multimodal capabilities:

1. **Claude 3.7 Models**: Support both reasoning parameters and multimodal input (images and video).

2. **OpenAI O1/O3 Models**: Support reasoning parameters but **DO NOT** support multimodal input. If you need both reasoning and vision capabilities with OpenAI, use GPT-4o instead (which supports vision but not the reasoning_effort parameter).

## Examples

### Claude 3.7 with Low Reasoning
```python
client = ClaudeClient(
    role="user",
    api_key=api_key,
    mode="ai-ai",
    domain="testing",
    model="claude-3-7-sonnet"
)
client.reasoning_level = "low"

response = client.generate_response(
    prompt="Solve this equation: 3x + 7 = 22",
    system_instruction="You are a math tutor."
)
```

### OpenAI O1 with High Reasoning
```python
client = OpenAIClient(
    role="user",
    api_key=api_key,
    mode="ai-ai",
    domain="testing",
    model="o1-reasoning-high"
)

response = client.generate_response(
    prompt="Solve this equation: 3x + 7 = 22",
    system_instruction="You are a math tutor."
)
```

## Best Practices

1. **Match the reasoning level to the task**: Use higher reasoning for complex tasks that benefit from showing work, and lower reasoning for straightforward questions.

2. **Consider user needs**: If users need to understand the "why" behind an answer, use medium or high reasoning.

3. **Be consistent**: Try to maintain a consistent reasoning level throughout a conversation for a coherent experience.

4. **Test different levels**: Experiment with different reasoning levels to find what works best for your specific application.

5. **Check model compatibility**: Ensure your model supports reasoning parameters before using this feature.