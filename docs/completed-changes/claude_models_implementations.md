# Claude 3.7 Extended Thinking Implementation Notes

## Overview

This document provides implementation notes for adding the extended thinking capability to the ClaudeClient in the `model_clients.py` file.

## Extended Thinking Parameters

Claude 3.7 supports two additional parameters for enhanced reasoning:

1. **thinking**: Boolean parameter that enables extended thinking mode
2. **budget_tokens**: Integer parameter that controls the maximum tokens Claude can use for internal reasoning

These parameters allow Claude to perform more thorough analysis internally before generating a response, which can be particularly useful for complex problems.

## Implementation Steps

Here's how to update the ClaudeClient to support extended thinking:

1. **Update the ClaudeClient.__init__ method**:
   ```python
   # Add to the existing initialization
   self.extended_thinking = False
   self.budget_tokens = None
   ```

2. **Update generate_response method**:
   ```python
   # Add to request_params in generate_response method
   if self.extended_thinking and self.capabilities.get("advanced_reasoning", False):
       request_params["thinking"] = True
       
       if self.budget_tokens is not None:
           # budget_tokens must be less than max_tokens
           if self.budget_tokens < request_params.get("max_tokens", 1024):
               request_params["budget_tokens"] = self.budget_tokens
           else:
               logger.warning(f"budget_tokens ({self.budget_tokens}) must be less than max_tokens ({request_params.get('max_tokens', 1024)}). Using max_tokens - 100.")
               request_params["budget_tokens"] = request_params.get("max_tokens", 1024) - 100
   ```

3. **Add setter methods for extended thinking**:
   ```python
   def set_extended_thinking(self, enabled: bool, budget_tokens: Optional[int] = None):
       """
       Enable or disable extended thinking mode.
       
       Args:
           enabled (bool): Whether to enable extended thinking.
           budget_tokens (Optional[int]): Maximum tokens for internal reasoning.
               Must be less than max_tokens.
       """
       self.extended_thinking = enabled
       self.budget_tokens = budget_tokens
   ```

4. **Update API client library checks**:
   ```python
   # Add to API parameter handling (alongside reasoning parameter)
   try:
       # Existing reasoning parameter handling
       if self.capabilities.get("advanced_reasoning", False):
           request_params["reasoning"] = self.reasoning_level
       
       # Add extended thinking parameters
       if self.extended_thinking and self.capabilities.get("advanced_reasoning", False):
           request_params["thinking"] = True
           if self.budget_tokens is not None:
               request_params["budget_tokens"] = self.budget_tokens
   except Exception as e:
       logger.warning(f"Client library doesn't support all reasoning parameters. Will try with basic parameters.")
   ```

## Usage Pattern

Example usage with the ClaudeClient:

```python
client = ClaudeClient(
    role="user",
    api_key=api_key,
    mode="ai-ai",
    domain="mathematics",
    model="claude-3-7-sonnet-20250219"
)

# Enable extended thinking
client.set_extended_thinking(True, budget_tokens=8000)

# Set reasoning level
client.reasoning_level = "medium"

response = client.generate_response(
    prompt="Solve this complex mathematical problem...",
    system_instruction="You are a mathematics expert.",
    model_config=ModelConfig(temperature=0.7, max_tokens=2048)
)
```

## Integration with ClaudeReasoningConfig

```python
config = get_reasoning_config(template_name="deep_analysis")
client = ClaudeClient(...)

# Apply all reasoning config parameters
client.reasoning_level = config.level
client.set_extended_thinking(config.extended_thinking, config.budget_tokens)

# Generate response with customized system instructions
response = client.generate_response(
    prompt=prompt,
    system_instruction=system_instruction + "\n" + config.to_system_instruction(),
    model_config=model_config
)
```

## Important Considerations

1. **budget_tokens must be less than max_tokens**: Always ensure budget_tokens is less than the max_tokens specified in the request.

2. **API compatibility**: Check for API client library compatibility with these newer parameters and provide appropriate fallbacks.

3. **Version detection**: These parameters are only available in Claude 3.7 and newer models. Ensure proper detection of model capabilities.

4. **Default values**: Consider reasonable defaults for budget_tokens based on the complexity of the task.