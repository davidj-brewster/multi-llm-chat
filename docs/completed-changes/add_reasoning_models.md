# Implementation Notes for Reasoning Capabilities

This document provides notes on how to complete the implementation of reasoning capabilities for Claude 3.7 and OpenAI O1/O3 models. It serves as a guide for implementing the changes in the `model_clients.py` file.

## ClaudeClient Changes

1. **Add reasoning property**:
   ```python
   # Add to ClaudeClient.__init__
   self.reasoning_level = "auto"  # Options: none, low, medium, high, auto
   ```

2. **Update _update_capabilities method**:
   ```python
   # Add to ClaudeClient._update_capabilities
   # Claude 3.7 and newer models support advanced reasoning
   if "claude-3-7" in self.model.lower():
       self.capabilities["advanced_reasoning"] = True
       # Keep reasoning at auto by default for 3.7 models
       if not hasattr(self, "reasoning_level") or self.reasoning_level is None:
           self.reasoning_level = "auto"
           logger.debug(f"Set default reasoning level to 'auto' for {self.model}")
   else:
       # For any non-3.7 models, make sure we don't try to use reasoning
       self.capabilities["advanced_reasoning"] = False
       self.reasoning_level = None
       logger.debug(f"Disabled reasoning capabilities for {self.model} (only available for Claude 3.7)")
   ```

3. **Update generate_response method**:
   ```python
   # Add to ClaudeClient.generate_response before creating the API request
   # Add reasoning parameter if model supports it
   request_params = {
       "model": self.model,
       "max_tokens": max_tokens,
       "messages": messages,
       "system": system_instruction,
       "temperature": temperature,
   }
   
   # Add reasoning level parameter for Claude 3.7
   if self.capabilities.get("advanced_reasoning", False):
       try:
           request_params["reasoning"] = self.reasoning_level
           logger.debug(f"Adding reasoning={self.reasoning_level} to request for {self.model}")
       except Exception as e:
           logger.warning(f"Client library doesn't support reasoning parameter. Removed '{self.reasoning_level}' and retrying.")
   
   # Handle seed parameter specially - some versions of the client don't support it
   if "seed" in request_params:
       try:
           seed_value = request_params.pop("seed")
           # Create the completion request
           response = self.client.messages.create(**request_params)
       except Exception as e:
           logger.error(f"Error generating Claude response: {e}")
           raise
   else:
       # Create the completion request
       response = self.client.messages.create(**request_params)
   ```

## OpenAIClient Changes

1. **Add reasoning property**:
   ```python
   # Add to OpenAIClient.__init__
   self.reasoning_level = "auto"  # Default level
   self.reasoning_compatible_models = ["o1", "o1-preview", "o3"]  # Models that support reasoning_effort
   ```

2. **Update generate_response method**:
   ```python
   # Add to OpenAIClient.generate_response before making API call
   if any(model_name in self.model.lower() for model_name in self.reasoning_compatible_models):
       # Maps our reasoning levels to OpenAI's reasoning_effort parameter
       reasoning_mapping = {
           "none": "low",
           "low": "low",
           "medium": "medium",
           "high": "high",
           "auto": "high"  # Default to high for auto
       }
       
       # Get the appropriate reasoning_effort based on our reasoning_level
       reasoning_effort = reasoning_mapping.get(self.reasoning_level, "high")
       logger.debug(f"Using reasoning_effort={reasoning_effort} for model {self.model} (from reasoning_level={self.reasoning_level})")
       
       # O1/O3 model with reasoning support
       response = self.client.chat.completions.create(
           model=self.model,
           messages=formatted_messages,
           temperature=1.0,
           max_tokens=13192,
           reasoning_effort=reasoning_effort,
           timeout=90,
           stream=False
       )
       return response.choices[0].message.content
   ```

## Model Templates

Consider implementing model templates for more maintainable configuration:

```python
# Add to the top of model_clients.py or in a dedicated module

# Claude model templates
CLAUDE_MODELS = {
    "claude-3-7": {
        "base": "claude-3-7-sonnet",
        "capabilities": ["vision", "reasoning"],
        "reasoning": "auto"
    },
    "claude-3-7-reasoning-high": {
        "base": "claude-3-7-sonnet",
        "capabilities": ["vision", "reasoning"],
        "reasoning": "high"
    },
    "claude-3-7-reasoning-medium": {
        "base": "claude-3-7-sonnet",
        "capabilities": ["vision", "reasoning"],
        "reasoning": "medium"
    },
    "claude-3-7-reasoning-low": {
        "base": "claude-3-7-sonnet",
        "capabilities": ["vision", "reasoning"],
        "reasoning": "low"
    },
    "claude-3-7-reasoning-none": {
        "base": "claude-3-7-sonnet",
        "capabilities": ["vision", "reasoning"],
        "reasoning": "none"
    }
}

# OpenAI model templates
OPENAI_MODELS = {
    "o1": {
        "base": "o1",
        "capabilities": ["reasoning"],
        "reasoning": "auto"
    },
    "o1-reasoning-high": {
        "base": "o1",
        "capabilities": ["reasoning"],
        "reasoning": "high"
    },
    "o1-reasoning-medium": {
        "base": "o1",
        "capabilities": ["reasoning"],
        "reasoning": "medium"
    },
    "o1-reasoning-low": {
        "base": "o1",
        "capabilities": ["reasoning"],
        "reasoning": "low"
    },
    "o3": {
        "base": "o3",
        "capabilities": ["reasoning"],
        "reasoning": "auto"
    },
    "o3-reasoning-high": {
        "base": "o3",
        "capabilities": ["reasoning"],
        "reasoning": "high"
    },
    "o3-reasoning-medium": {
        "base": "o3",
        "capabilities": ["reasoning"],
        "reasoning": "medium"
    },
    "o3-reasoning-low": {
        "base": "o3",
        "capabilities": ["reasoning"],
        "reasoning": "low"
    }
}
```

## Configuration Updates

1. Add all new model variants to the `SUPPORTED_MODELS` lists in both `configuration.py` and `configdataclasses.py`
2. Add reasoning capability detection to the `detect_model_capabilities` function
3. Update example YAML configurations to demonstrate the use of reasoning-enabled models

## Remaining Tasks

1. Implement the modifications to `model_clients.py` for both client classes
2. Add unit tests to verify the implementation (use examples/reasoning_test.py as a base)
3. Update documentation with examples of using reasoning parameters
4. Test with real prompts and compare output quality at different reasoning levels