# Basic Usage Examples

This page provides basic examples of how to use the AI Battle framework.

## Simple Conversation

The following example demonstrates how to set up a simple conversation between two AI models:

```python
import asyncio
from ai_battle import ConversationManager

async def run_simple_conversation():
    # Initialize the conversation manager
    manager = ConversationManager(
        domain="Quantum Computing",
        mode="human-ai"
    )
    
    # Run a conversation
    conversation = await manager.run_conversation(
        initial_prompt="Explain quantum entanglement and its implications for quantum computing",
        human_model="claude",  # Model for the human role
        ai_model="gemini",     # Model for the AI role
        mode="human-ai",       # Conversation mode
        rounds=3               # Number of conversation rounds
    )
    
    # Print the conversation
    for i, message in enumerate(conversation):
        if message["role"] == "system":
            continue
        print(f"Turn {i}: {message['role'].upper()}")
        print(f"{message['content'][:200]}...\n")
    
    # Save the conversation to an HTML file
    manager.save_conversation(
        conversation=conversation,
        filename="quantum_conversation.html",
        human_model="claude",
        ai_model="gemini"
    )

# Run the example
if __name__ == "__main__":
    asyncio.run(run_simple_conversation())
```

## Using Different Models

You can use different models for the human and AI roles:

```python
# Using OpenAI's GPT-4o for the human role and Claude for the AI role
conversation = await manager.run_conversation(
    initial_prompt="Discuss the ethical implications of artificial intelligence",
    human_model="gpt-4o",
    ai_model="claude",
    mode="human-ai",
    rounds=5
)
```

## Customizing System Instructions

You can provide custom system instructions for both the human and AI roles:

```python
# Custom system instructions
human_system_instruction = """
You are an expert in artificial intelligence ethics with a background in philosophy.
Your goal is to explore the ethical implications of AI from multiple perspectives.
"""

ai_system_instruction = """
You are an AI assistant with expertise in technology ethics and policy.
Provide balanced, nuanced responses that consider different viewpoints.
"""

# Run conversation with custom system instructions
conversation = await manager.run_conversation(
    initial_prompt="Discuss the ethical implications of artificial intelligence",
    human_model="claude",
    ai_model="gemini",
    mode="human-ai",
    human_system_instruction=human_system_instruction,
    ai_system_instruction=ai_system_instruction,
    rounds=4
)
```

## AI-AI Mode

You can also run conversations in AI-AI mode, where both participants are AI models:

```python
# Initialize manager with AI-AI mode
manager = ConversationManager(
    domain="Climate Change Solutions",
    mode="ai-ai"
)

# Run AI-AI conversation
conversation = await manager.run_conversation(
    initial_prompt="Discuss innovative solutions to address climate change",
    human_model="claude",  # First AI model
    ai_model="gemini",     # Second AI model
    mode="ai-ai",          # Both participants are AI models
    rounds=6
)
```

## Error Handling

It's important to handle potential errors when working with the framework:

```python
import asyncio
from ai_battle import ConversationManager

async def run_conversation_with_error_handling():
    try:
        # Initialize manager
        manager = ConversationManager(
            domain="Space Exploration",
            mode="human-ai"
        )
        
        # Run conversation
        conversation = await manager.run_conversation(
            initial_prompt="Discuss the future of human space exploration",
            human_model="claude",
            ai_model="gemini",
            mode="human-ai",
            rounds=3
        )
        
        # Process results
        return conversation
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Run with error handling
conversation = asyncio.run(run_conversation_with_error_handling())
if conversation:
    print(f"Conversation completed with {len(conversation)} messages")
```

## Parameter Types Reference

Here's a reference for the parameter types used in the main methods:

```python
ConversationManager(
    domain: str,          # Domain/topic for the conversation
    mode: Literal["human-ai", "ai-ai"],  # Conversation mode
    human_delay: float = 4.0,  # Delay between human messages in seconds
    min_delay: float = 2,      # Minimum delay between requests in seconds
    gemini_api_key: Optional[str] = None,  # API key for Gemini models
    claude_api_key: Optional[str] = None,  # API key for Claude models
    openai_api_key: Optional[str] = None   # API key for OpenAI models
)

run_conversation(
    initial_prompt: str,  # Starting prompt for the conversation
    human_model: str,     # Model name for human role
    ai_model: str,        # Model name for AI role
    mode: str,            # Must match manager's mode
    rounds: int,          # Number of conversation rounds
    human_system_instruction: Optional[str] = None,  # Custom instructions for human role
    ai_system_instruction: Optional[str] = None      # Custom instructions for AI role
) -> List[Dict[str, str]]  # Returns conversation history
```

## Conversation Output Format

The `run_conversation()` method returns a list of message dictionaries representing the conversation history. Each message has the following structure:

```python
{
    "role": str,       # Either "system", "user", or "assistant"
    "content": str     # The message content
}
```

The conversation list typically starts with a system message that sets the context, followed by alternating user (human role) and assistant (AI role) messages. Here's an example of the structure:

```python
[
    {
        "role": "system",
        "content": "Discuss the future of human space exploration"
    },
    {
        "role": "user",
        "content": "What are the most promising developments in human space exploration?"
    },
    {
        "role": "assistant",
        "content": "There are several exciting developments in human space exploration..."
    },
    {
        "role": "user",
        "content": "How might these developments change our approach to Mars missions?"
    },
    {
        "role": "assistant",
        "content": "These developments could significantly impact Mars missions in several ways..."
    }
]
```

## Common Error Types

The framework can raise several types of errors that you should be prepared to handle:

1. **ValueError**: Raised when there are issues with configuration parameters, such as:
   - Invalid model names
   - Incompatible mode settings
   - Missing required parameters

2. **ConnectionError**: Raised when there are issues connecting to model APIs:
   - Network connectivity problems
   - API authentication failures
   - Rate limiting issues

3. **TimeoutError**: Raised when a model request times out:
   - Model response takes too long
   - Network delays

4. **MediaProcessingError**: Raised when there are issues processing media files:
   - Unsupported file types
   - File size or resolution exceeds limits
   - File not found or inaccessible

5. **MemoryError**: Raised when there are memory-related issues:
   - Conversation history too large
   - Media files too large for processing

Example of handling specific error types:

```python
try:
    conversation = await manager.run_conversation(...)
except ValueError as e:
    print(f"Configuration error: {e}")
except ConnectionError as e:
    print(f"API connection error: {e}")
except TimeoutError as e:
    print(f"Request timeout: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Next Steps

- [Configuration Guide](configuration.md): Detailed configuration options
- [Model Clients](model_clients.md): Supported models and capabilities
- [Configuration-Driven Examples](configuration_driven.md): Learn how to use configuration files
- [File-Based Discussions](file_based.md): Learn how to include files in conversations
- [Metrics Collection](metrics_collection.md): Learn how to collect and analyze conversation metrics
