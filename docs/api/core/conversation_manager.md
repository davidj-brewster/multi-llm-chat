# Class: ConversationManager

## Overview

The ConversationManager class is the core component of the AI Battle framework. It manages conversations between AI models with memory optimization, handling the flow of messages, model interactions, and file-based discussions.

## Constructor

```python
ConversationManager(config: DiscussionConfig = None, domain: str = "General knowledge", human_delay: float = 4.0, mode: str = None, min_delay: float = 2, gemini_api_key: Optional[str] = None, claude_api_key: Optional[str] = None, openai_api_key: Optional[str] = None)
```

### Parameters

- **config** (`DiscussionConfig`): Configuration for the conversation (default: None)
- **domain** (`str`): Domain or topic of the conversation (default: "General knowledge")
- **human_delay** (`float`): Delay between human messages in seconds (default: 4.0)
- **mode** (`str`): Conversation mode, either "human-aiai" or "ai-ai" (default: None)
- **min_delay** (`float`): Minimum delay between requests in seconds (default: 2)
- **gemini_api_key** (`Optional[str]`): API key for Gemini models (default: None)
- **claude_api_key** (`Optional[str]`): API key for Claude models (default: None)
- **openai_api_key** (`Optional[str]`): API key for OpenAI models (default: None)

## Properties

- **media_handler**: Lazy-initialized ConversationMediaHandler for processing media files

## Methods

- [run_conversation](conversation_manager_run_conversation.md): Run a conversation between AI models
- [run_conversation_with_file](conversation_manager_run_conversation_with_file.md): Run a conversation with a file input
- [run_conversation_turn](conversation_manager_run_conversation_turn.md): Execute a single conversation turn
- [_get_client](conversation_manager__get_client.md): Get or create a client instance for a model
- [cleanup_unused_clients](conversation_manager_cleanup_unused_clients.md): Clean up clients that haven't been used recently
- [validate_connections](conversation_manager_validate_connections.md): Validate required model connections
- [rate_limited_request](conversation_manager_rate_limited_request.md): Apply rate limiting to requests
- [from_config](conversation_manager_from_config.md): Create a ConversationManager from a configuration file

## Usage Examples

```python
# Example 1: Basic conversation
from ai_battle import ConversationManager

# Initialize manager
manager = ConversationManager(
    domain="Quantum Computing",
    mode="human-ai"
)

# Run standard conversation
conversation = await manager.run_conversation(
    initial_prompt="Explain quantum entanglement",
    human_model="claude",
    ai_model="gemini",
    rounds=3
)

# Save conversation
manager.save_conversation(conversation, filename="quantum_conversation.html")

# Example 2: File-based conversation
from configdataclasses import FileConfig

# Run file-based conversation
conversation = await manager.run_conversation_with_file(
    initial_prompt="Analyze this image and explain what you see",
    human_model="claude-3-sonnet",
    ai_model="gemini-pro-vision",
    mode="ai-ai",
    file_config=FileConfig(path="./image.jpg", type="image")
)
```

## Related Classes

- [AdaptiveInstructionManager](adaptive_instructions.md): Manager for adaptive instructions
- [BaseClient](../model_clients/base_client.md): Base class for model clients
- [FileConfig](../configuration/config_classes.md): Configuration for file inputs
- [ConversationMediaHandler](../file_handling/media_handler.md): Handler for media files