# Method: run_conversation

## Signature

```python
async def run_conversation(self, initial_prompt: str, human_model: str, ai_model: str, mode: str, human_system_instruction: str = None, ai_system_instruction: str = None, rounds: int = 1) -> List[Dict[str, str]]
```

## Description

Runs a conversation between AI models, ensuring proper role assignment and history maintenance. This method manages the entire conversation flow, including initializing the conversation history, extracting the core topic, getting client instances, and executing conversation turns.

The method supports different conversation modes, including "human-ai" and "ai-ai", and allows for customization of system instructions for both the human and AI roles.

## Parameters

- **initial_prompt** (`str`): The initial prompt to start the conversation with
- **human_model** (`str`): The model to use for the human role (e.g., "claude", "gpt-4o", "gemini")
- **ai_model** (`str`): The model to use for the AI role (e.g., "claude", "gpt-4o", "gemini")
- **mode** (`str`): The conversation mode, either "human-ai" or "ai-ai"
- **human_system_instruction** (`str`): Custom system instruction for the human role (default: None)
- **ai_system_instruction** (`str`): Custom system instruction for the AI role (default: None)
- **rounds** (`int`): Number of conversation rounds to execute (default: 1)

## Return Value

`List[Dict[str, str]]`: A list of message dictionaries representing the conversation history. Each dictionary contains "role" and "content" keys.

## Exceptions

- **ValueError**: If the required models cannot be initialized
- **Exception**: If there's an error during the conversation

## Example

```python
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
    mode="human-ai",
    rounds=3
)

# Print conversation
for message in conversation:
    print(f"{message['role'].upper()}: {message['content'][:100]}...")
```

## Notes

- The method clears the conversation history before starting a new conversation.
- The initial prompt is used as the domain for the conversation.
- The method extracts a core topic from the initial prompt for better context.
- The method automatically handles the conversation flow, alternating between human and AI turns.
- The conversation history is maintained in the `conversation_history` property of the ConversationManager instance.

## See Also

- [run_conversation_with_file](conversation_manager_run_conversation_with_file.md): Run a conversation with a file input
- [run_conversation_turn](conversation_manager_run_conversation_turn.md): Execute a single conversation turn
- [_run_conversation_with_file_data](conversation_manager__run_conversation_with_file_data.md): Internal method to run conversation with optional file data