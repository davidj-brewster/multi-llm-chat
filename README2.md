```markdown
# AI Battle - Multi-Model Conversational Framework

A Python framework for orchestrating autonomous and semi-supervised multi-model conversations between multiple AI models (Claude, Gemini, OpenAI/OpenRouter, MLX, Ollama/LLama.cpp) where one model acts as a "human" prompt engineer whilst others respond as AI assistants. A third assistant
then fact-checks and summarises the conversation content and objectively
assess the contribution of each participant.

## Features

- **Multi-Model API Support**
  - Claude (Anthropic)
  - Gemini (Google)
  - OpenAI (GPT/o1 models)
  - MLX (Local inference on Apple Silicon)
  - Ollama integration
  - llama.cpp integration

- **Role Management**
  - Models are assigned either "human" or "AI" roles
  - Dynamic conversation flow control and optimised setups for large context LLMs and unfiltered local models
  - Coherent and on-point AI-AI conversations through upwards of 20 turns
  - Code-focused autonomous "pair-programmers mode" 

- **Advanced Prompting**
  - Sophisticated meta-prompt engineering patterns give the "Human" AI a life-like command of the conversation without robotic or mechanical communication patterns
  - Dynamic strategy adaptation based on subject matter and "AI" responses
  - Context-aware responses building collaboration and clarifying uncertain points
  - Thinking tag support for reasoning visualization

- **Output Management**
  - Formatted HTML conversation exports
  - Conversation history tracking
  - Support for code or goal-focused discussions or theoretical deep dives and discussions
  - Plays nicely with anything from tiny quantized local models all the way up to o1

- **Human Moderator Controls**
  - Inject messages (via streamlit UI)
  - Provide initial prompt
  - Real human moderator intervention support (WIP)

  # Moderator intervention
  `python
  await manager.moderator_intervention(
      "Let's focus more on practical applications"
  )

### Role Management

- **Dynamic Role Assignment**
  - Models can be configured as either "human" prompt engineers or AI assistants
  - Multiple AI models can participate simultaneously
  - Streamlit UI for more direct human turn-by-turn guidance and "moderator" interaction

## Advanced Features

- Temperature and parameter control based on model type
- System instruction management
- Conversation history tracking, logging and error handling
- Real-time human moderation
- Thinking tag visualization

 - **Code Focus Mode**
  - Code block extraction
  - Iterative improvement tracking through multiple rounds

### Advanced Prompting

- **Strategy Patterns**
  - Systematic analysis prompts
  - Multi-perspective examination
  - Socratic questioning
  - Adversarial challenges
  - abliterated local models can be specifically orchestrated as unfiltered agents

- **Dynamic Adaptation**
  - Context-aware prompt modification
  - Response quality monitoring
  - Strategy switching based on AI responses and human-mimicing behaviours
  - Thinking tag visualization
 




## Requirements

`python
pip install -r 

requirements.txt

Dependencies:
- google.genai (new API)
- anthropic
- openai
- ollama / llama.cpp
- mlx-lm
- streamlit
- requests
- asyncio

## Configuration

Set your API keys as environment variables:
`bash
export GEMINI_KEY="your-gemini-key"
export CLAUDE_KEY="your-claude-key"
export OPENAI_KEY="your-openai-key"
`

or omit any or all of them and they will be ignored.

## Usage

### Basic Usage

`python
from ai_battle import ConversationManager

manager = ConversationManager(
    gemini_api_key="...",
    claude_api_key="...",
    openai_api_key="..."
)

# Run a conversation with Claude as "human" and Gemini as "AI"
conversation = await manager.run_conversation(
    initial_prompt="Discuss quantum computing advances",
    human_model="claude",
    ai_model="gemini",
    rounds=3
)
`

### Web Interface (WIP)

Allows human in the middle turn by turn steering of conversations as "moderator"
`bash
streamlit run app.py
`

## Architecture

- `BaseClient`: Abstract base class for model clients
- `ConversationManager`: Orchestrates multi-turn conversations
- Model-specific clients (GeminiClient, ClaudeClient, etc.)
- HTML output generator with conversation filtering and output formatting




  # Example thinking tag in response
  `python
  "<thinking>Consider edge cases in quantum error correction</thinking>"
  `

  # Example: Configure Claude as human, Gemini as AI
  `python
  manager = ConversationManager()
  await manager.run_conversation(
      human_model="claude",
      ai_model="gemini",
      initial_prompt="Let's discuss quantum computing"
  )
  `

  # Example system instructions for "human" role
  `python
  system_instruction = """
  Approach topics systematically:
  1. Initial exploration
  2. Deep dive analysis
  3. Challenge assumptions
  4. Synthesize insights
  """
  `


### Output Management

- **Conversation Export**
  - Rich HTML formatting
  - Role-based styling
  - Thinking tag highlighting
  # Export filtered conversation
  `python
  manager.save_filtered_conversation(
      filename="quantum_discussion.html",
      exclude_thinking_tags=True,
      highlight_code_blocks=True
  )

- **History Management**
  - Full conversation context retention
  - Role-specific message tracking
  - System instruction preservation

  # Access conversation history
  `python
  for msg in manager.conversation_history:
      print(f"{msg['role']}: {msg['content']}")
  `


  # Example code-focused output prompt
  `Let's develop a sophisticated machine learning framework for analyzing brain MRIs using PyTorch. I'll start with a minimal solution and discuss iterative improvements to add sophistication, robustness, and we will iterate together step by step to build something truly world class and highly sophisticated. Make assumptions along the way as needed and fill in the gaps later so we don't get stuck on tomorrow problems!
  `

### Configuration Options

`python
@dataclass
class ConversationConfig:
    # Role Management
    allow_moderator: bool = True
`

# AI Battle Examples

## Basic Conversation Setup


# Initialize conversation with Claude as human, Gemini as AI
`python
await manager.run_conversation(
    human_model="claude",
    ai_model="gemini",
    initial_prompt="Let's discuss quantum computing"
)
`

## Moderator Controls


# Inject moderator message to guide conversation
`python
await manager.moderator_intervention(
    "Let's focus more on practical applications"
)
`

## System Instructions


# Example system instructions for "human" role
`python
system_instruction = """
Approach topics systematically:
1. Initial exploration
2. Deep dive analysis
3. Challenge assumptions
4. Synthesize insights
"""
`

## Thinking Tags

`python
# Example thinking tag format in responses
"<thinking>Consider edge cases in quantum error correction</thinking>"
`
````

## Natural-sounding prompt generation
`
Okay, so we've got agents that can respond to messages, and a human can now interact with them, but the agents are still mostly waiting for things to happen. They're not really initiating any actions on their own. Can you show me how we can modify the Agent class to include a basic "proactive behavior" system, where agents can periodically perform tasks or generate messages based on their role and the current context? I'm thinking that each agent could have a proactive_task method that gets called periodically, and this method could then use the LLM interface to generate a new message or perform some other action. Also, as a side-note, what about handling multiple concurrent conversations? How can we ensure that agents don't get confused when they're involved in multiple discussions at the same time?
`

```