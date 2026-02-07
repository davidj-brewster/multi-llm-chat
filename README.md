<table>
  <tr>
    <td align="center"><img height="170" alt="Example 3" src="https://github.com/user-attachments/assets/3be3373a-8c92-4aa5-85a1-b4c2afdabce8" />
    </td>
    <td>
      <img height="170" src="https://github.com/user-attachments/assets/e4d2b766-a585-40e6-b37e-ac6b7b98f3d1" />
      </td>
  </tr>
  <tr>
    <td><img height="150" alt="Example 3" src="https://github.com/user-attachments/assets/b18cdc6d-7f5c-479f-9d1f-56ed183ae9c0" alt="Example 2" />
    </td>
    <td><img height="150" alt="Example 4" src="https://github.com/user-attachments/assets/19524ff8-ec8a-47d8-aa24-4eb0066b3379" />
</td>
  </tr>
</table>

# AI Battle

A framework for dynamic multi-AI conversations featuring adaptive personas, conversation management, and deep analysis. Watch AIs collaborate, debate, or battle in real-time across any topic.

## What It Does

- **AI-to-AI Conversations**: Two or more AIs discuss, debate, or collaborate on any topic
- **Adaptive Conversation Management**: Automatically maintains coherence and engagement through multi-dimensional analysis
- **Multi-Modal Support**: Works with images, videos, code, and text
- **Model Agnostic**: Works with local models (Ollama, MLX) and cloud providers (Claude, GPT, Gemini)
- **Real-time Analysis**: Context vectors track coherence, topic evolution, reasoning patterns, and more

## Quick Start

### Installation

```bash
# Install uv (recommended)
pip install uv

# Create environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Optional: Install local models
curl https://ollama.ai/install.sh | sh
ollama pull phi4:latest
```

### Basic Usage

```python
from ai_battle import ConversationManager

# Initialize manager
manager = ConversationManager(
    domain="Quantum Computing",
    mode="human-ai"
)

# Run conversation
conversation = await manager.run_conversation(
    initial_prompt="Explain quantum entanglement",
    human_model="claude",
    ai_model="gemini"
)
```

### Using Configuration Files

```yaml
# config.yaml
discussion:
  turns: 5
  models:
    model1:
      type: "claude-3-sonnet"
      role: "human"
    model2:
      type: "gemini-pro"
      role: "assistant"
  goal: "Discuss the implications of quantum computing"
```

```python
manager = ConversationManager.from_config("config.yaml")
result = await manager.run_discussion()
```

## Supported Models

- **Claude** (Sonnet/Haiku) - Anthropic API
- **GPT** (4o/o1/o3) - OpenAI API
- **Gemini** (Flash/Pro/Thinking) - Google API
- **Ollama** (llama, phi, gemma, etc.) - Local inference
- **MLX** - Apple Silicon local inference
- **LMStudio** - Local GGUF/MLX models
- Custom models via OpenAI-compatible endpoints

## Key Features

- **Context-Adaptive System**: Tracks semantic coherence, topic evolution, engagement metrics, cognitive load, knowledge depth, reasoning patterns, and uncertainty markers
- **Dynamic Role Assignment**: Models can act as "human" experts or AI assistants with adaptive personas
- **Multi-Modal Analysis**: Full video processing, image analysis, code review
- **YAML Configuration**: Easy setup for complex multi-turn discussions
- **Real-time Metrics**: Live conversation quality assessment and adaptation

## Example Use Cases

- **Code Review**: Two AIs pair-program and review code together
- **Research Discussion**: Deep dives into complex topics with adaptive questioning
- **Content Analysis**: Analyze images, videos, or documents collaboratively
- **Debate & Argumentation**: Watch AIs challenge each other's reasoning
- **Education**: Socratic dialogue for learning complex subjects

## Documentation

- [Detailed Overview & Context System](docs/detailed-overview.md) - Full technical details, context vectors, performance insights
- [Configuration Guide](docs/configuration.md) - YAML configuration system
- [Architecture Overview](docs/architectural-analysis.md) - System design and components
- [Research Paper](docs/research-paper.md) - Theoretical foundations

## Environment Setup

```bash
export GEMINI_API_KEY="your-gemini-key"
export GOOGLE_API_KEY="$GEMINI_API_KEY"
export ANTHROPIC_API_KEY="your-claude-key"
export OPENAI_API_KEY="your-openai-key"
```

## Performance Highlights

- **45%** improvement in conversation depth with dual-AI human personas
- **50%** enhancement in topic coherence
- **40%** optimization in information density
- Works with models from 1B to 100B+ parameters

## License

MIT License - see LICENSE file for details.
