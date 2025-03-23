# Configuration-Driven Examples

This page provides examples of how to use the AI Battle framework with configuration files.

## Using YAML Configuration Files

The AI Battle framework supports YAML configuration files for defining complex discussions with multiple models and file inputs.

### Basic Configuration File

Here's a basic configuration file (`basic_config.yaml`):

```yaml
discussion:
  turns: 3  # Number of back-and-forth exchanges
  models:
    expert:
      type: "claude-3-sonnet"  # Model identifier
      role: "human"            # Role in conversation
      persona: |               # Persona-based system instructions
        You are an expert in artificial intelligence with the following characteristics:
         - Deep knowledge of machine learning algorithms
         - Understanding of neural network architectures
         - Familiarity with current AI research trends
         - Ability to explain complex concepts clearly
    assistant:
      type: "gemini-pro"
      role: "assistant"
      persona: |
        You are an AI assistant with the following characteristics:
         - Helpful and informative responses
         - Clear explanations of technical concepts
         - Balanced perspective on AI capabilities
         - Acknowledgment of limitations when appropriate
   
  timeouts:
    request: 300             # Request timeout in seconds
    retry_count: 3           # Number of retries
    notify_on:
      - timeout              # Notify on timeout
      - retry                # Notify on retry
      - error                # Notify on error
    
  goal: |
    Discuss the current state of artificial general intelligence (AGI) research,
    including recent breakthroughs, major challenges, and realistic timelines
    for achieving human-level AI.
```

### Using the Configuration File

```python
import asyncio
from ai_battle import ConversationManager

async def run_from_config():
    # Initialize manager from config file
    manager = ConversationManager.from_config("basic_config.yaml")
    
    # Run discussion based on configuration
    conversation = await manager.run_discussion()
    
    # Save the conversation
    manager.save_conversation(
        conversation=conversation,
        filename="agi_discussion.html",
        human_model="expert",
        ai_model="assistant",
        mode="human-ai"
    )

# Run the example
if __name__ == "__main__":
    asyncio.run(run_from_config())
```

## Configuration with File Input

You can include file inputs in your configuration:

```yaml
discussion:
  turns: 4
  models:
    analyst:
      type: "claude-3-sonnet"
      role: "human"
      persona: |
        You are a data analyst with expertise in interpreting visual data.
        You ask insightful questions about data visualizations and help
        extract meaningful insights from them.
    assistant:
      type: "gemini-pro-vision"
      role: "assistant"
      persona: |
        You are an AI assistant with strong capabilities in image analysis.
        You provide detailed explanations of data visualizations and help
        identify patterns and trends.
  
  input_file:
    path: "./data/quarterly_results.png"   # Path to input file
    type: "image"                          # File type
    max_resolution: "1024x1024"            # Maximum resolution
    
  goal: |
    Analyze the quarterly results chart and discuss the key trends,
    notable changes, and potential implications for the business.
```

## Programmatic Configuration

You can also create configuration objects programmatically:

```python
from ai_battle import ConversationManager
from configdataclasses import DiscussionConfig, ModelConfig, FileConfig, TimeoutConfig

# Create model configurations
expert_model = ModelConfig(
    type="claude-3-sonnet",
    role="human",
    persona="You are an expert in climate science with deep knowledge of global warming trends."
)

assistant_model = ModelConfig(
    type="gemini-pro",
    role="assistant",
    persona="You are an AI assistant with expertise in environmental policy and climate data analysis."
)

# Create timeout configuration
timeouts = TimeoutConfig(
    request=300,
    retry_count=2,
    notify_on=["timeout", "error"]
)

# Create file configuration (optional)
file_config = FileConfig(
    path="./data/global_temperature.csv",
    type="text"
)

# Create discussion configuration
config = DiscussionConfig(
    turns=5,
    models={
        "expert": expert_model,
        "assistant": assistant_model
    },
    timeouts=timeouts,
    input_file=file_config,
    goal="Analyze global temperature trends and discuss implications for climate policy."
)

# Initialize manager with programmatic configuration
manager = ConversationManager(config=config)

# Run discussion
conversation = await manager.run_discussion()
```

## Advanced Configuration Options

### Multiple Models

You can configure discussions with more than two models:

```yaml
discussion:
  turns: 6
  models:
    scientist:
      type: "claude-3-opus"
      role: "human"
      persona: "You are a research scientist specializing in quantum physics..."
    engineer:
      type: "gpt-4o"
      role: "human"
      persona: "You are a quantum computing engineer with practical experience..."
    moderator:
      type: "gemini-pro"
      role: "assistant"
      persona: "You are a moderator guiding the discussion between the scientist and engineer..."
  
  goal: "Explore the practical challenges of implementing quantum algorithms..."
```

### Execution Settings

You can configure execution settings:

```yaml
discussion:
  # ... other configuration ...
  
  execution:
    parallel: false                  # Run turns sequentially
    delay_between_turns: 2.0         # Delay between turns in seconds
    max_conversation_tokens: 32768   # Maximum tokens for the entire conversation
```

## Best Practices

1. **Clear Goals**: Define specific, clear goals for the discussion
2. **Complementary Personas**: Create personas that complement each other for richer discussions
3. **Appropriate Models**: Choose models with capabilities that match your file type
4. **File Optimization**: Optimize large files before including them in discussions
5. **Testing**: Test your configuration with smaller discussions before running longer ones

## Next Steps

- [File-Based Discussions](file_based.md): Learn more about including files in conversations
- [Metrics Collection](metrics_collection.md): Learn how to collect and analyze conversation metrics
- [Basic Usage](basic_usage.md): Return to basic usage examples