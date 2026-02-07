# AI Battle - Detailed Overview

This document contains the comprehensive technical details, theoretical foundations, and performance insights for the AI Battle framework.

## Table of Contents

- [Theoretical Foundation](#theoretical-foundation)
- [Context Vector System](#context-vector-system)
- [Workflow Architecture](#workflow-architecture)
- [Performance Insights](#performance-insights)
- [Advanced Usage Examples](#advanced-usage-examples)
- [Video Analysis Capabilities](#video-analysis-capabilities)

## Theoretical Foundation

AI Battle simulates an advanced human interlocutor within AI-to-AI conversations, keeping the LMs goal-oriented, engaged, coherent and at a significantly increased cognitive and conversational level. It is not a role assignment, or even a static meta-prompt, but a meta-cognitive structuring mechanism that ensures the "Human" AI engages as a persistent, adaptive, and critically inquisitive entity—effectively simulating a skilled researcher, debater, or domain expert without any further extension to the "system instructions".

This meta-instruction goes far beyond standard prompting paradigms, incorporating elements that explicitly shape AI conversation structure, thought progression, and reasoning dynamics.

### AI Reasoning Should Be Benchmarked on Adaptability in Dialogue-driven Reasoning

Critically this framework has universal benefits from the tiniest 1B parameter models all the way to the largest commercial offerings - in fact, evening the playing field and bringing some surprisingly tiny LLMs up to a high level of conversational coherence.

## Context Vector System

The framework employs a multi-dimensional real-time conversation state management system called a "Context Vector" that actively tracks and analyzes conversations turn-by-turn.

### Semantic Coherence

**Why**: Measures how well consecutive messages relate to each other, indicating topic focus and logical flow. A low score suggests the conversation might be drifting, becoming disjointed, or losing focus.

**How**: Calculates the TF-IDF (Term Frequency-Inverse Document Frequency) vectors for the content of the last few messages. Then, it computes the mean cosine similarity between adjacent message vectors. A higher similarity indicates better coherence. The result is normalized.

**Impact**: A low coherence score might trigger the selection of the structured template to bring focus back.

### Topic Evolution

**Why**: Tracks the main subjects being discussed and their relative prominence. Helps understand if the conversation is staying on the intended domain or shifting significantly.

**How**: Uses spaCy's noun chunking (if spaCy model is available and loaded) to identify key topics (nouns/noun phrases) in recent messages. If spaCy is unavailable, it falls back to simple word frequency analysis (counting non-numeric words longer than 3 characters). Counts are normalized into frequencies.

**Impact**: While not directly used for template selection in the current logic, this provides valuable debugging information and could be used for more advanced topic steering in the future. It also plays a role in detecting the GOAL: keyword if it appears within the message content.

### Response Patterns

**Why**: Identifies the prevalence of different interaction styles (e.g., asking questions, challenging points, agreeing). This helps characterize the conversational dynamic.

**How**: Uses simple keyword and punctuation counting across the history (e.g., counting "?", "however", "but", "agree", "yes"). Counts are normalized by the total number of messages.

**Impact**: Can inform fine-grained adjustments in the customization phase, although not heavily used for major strategy shifts currently.

### Engagement Metrics

**Why**: Assesses the quality and balance of participation.

**How**: Calculates the average response length (in words) across the history. It also computes the turn-taking balance (ratio of user/human turns to assistant/AI turns).

**Impact**: A low turn_taking_balance (indicating one participant is dominating) might trigger a guideline like "Ask more follow-up questions" during customization.

### Cognitive Load

**Why**: Estimates the complexity of the current discussion. A very high load might indicate the conversation is becoming too dense or difficult to follow, potentially requiring simplification or synthesis.

**How**: Combines several factors from recent messages: average sentence length, vocabulary complexity (ratio of unique words to total words), and the frequency of specific technical keywords (e.g., "algorithm", "framework"). It uses spaCy for more accurate sentence and token analysis if available.

**Impact**: A high cognitive load score (> 0.8) can trigger the selection of the synthesis template, aiming to consolidate information.

### Knowledge Depth

**Why**: Gauges the level of detail, specificity, and domain understanding demonstrated in the conversation. High depth suggests a sophisticated discussion, potentially suitable for more critical analysis.

**How**: Combines factors from recent messages: density of technical terms (identified via spaCy POS tags like NOUN/PROPN, or fallback to capitalized words/keyword lists), frequency of explanation patterns (e.g., "because", "means that"), references to abstract concepts (e.g., "theory", "principle"), and use of interconnection markers (e.g., "related to", "depends on").

**Impact**: High knowledge depth (> 0.8) can trigger the selection of the critical template to encourage deeper scrutiny.

### Reasoning Patterns

**Why**: Detects the types of logical reasoning being employed (or keywords associated with them). This can help understand the analytical style of the conversation and guide instructions towards desired reasoning approaches.

**How**: Uses regex matching to count keywords associated with different reasoning types (deductive: "therefore", inductive: "generally", abductive: "most likely", analogical: "similar to", causal: "because"). In ai-ai mode, it also counts patterns related to formal logic, systematic approaches, and technical precision. Counts are normalized.

**Impact**: Specific reasoning pattern scores (e.g., low deductive or low formal_logic in ai-ai mode) can trigger corresponding guidelines during instruction customization (e.g., "Encourage logical reasoning", "Use more formal logical structures").

### Uncertainty Markers

**Why**: Assesses the expressed confidence or doubt in the conversation. High uncertainty might indicate a need for clarification or grounding.

**How**: Uses regex matching to count keywords indicating confidence ("definitely", "clearly"), uncertainty ("maybe", "could", "unsure"), qualification ("however", "possibly"), and Socratic questioning patterns.

**Impact**: High uncertainty (> 0.6) can trigger a guideline like "Request specific clarification on unclear points" during customization.

## Workflow Architecture

Basic workflow of the ai-battle.py script:

1. **User Input/Streamlit UI**: Give the conversation a "seed" topic or goal either through command-line prompts, configuration file (which can include references to multiple images, videos, code files) or the Streamlit UI which also supports multimodal input.

2. **ConversationManager**: The central component that orchestrates the conversation flow.

3. **HumanClient**: One of the AI clients, configured to act as the "human" participant.

4. **AIClient**: Another AI client, acting as the "AI" participant in the conversation.

5. **Multi-dimensional Analysis**: Turn-by-turn analysis to understand and optimize conversations, maintain coherence and subtly adjust system prompts/AI actor "personas"

6. **Adjudication**: Feedback "post-mortem" by a third AI acting as "arbiter" after the debate/discussion is completed, including Google Search powered scrutiny of claims made by either AI.

```mermaid
graph TD
    User[User Input] --> CM[Conversation Manager]
    CM --> CA[Context Analysis]
    CM --> AI[Adaptive Instructions]
    CM --> Models[Model Orchestration]

    subgraph Analysis
        CA --> Context[Context Vector]
        CA --> Metrics[Performance Metrics]
    end

    subgraph Learning
        AI --> Strategy[Strategy Selection]
        AI --> Feedback[Feedback Loop]
    end

    subgraph Execution
        Models --> Cloud[Cloud Models]
        Models --> Local[Local Models]
    end

    subgraph "New Features"
        Config[Configuration System] --> CM
        Files[File Processing] --> CM
    end
```

## Performance Insights

### Quantitative Improvements of Human-Human Mode over Human-AI

Performance analysis (via Claude 3.5 using Anthropic API Console Dashboard) of the adaptive instruction system measured improvements in conversation quality:

- **Conversation Depth**: With two LMs collaborating in "human" personas, it measured a 45% improvement in critical conversation success and relevance metrics
  - Achieved through dynamic turn-by-turn template selection and instruction modification
  - That's compared to a single highly-effectively prompted "Human" LM, where the improvement is already significant
- **Topic Coherence**: 50% enhancement via real-time coherence assessment
- **Information Density**: 40% optimization through balanced content delivery
- **Engagement Quality**: 35% increase in sustained interaction quality

### Bayesian Strategy Selection Framework

The framework is rated highly against the following criteria:
- Optimizing response patterns based on prior effectiveness
- Adapting to conversation state changes in real-time
- Resolving competing conversational priorities
- Maintaining coherence while exploring new directions

## Example Case Studies

### Gemma3 4B (as Human) Reviews an MRI Video with Gemma3 27B

This dialogue is particularly entertaining because the "Human" prompt was based on responses from ChatGPT, Claude.ai and Gemini about conversational style. This results in a very task-oriented, slightly sardonic, irritable and conversely highly engaged "Human" who completely dominated the MRI review with the "AI", despite Gemma 3 27B being clearly the more capable model from a technical point of view.

On a technical level the meta-prompting of both AIs clearly advances this conversation extremely significantly, well beyond what would be possible with such small self-hosted models.

### Phi 4 Local LLM Dominates Claude Haiku 3.5

Given the topic of why AI-based radiology hasn't been more widely adopted:

- **"Human"**: Phi 4, a 14B open source model from Microsoft, running at Q4 quantization via `ollama`
- **"AI"**: Claude Haiku 3.5

Objectively, and subjectively, the human decisively dominated and guided the conversation into ever deepening and complex aspects on that topic.

Evaluation by a third "arbiter" LLM (Gemini Pro 2 with Google Search grounding, to validate all factual claims):

- **Prompt Effectiveness (Human)**: Rated highly due to the clear, focused, and progressively complex questions that drove the conversation
- **Personality**: The human prompter showed a moderate level of personality through its use of "thinking" and "side-note" asides, while the AI's personality was more limited
- **Curiosity**: The human prompter demonstrated more curiosity by exploring new angles and asking "what if" questions, while the AI was more reactive
- **Intuition**: The human prompter showed a slightly higher level of intuition by anticipating potential challenges and shifting the focus of the conversation
- **Reasoning**: Both the human prompter and the AI model demonstrated strong reasoning skills in their questions and responses

### Gemini Flash vs ChatGPT 4o: German Reunification

In a striking example, Gemini 2.0 Flash convinced GPT-4o to completely reverse its "positive" stance on East/West German reunification, by introducing opportunity cost analysis, economic and political repercussions, and alternative paths not taken.

This demonstrates:
- The power of structured prompting in influencing AI-generated perspectives
- Implications for how LLMs can be used to challenge or strengthen reasoning of other models
- The effectiveness of adaptive conversation strategies in achieving argumentative goals

## Advanced Usage Examples

### File-Based Discussions with Video

```python
from ai_battle import ConversationManager
from configdataclasses import FileConfig

# Initialize manager
manager = ConversationManager(
    domain="Medical Analysis",
    mode="ai-ai"
)

# Run video-based conversation
conversation = await manager.run_conversation_with_file(
    initial_prompt="Analyze this MRI scan video and describe what you see",
    human_model="gemini-2.0-flash-exp",
    ai_model="ollama-llava",
    file_config=FileConfig(path="./mri_scan.mp4", type="video")
)
```

### Configuration-Driven Setup

```yaml
discussion:
  input_file:
    path: "./medical_scan.mp4"
    type: "video"
    max_resolution: "1280x1280"
  established_facts:
    video_processing_information:
      - "The ENTIRE VIDEO CONTENT is sent to models, not just individual frames"
      - "Videos are processed in chunks to handle size limitations"
      - "The processed video is resized to a maximum dimension of 1280 pixels"
      - "The video is processed at a reduced framerate (2 fps) for efficiency"
  turns: 5
  models:
    model1:
      type: "gemini-2.0-flash-exp"
      role: "human"
      persona: "Medical imaging specialist with expertise in radiology"
    model2:
      type: "claude-3-sonnet"
      role: "assistant"
      persona: "AI assistant with medical knowledge"
  goal: "Provide a comprehensive analysis of the medical imaging video"
```

### Custom Model Integration

```python
from ai_battle import BaseClient
from typing import Dict, List, Optional

class CustomModelClient(BaseClient):
    def __init__(self, api_key: str, domain: str):
        super().__init__(api_key, domain)

    async def generate_response(self,
                              prompt: str,
                              system_instruction: str = None,
                              history: List[Dict[str, str]] = None,
                              model_config: Optional[ModelConfig] = None) -> str:
        # Custom implementation
        pass
```

### Conversation Analysis

```python
from ai_battle import ContextAnalyzer

analyzer = ContextAnalyzer()
context = analyzer.analyze(conversation_history)

print(f"Semantic Coherence: {context.semantic_coherence:.2f}")
print(f"Topic Evolution: {dict(context.topic_evolution)}")
print(f"Knowledge Depth: {context.knowledge_depth:.2f}")
print(f"Cognitive Load: {context.cognitive_load:.2f}")
print(f"Reasoning Patterns: {dict(context.reasoning_patterns)}")
```

## Video Analysis Capabilities

The framework supports comprehensive video analysis with both cloud-based and local models:

### Features

- **Full Video Processing**: Send entire videos to models, not just individual frames
- **Format Conversion**: Automatic conversion from .mov to .mp4 format for better compatibility
- **Chunked Processing**: Videos are processed in manageable chunks to handle API size limitations
- **Resolution Optimization**: Videos are resized to a maximum dimension while maintaining aspect ratio
- **Framerate Adjustment**: Videos are processed at an optimized framerate for efficiency
- **Sequential Analysis**: Each chunk is analyzed sequentially, with insights combined from all chunks
- **Conversation Context**: Models maintain context about the video between conversation turns

### Supported Models for Video

- **Gemini Models**: Full video support with gemini-2.0-flash-exp and gemini-2.0-pro models
- **Local Ollama Models**: Video support for vision-capable models like llava and gemma3

## Conversation Modes

### 1. Human-AI Collaboration

- One model acts as a human expert
- Natural conversation flow
- Sophisticated prompting techniques

### 2. AI-AI Interaction

- Peer-level technical discussions
- Cross-model knowledge synthesis
- Formal reasoning patterns

### 3. Goal-Based Collaboration

- Task-oriented conversations
- Progress tracking
- Outcome evaluation

### 4. File-Based Discussions

- Image analysis and interpretation
- Video analysis with full content processing
- Text file analysis
- Code review and explanation

## Advanced Features

### Code Focus Mode

- Code block extraction
- Iterative improvement tracking through multiple rounds
- Pair-programming dynamics

### Advanced Prompting Strategies

- **Systematic Analysis**: Structured approach to problem decomposition
- **Multi-Perspective Examination**: Exploring topics from different angles
- **Socratic Questioning**: Guided discovery through questions
- **Adversarial Challenges**: Testing ideas through opposition
- **Abliterated Local Models**: Specifically orchestrated as unfiltered agents

### Dynamic Adaptation

- Context-aware prompt modification
- Response quality monitoring
- Strategy switching based on AI responses and human-mimicking behaviours
- Thinking tag visualization for reasoning transparency

## Context Analysis Visualization

```mermaid
graph TD
    subgraph "Context Vector"
        SC[Semantic Coherence]
        TE[Topic Evolution]
        RP[Response Patterns]
        EM[Engagement Metrics]
    end

    subgraph "Analysis"
        TF[TF-IDF Analysis]
        PA[Pattern Recognition]
        TA[Topic Tracking]
        MA[Metrics Assessment]
    end

    Context --> Analysis
    Analysis --> Feedback[Feedback Loop]
    Feedback --> Context
```

## Key Capabilities

### Role Management

- Models are assigned either "human" or "AI" roles
- Dynamic conversation flow control and optimized setups for large context LLMs and unfiltered local models
- Coherent and on-point AI-AI conversations through upwards of 20 turns
- Code-focused autonomous "pair-programmers mode"

### Advanced Prompting

- Sophisticated meta-prompt engineering patterns give the "Human" AI a life-like command of the conversation without robotic or mechanical communication patterns
- Dynamic strategy adaptation based on subject matter and "AI" responses
- Context-aware responses building collaboration and clarifying uncertain points
- Thinking tag support for reasoning visualization

### Output Management

- Formatted HTML conversation exports
- Conversation history tracking
- Support for code or goal-focused discussions or theoretical deep dives and discussions
- Plays nicely with anything from tiny quantized local models all the way up to o1

### Human Moderator Controls

- Inject messages (via streamlit UI)
- Provide initial prompt
- Real human moderator intervention support (WIP)

### Dynamic Role Assignment

- Models can be configured as either "human" prompt engineers or AI assistants
- Multiple AI models can participate simultaneously
- Streamlit UI for more direct human turn-by-turn guidance and "moderator" interaction

## Advanced Configuration

### Temperature and Parameter Control

- Based on model type/modality
- System instruction management
- Conversation history tracking, logging and error handling
- Thinking tag visualization

### Multi-Model Support Details

- **Claude Sonnet/Haiku (Anthropic)**: Multimodal with Image support, and new Messages API and Reasoning efforts/tokens
- **Gemini (Flash/Pro/Thinking)**: Multimodal with Video + Image support
- **OpenAI (GPT 4o/o1/o3/4.5/4.1/o4 models)**: Multimodal with Image support and new Response API with reasoning effort configuration support
- **Ollama** (e.g., llama3.2-vision, gemma3, phi4): Multimodal including Video via langchain or native ollama python API
- **MLX**: Local inference on Apple Silicon
- **Pico client**: Via Ollama API
- **LMStudio client**: Via OpenAI API supporting GGUF and MLX model configurations
- **Direct llama.cpp**: Or anything exposing an OpenAI endpoint, e.g., OpenRouter

## System Dependencies

### macOS

```bash
brew install cmake pkg-config
brew install spacy
```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install cmake pkg-config python3-dev
python -m spacy download en_core_web_sm
```

## Research and Development

For the theoretical underpinnings and research paper (on a single AI playing Human, not yet updated for multi-configurable AIs), see:
[Research Paper](research-paper.md)
