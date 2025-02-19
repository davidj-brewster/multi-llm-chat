# AI Battle - Multi-Model Conversational Framework

A sophisticated Python framework for orchestrating dynamic conversations between multiple AI models, featuring advanced context analysis, adaptive instruction systems, and multi-modal execution formats. The framework enables complex interactions between different AI models, allowing them to engage in structured discussions, collaborative problem-solving, and knowledge synthesis.

## Architecture Overview

```mermaid
graph TB
    subgraph Core Components
        CM[Conversation Manager] --> CA[Context Analyzer]
        CM --> AIM[Adaptive Instruction Manager]
        CM --> CH[Conversation History]
        
        subgraph Model Clients
            CC[Claude Client]
            GC[Gemini Client]
            OC[OpenAI Client]
            LC[Local Models Client]
        end
        
        CM --> Model Clients
        
        subgraph Conversation Flow
            CF[Flow Controller]
            RT[Rate Limiter]
            VL[Validation Layer]
        end
        
        CM --> Conversation Flow
    end
    
    subgraph Context Analysis System
        CA --> SemC[Semantic Coherence]
        CA --> TE[Topic Evolution]
        CA --> RP[Response Patterns]
        CA --> EM[Engagement Metrics]
        CA --> CL[Cognitive Load]
        CA --> KD[Knowledge Depth]
        
        subgraph Analysis Pipeline
            TF[TF-IDF Vectorization]
            CS[Cosine Similarity]
            PD[Pattern Detection]
            TA[Topic Analysis]
        end
        
        SemC --> TF
        TE --> TA
        RP --> PD
    end
    
    subgraph Adaptive Instructions
        AIM --> BSS[Bayesian Strategy Selector]
        AIM --> SC[Strategy Composer]
        AIM --> FP[Feedback Processor]
        AIM --> ML[Meta-Learning Optimizer]
        
        subgraph Strategy Components
            PS[Prior Selection]
            LM[Likelihood Models]
            PP[Posterior Probability]
        end
        
        BSS --> Strategy Components
    end
```

## Key Features

### 1. Multi-Model Support

#### Cloud Services
- **Claude (Anthropic)**
  - Models: claude-3-5-sonnet, claude-3.5-haiku
  - Specialized in structured analysis and reasoning
  - Advanced context handling capabilities
  - HTML output formatting

- **Gemini (Google)**
  - Models: gemini-2.0-flash-exp, gemini-2.0-flash-thinking-exp
  - Enhanced reasoning capabilities
  - Fact verification and grounding
  - Real-time response analysis

- **OpenAI (GPT models)**
  - Models: gpt-4o, o1, gpt-4o-mini
  - Configurable temperature and sampling
  - Stream-based response generation
  - Advanced token management

#### Local Models
- **MLX (Apple Silicon)**
  - Optimized for M1/M2 processors
  - Local inference capabilities
  - Reduced latency operations
  - Custom model support

- **Ollama Integration**
  - Models: phi4, llama3.1-8b-lexi, llama3.2-instruct
  - Local deployment options
  - Custom model fine-tuning
  - Resource-efficient execution

### 2. Execution Formats

#### Human-AI Collaboration
- **Role Simulation**
  - Sophisticated human expert emulation
  - Domain-specific knowledge integration
  - Natural conversation patterns
  - Adaptive response strategies

- **Prompting Techniques**
  - Framework-based analysis
  - Multi-step reasoning processes
  - Dynamic strategy adaptation
  - Knowledge synthesis patterns

- **Interaction Patterns**
  - Turn-based conversation flow
  - Context-aware responses
  - Engagement optimization
  - Natural language understanding

#### AI-AI Interaction
- **Peer Discussion**
  - Model-specific role assignment
  - Cross-model knowledge transfer
  - Collaborative problem-solving
  - Dynamic topic exploration

- **Technical Depth**
  - Formal reasoning patterns
  - Structured analysis frameworks
  - Deep domain exploration
  - Knowledge verification

- **Optimization Strategies**
  - Response quality metrics
  - Engagement tracking
  - Performance analysis
  - Continuous improvement

#### Goal-Based Collaboration
- **Task Management**
  - Clear objective definition
  - Progress tracking
  - Milestone achievement
  - Outcome evaluation

- **Problem-Solving Framework**
  - Structured approach definition
  - Solution exploration
  - Alternative analysis
  - Implementation planning

### 3. Context Analysis System

```mermaid
graph LR
    subgraph Context Vector Components
        SC[Semantic Coherence]
        TE[Topic Evolution]
        RP[Response Patterns]
        EM[Engagement Metrics]
        CL[Cognitive Load]
        KD[Knowledge Depth]
    end
    
    subgraph Analysis Methods
        TF[TF-IDF Analysis]
        PA[Pattern Analysis]
        TA[Topic Analysis]
        MA[Metrics Analysis]
        
        subgraph Pattern Recognition
            RP1[Reasoning Patterns]
            RP2[Uncertainty Markers]
            RP3[Technical Terms]
        end
        
        subgraph Topic Tracking
            TT1[Drift Analysis]
            TT2[Focus Metrics]
            TT3[Evolution Mapping]
        end
    end
    
    Context Vector Components --> Analysis Methods
    Analysis Methods --> Feedback[Feedback Loop]
    Feedback --> Context Vector Components
```

#### Multi-dimensional Analysis
- **Semantic Coherence**
  - TF-IDF vectorization
  - Cosine similarity metrics
  - Context window analysis
  - Response relevance scoring

- **Topic Evolution**
  - Dynamic topic tracking
  - Drift analysis
  - Focus metrics
  - Knowledge graph mapping

- **Response Patterns**
  - Pattern recognition
  - Style analysis
  - Engagement tracking
  - Quality assessment

- **Cognitive Load**
  - Complexity estimation
  - Technical density
  - Context depth
  - Understanding metrics

### 4. Adaptive Instruction System

```mermaid
graph TB
    subgraph Strategy Management
        BSS[Bayesian Strategy Selector]
        SC[Strategy Composer]
        CR[Conflict Resolver]
        
        subgraph Strategy Selection
            PS[Prior Calculation]
            LM[Likelihood Models]
            PP[Posterior Update]
        end
        
        subgraph Composition
            BC[Base Components]
            SM[Strategy Modifiers]
            CC[Coherence Check]
        end
    end
    
    subgraph Feedback Processing
        FP[Feedback Processor]
        ML[Meta-Learning]
        EM[Effectiveness Metrics]
        
        subgraph Learning Loop
            PE[Pattern Extraction]
            WU[Weight Updates]
            MO[Model Optimization]
        end
    end
    
    Context[Context Vector] --> BSS
    BSS --> Strategy Selection
    Strategy Selection --> SC
    SC --> Composition
    Composition --> Instructions[Generated Instructions]
    Instructions --> FP
    FP --> Learning Loop
    Learning Loop --> BSS
```

#### Dynamic Strategy Selection
- **Bayesian Framework**
  - Prior probability calculation
  - Likelihood model updates
  - Posterior probability estimation
  - Strategy weighting

- **Composition Rules**
  - Base component selection
  - Modifier application
  - Coherence validation
  - Conflict resolution

#### Continuous Optimization
- **Feedback Processing**
  - Response analysis
  - Effectiveness measurement
  - Pattern extraction
  - Performance metrics

- **Meta-Learning**
  - Strategy effectiveness tracking
  - Weight optimization
  - Model updates
  - Rule refinement

### 5. Output Management

#### HTML Conversation Export
- **Formatting**
  - Rich text support
  - Role-based styling
  - Timeline visualization
  - Thinking tag rendering

- **Analysis Features**
  - Quality metrics display
  - Performance analysis
  - Knowledge tracking
  - Engagement visualization

#### Analytics Framework (In Development)
- **Performance Metrics**
  - Model comparison
  - Response quality
  - Engagement levels
  - Knowledge depth

- **Visualization Tools**
  - Conversation flow
  - Topic evolution
  - Pattern recognition
  - Quality trends

## Roadmap

### 1. Analytics Framework
- **Model Performance**
  - Response quality metrics
  - Engagement tracking
  - Knowledge assessment
  - Pattern analysis

- **Conversation Analysis**
  - Flow optimization
  - Topic coherence
  - Depth measurement
  - Quality scoring

- **Comparative Analytics**
  - Model benchmarking
  - Strategy comparison
  - Performance tracking
  - Optimization metrics

### 2. Context Enhancement
- **Vector Analysis**
  - Dimension expansion
  - Feature extraction
  - Pattern recognition
  - Temporal analysis

- **Knowledge Integration**
  - Graph construction
  - Relationship mapping
  - Context preservation
  - Information synthesis

### 3. Technical Optimization
- **Caching System**
  - Context preservation
  - Response optimization
  - Memory management
  - Performance tuning

- **Message Processing**
  - Deduplication
  - Summarization
  - Context pruning
  - Quality filtering

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- anthropic
- google-generativeai
- openai
- numpy
- scikit-learn
- spacy
- ollama
- requests
- asyncio

## Usage

### Basic Configuration

```python
from ai_battle import ConversationManager

# Initialize with API keys
manager = ConversationManager(
    domain="Your topic",
    mode="human-ai",  # or "ai-ai"
    gemini_api_key="your-gemini-key",
    claude_api_key="your-claude-key",
    openai_api_key="your-openai-key"
)

# Configure model parameters
model_config = ModelConfig(
    temperature=0.7,
    max_tokens=2048,
    stop_sequences=None
)
```

### Running Conversations

```python
# Run a conversation
conversation = await manager.run_conversation(
    initial_prompt="Your prompt",
    human_model="claude",
    ai_model="gemini",
    rounds=4,
    human_system_instruction="Custom human instructions",
    ai_system_instruction="Custom AI instructions"
)

# Save the conversation
await save_conversation(
    conversation,
    "output.html",
    human_model="claude",
    ai_model="gemini",
    mode="human-ai"
)
```

### Custom Model Integration

```python
# Add custom model client
class CustomModelClient(BaseClient):
    def __init__(self, api_key: str, domain: str):
        super().__init__(api_key, domain)
        # Custom initialization
        
    async def generate_response(self,
                              prompt: str,
                              system_instruction: str = None,
                              history: List[Dict[str, str]] = None,
                              model_config: Optional[ModelConfig] = None) -> str:
        # Custom response generation logic
        pass
```

## Contributing

Contributions are welcome! Please read our contributing guidelines and code of conduct before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.