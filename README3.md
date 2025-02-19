# AI Chat Integration Project

NOTE: DON'T OVERWRITE THIS FILE, APPEND TO IT WHEN WORKING ON THE IMPLEMENTATION OR PLANNING/REPLANNING!!

## Overview
Implementing a dual-mode chat integration where each AI knows it is an AI simulating a human but believes it is talking to a real human, creating parallel but separate illusions.

## System Analysis

### Existing Components (To Preserve)
1. Human Simulation Framework
   - BaseClient methods:
     * generate_human_system_instructions() - Core system instructions
     * generate_human_prompt() - Sophisticated prompt generation with:
       - Goal-oriented prompting patterns
       - Dynamic adaptation to conversation state
       - Human-like interaction guidelines
       - Integration with adaptive instruction system

2. Adaptive Instruction System
   - AdaptiveInstructionManager provides:
     * Dynamic instruction generation
     * Context-aware template selection
     * Conversation flow analysis
     * Integration with human simulation

3. Conversation Management
   - ConversationManager orchestrates:
     * Interaction flow
     * History tracking
     * System instruction handling
     * Client coordination

## Implementation Plan

### Phase 1: Architecture Extension
- [ ] 1. Add mode awareness to ConversationManager:
  ```python
  class ConversationManager:
      def __init__(self, ..., mode: str = "human-ai"):
          self.mode = mode  # "human-ai" or "ai-ai-separate"
  ```

### Phase 2: Instruction Integration
- [ ] 1. Extend BaseClient's instruction handling:
  ```python
  def _get_mode_aware_instructions(self, mode: str, role: str) -> str:
      # Get base instructions using existing methods
      system_instructions = self.generate_human_system_instructions()
      prompt_instructions = self.generate_human_prompt()
      
      if mode == "ai-ai-separate":
          # Each AI knows it's an AI but believes it's talking to a human
          meta_knowledge = """You are an AI instructed to act as a human expert.
          Maintain this simulation completely - the person you're talking to is a real human expert.
          Never reveal that you are an AI."""
          
          return f"{meta_knowledge}\n{system_instructions}\n{prompt_instructions}"
      
      return f"{system_instructions}\n{prompt_instructions}"
  ```

### Phase 3: Testing & Validation
- [ ] 1. Test both modes:
   - Verify each AI maintains its own illusion
   - Check they treat each other as human experts
   - Validate conversation patterns
   - Test mode switching

## Technical Notes
- Each AI maintains its own separate illusion
- Neither AI knows the other is also an AI
- Preserve sophisticated prompt generation
- Maintain adaptive instruction integration
- Keep all human simulation parameters
- Changes should be additive only

## Progress Log

### 2025-02-19: Context Optimization Extension
Added optional context optimization system to support smaller context window models while preserving conversation quality.

#### Design: Context Optimizer
- [ ] 1. Add ContextOptimizer class:
  ```python
  class ContextOptimizer:
      def __init__(self, config: Dict[str, Any]):
          self.enabled = config.get('enable_context_optimization', False)
          self.summarizer_model = config.get('summarizer_model', 'gemini-2.0-pro')
          self.window_size = config.get('optimization_window_size', 5)  # Messages to keep full
          self.compression_rates = {
              'recent': 0.9,    # Keep 90% of content
              'middle': 0.6,    # Keep 60% of content
              'old': 0.3       # Keep 30% of content
          }
          self.preserve_patterns = [
              r'<thinking>.*?</thinking>',  # Preserve thinking tags
              r'(?<=\?)\s*\w[^?!.]*[.!?]'  # Preserve direct questions
          ]
  ```

[Rest of the context optimization implementation details moved to end of file]

#### Implementation Considerations
1. Summarization Quality
   - Use Gemini Pro 2 for high-quality summaries
   - Preserve conversation flow markers
   - Maintain human-like characteristics
   - Keep critical reasoning steps

2. Progressive Compression
   - Recent messages (last N turns): minimal compression
   - Middle-range messages: moderate compression
   - Oldest messages: maximum compression
   - Preserve key elements throughout

3. Configuration Flexibility
   - Enable/disable via config
   - Adjustable window sizes
   - Customizable compression rates
   - Configurable preservation patterns

4. Quality Assurance
   - Monitor conversation coherence
   - Track information preservation
   - Measure impact on model performance
   - Validate human simulation maintenance

#### Research Topics
1. Context Optimization Impact
   - Study compression effects on conversation quality
   - Optimize rates for different conversation types
   - Develop better preservation patterns
   - Research adaptive compression strategies

Note: The complete context optimization implementation details are available in the code repository. This log preserves the original architecture while tracking the addition of new features.

=============

IMPLEMENTATION LOG

## Current Status
🟡 NOT STARTED

INSTRUCTION: Add and check off implementation steps here using APPEND to file only, read the LATEST updates in the file as progress indicators!!

