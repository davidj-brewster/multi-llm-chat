# Conversation Reporting Integration Guide

This document explains how to integrate the conversation reporting system with the AI Battle framework.

## Overview

The reporting system provides:
- Metrics tracking for conversations
- Strategy analysis and effectiveness tracking
- Configuration visualization
- Interactive HTML reports with charts
- Integration with existing HTML output

## Integration Points

### 1. ConversationManager Class

Add metrics and strategy tracking to the ConversationManager:

```python
from conversation_reporting import ConversationMetrics, StrategyTracker

class ConversationManager:
    def __init__(self, ...):
        # Existing initialization
        self.metrics = ConversationMetrics()
        self.strategy_tracker = StrategyTracker()
```

### 2. Conversation Turn Processing

Update run_conversation_turn to collect metrics:

```python
def run_conversation_turn(self, prompt: str, model_type: str, ...):
    # Before generating response
    turn_number = len(self.conversation_history)
    
    # Track strategy selection
    strategy = self._select_conversation_strategy(prompt, history)
    self.strategy_tracker.add_strategy(
        turn=turn_number,
        model=model_type,
        strategy=strategy,
        context=prompt
    )
    
    # Generate response
    response = ...
    
    # Collect metrics
    self.metrics.add_turn_metrics(turn_number, {
        "semantic_coherence": self._calculate_coherence(response),
        "topic_relevance": self._calculate_relevance(response, self.domain),
        "engagement_level": self._calculate_engagement(response),
        "reasoning_depth": self._calculate_reasoning_depth(response),
        "knowledge_integration": self._calculate_knowledge_integration(response)
    })
    
    # Track model performance
    self.metrics.add_model_metrics(model_type, {
        "response_quality": self._assess_response_quality(response),
        "strategy_adherence": self._assess_strategy_adherence(response, strategy)
    })
    
    # Update strategy effectiveness
    effectiveness = self._calculate_strategy_effectiveness(response, strategy)
    self.strategy_tracker.update_effectiveness(strategy, effectiveness)
    
    return response
```

### 3. Strategy Adaptation

Track strategy changes in the adaptive instruction system:

```python
def _adapt_strategy(self, current_strategy: str, metrics: Dict) -> str:
    new_strategy = self._select_new_strategy(metrics)
    
    self.strategy_tracker.add_adaptation(
        turn=len(self.conversation_history),
        reason=self._get_adaptation_reason(metrics),
        old_strategy=current_strategy,
        new_strategy=new_strategy
    )
    
    return new_strategy
```

### 4. Report Generation

Update the save_conversation function to include metrics and strategy analysis:

```python
def save_conversation(self, conversation: List[Dict], filename: str, ...):
    from conversation_reporting import generate_conversation_report
    
    # Generate comprehensive report
    generate_conversation_report(
        config=self.config,
        conversation=conversation,
        metrics=self.metrics,
        strategy_tracker=self.strategy_tracker,
        output_path=filename
    )
```

## Metric Collection Functions

Implement these helper functions in ConversationManager:

```python
def _calculate_coherence(self, response: str) -> float:
    """Calculate semantic coherence of response"""
    # Use existing context analysis system
    return self.context_analyzer.calculate_coherence(response)

def _calculate_relevance(self, response: str, domain: str) -> float:
    """Calculate topic relevance score"""
    return self.context_analyzer.calculate_relevance(response, domain)

def _calculate_engagement(self, response: str) -> float:
    """Calculate engagement level based on response characteristics"""
    return self.context_analyzer.calculate_engagement(response)

def _calculate_reasoning_depth(self, response: str) -> float:
    """Analyze depth of reasoning in response"""
    # Look for thinking tags, logical structure, etc.
    return self.context_analyzer.calculate_reasoning_depth(response)

def _calculate_knowledge_integration(self, response: str) -> float:
    """Measure how well response integrates previous knowledge"""
    return self.context_analyzer.calculate_knowledge_integration(
        response, self.conversation_history
    )
```

## Strategy Tracking Functions

Add these functions to track conversation strategies:

```python
def _select_conversation_strategy(self, prompt: str, history: List[Dict]) -> str:
    """Select appropriate conversation strategy based on context"""
    return self.adaptive_manager.select_strategy(prompt, history)

def _assess_strategy_adherence(self, response: str, strategy: str) -> float:
    """Measure how well response follows selected strategy"""
    return self.adaptive_manager.assess_adherence(response, strategy)

def _calculate_strategy_effectiveness(self, response: str, strategy: str) -> float:
    """Calculate effectiveness score for strategy"""
    return self.adaptive_manager.calculate_effectiveness(response, strategy)
```

## HTML Integration

The reporting system integrates with the existing HTML output:

1. Configuration Section:
   - Added before conversation transcript
   - Includes model settings, file inputs, timeouts
   - Interactive charts for metrics

2. Metrics Section:
   - Shows conversation quality over time
   - Per-model performance metrics
   - Strategy effectiveness charts

3. Strategy Section:
   - Strategy usage and adaptation visualization
   - Effectiveness tracking
   - Context for strategy changes

4. Enhanced Conversation Transcript:
   - Existing message formatting preserved
   - Added metrics per message
   - Strategy annotations
   - Thinking tag visualization

## Usage Example

```python
# Initialize manager with reporting
manager = ConversationManager(...)

# Run conversation
conversation = await manager.run_conversation(...)

# Generate report with metrics
safe_prompt = manager._sanitize_filename_part(initial_prompt)
time_stamp = datetime.datetime.now().strftime("%m%d-%H%M")
filename = f"conversation-report_{safe_prompt}_{time_stamp}.html"

# Save comprehensive report
manager.save_conversation(
    conversation=conversation,
    filename=filename,
    human_model=human_model,
    ai_model=ai_model
)
```

## Best Practices

1. Metric Collection
   - Collect metrics for every turn
   - Use consistent scoring scales (0.0-1.0)
   - Include confidence scores
   - Handle missing data gracefully

2. Strategy Tracking
   - Document strategy selection criteria
   - Track adaptation reasons
   - Monitor effectiveness trends
   - Adjust thresholds based on data

3. Report Generation
   - Use clear visualizations
   - Include raw data for analysis
   - Maintain consistent styling
   - Optimize for readability

4. Performance Considerations
   - Cache metric calculations
   - Optimize chart rendering
   - Use incremental updates
   - Handle large conversations efficiently

## Future Enhancements

1. Advanced Analytics
   - Conversation pattern detection
   - Strategy optimization
   - Model comparison tools
   - Performance prediction

2. Enhanced Visualization
   - Real-time metrics updates
   - Interactive strategy analysis
   - Custom chart configurations
   - Export capabilities

3. Integration Features
   - API for external analysis
   - Custom metric plugins
   - Strategy suggestion system
   - Automated optimization