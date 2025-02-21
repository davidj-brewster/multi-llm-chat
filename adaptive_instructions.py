from context_analysis import ContextAnalyzer, ContextVector
from typing import List, Dict, Optional
import logging
from shared_resources import InstructionTemplates, MemoryManager

logger = logging.getLogger(__name__)

class AdaptiveInstructionManager:
    """Manages dynamic instruction generation based on conversation context"""
    
    def __init__(self, mode: str = "human-ai"):
        self._context_analyzer = None  # Lazy initialization
        self.mode = mode
        
    @property
    def context_analyzer(self):
        """Lazy initialization of context analyzer."""
        if self._context_analyzer is None:
            self._context_analyzer = ContextAnalyzer(mode=self.mode)
        return self._context_analyzer
        
    def generate_instructions(self, 
                            conversation_history: List[Dict[str, str]], 
                            domain: str) -> str:
        """Generate adaptive instructions based on conversation context"""
        logger.info("Applying adaptive instruction generation..")
        
        # Limit conversation history for memory efficiency
        MAX_HISTORY = 10
        if len(conversation_history) > MAX_HISTORY:
            conversation_history = conversation_history[-MAX_HISTORY:]
        
        # Analyze current context
        context = self.context_analyzer.analyze(conversation_history)
        
        # Select appropriate instruction template based on context
        template = self._select_template(context, self.mode)
        
        # Customize template based on context metrics
        instructions = self._customize_template(template, context, domain)
        
        # Log memory usage in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(MemoryManager.get_memory_usage())
            
        logger.debug("New prompt: {}".format(instructions))
        return instructions
        
    def _select_template(self, context: ContextVector, mode: str) -> str:
        """Select most appropriate instruction template based on context"""
        templates = InstructionTemplates.get_templates()
        
        template_prefix = 'ai-ai-' if mode == 'ai-ai' else ''
        
        if len(context.topic_evolution) < 2:
            # Early in conversation - use exploratory template
            logger.debug("_select_template: Early in conversation - using exploratory template")
            return templates[f'{template_prefix}exploratory']
            
        if context.semantic_coherence < 0.5:
            # Low coherence - switch to structured template
            logger.debug("_select_template: low coherence - using structured template")
            return templates[f'{template_prefix}structured']
            
        if context.cognitive_load > 0.8:
            # High complexity - switch to synthesis template
            logger.debug("_select_template: low coherence - using structured template")
            return templates[f'{template_prefix}synthesis']
            
        if context.knowledge_depth > 0.8:
            # Deep discussion - switch to critical template
            logger.debug("_select_template: low coherence - Deep discussion - switch to critical template")
            return templates[f'{template_prefix}critical']
            
        # Default to exploratory
        logger.debug("_select_template: Defaulting to exploratory template")
        return templates[f'{template_prefix}exploratory']
        
    def _customize_template(self,
                          template: str,
                          context: ContextVector,
                          domain: str) -> str:
        """Customize instruction template based on context metrics"""
        
        modifications = []
        
        # Core instructions
        if self.mode == "ai-ai":
            # Add mode-specific formatting for AI-AI conversations
            instructions = template.format(domain=domain).strip() + """
            Format responses with clear structure and explicit reasoning steps.
            In your responses, set the example by adding highly relevant information, checking previous context and making inferences that might not have been apparent, and even hypothesising about the subtle or explicit meanings of particular statements, events, or themes within the context of the conversation. Also bring in new and fresh information tightly related to the AIs responses, particularly when it might challenge any assertions/assumptions of the AI, but in general to keep a strong human-human like interaction and sharing of ideas, building of knowledge happening. Exercise deep reasoning when applicable to enhance this further."""

        instructions = template.format(domain=domain).strip()
        
        # Add context-specific modifications
        if context.uncertainty_markers.get('uncertainty', 0) > 0.6:
            modifications.append("Request specific clarification on unclear points")
            
        if context.reasoning_patterns.get('deductive', 0) < 0.3:
            modifications.append("Encourage logical reasoning and clear arguments")
            
        # Add AI-AI specific modifications if in AI-AI mode
        if self.mode == "ai-ai":
            if context.reasoning_patterns.get('formal_logic', 0) < 0.3:
                modifications.append("Use more formal logical structures in responses")
            if context.reasoning_patterns.get('technical', 0) < 0.4:
                modifications.append("Increase use of precise technical terminology")
            
        if context.engagement_metrics.get('turn_taking_balance', 1) < 0.4:
            modifications.append("Ask more follow-up questions to maintain engagement")
        
        if "GOAL" in domain:
            modifications.append("** Focus on achieving the specified goal! **")
        
        # Format output
        if modifications:
            instructions += "\n\nAdditional Guidelines:\n- " + "\n- ".join(modifications)
            
        # Add formatting requirements
        instructions += """

**Output**:
- Use HTML formatting for readability
- Use Code blocks or Quote blocks (foramtted to appear as such in HTML) as required and ensure they are formatted correctly
- Use HTML bullet points for lists
- Default to paragraph form, use lists sparingly
- No opening/closing HTML/BODY tags
*** REMINDER!!  ***
Keep your thinking clear and expose it via thinking tags. Actively consider and respond to the other participants' inputs and build upon them to ensure the conversation stays two sided more like a human conversation between peers, instead of completely dominating/interrogating with new topics or questions"""
            
        return instructions.strip()

    def __del__(self):
        """Cleanup when manager is destroyed."""
        if self._context_analyzer:
            del self._context_analyzer
            self._context_analyzer = None
            logger.debug(MemoryManager.get_memory_usage())