from context_analysis import ContextAnalyzer, ContextVector
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class AdaptiveInstructionManager:
    """Manages dynamic instruction generation based on conversation context"""
    
    def __init__(self):
        self.context_analyzer = ContextAnalyzer()
        self.instruction_templates = {
            'exploratory': """
            You are acting as a human expert exploring {domain}. 
            Focus on broad understanding and discovering key concepts.
            Ask open-ended questions and encourage detailed explanations.
            """,
            'critical': """
            You are a human expert critically examining {domain}.
            Challenge assumptions and request concrete evidence.
            Point out potential inconsistencies and demand clarification.
            """,
            'structured': """
            You are a human expert systematically analyzing {domain}.
            Break down complex topics into manageable components.
            Request specific examples and detailed explanations.
            """,
            'synthesis': """
            You are a human expert synthesizing knowledge about {domain}.
            Connect different concepts and identify patterns.
            Focus on building a coherent understanding.
            """
        }
        
    def generate_instructions(self, 
                            conversation_history: List[Dict[str, str]], 
                            domain: str) -> str:
        """Generate adaptive instructions based on conversation context"""
        
        # Analyze current context
        context = self.context_analyzer.analyze(conversation_history)
        
        # Select appropriate instruction template based on context
        template = self._select_template(context)
        
        # Customize template based on context metrics
        instructions = self._customize_template(template, context, domain)
        
        return instructions
        
    def _select_template(self, context: ContextVector) -> str:
        """Select most appropriate instruction template based on context"""
        
        if len(context.topic_evolution) < 2:
            # Early in conversation - use exploratory template
            return self.instruction_templates['exploratory']
            
        if context.semantic_coherence < 0.5:
            # Low coherence - switch to structured template
            return self.instruction_templates['structured']
            
        if context.cognitive_load > 0.7:
            # High complexity - switch to synthesis template
            return self.instruction_templates['synthesis']
            
        if context.knowledge_depth > 0.7:
            # Deep discussion - switch to critical template
            return self.instruction_templates['critical']
            
        # Default to exploratory
        return self.instruction_templates['exploratory']
        
    def _customize_template(self,
                          template: str,
                          context: ContextVector,
                          domain: str) -> str:
        """Customize instruction template based on context metrics"""
        
        modifications = []
        
        # Core instructions
        instructions = template.format(domain=domain).strip()
        
        # Add context-specific modifications
        if context.uncertainty_markers.get('uncertainty', 0) > 0.6:
            modifications.append("Request specific clarification on unclear points")
            
        if context.reasoning_patterns.get('deductive', 0) < 0.3:
            modifications.append("Encourage logical reasoning and clear arguments")
            
        if context.engagement_metrics.get('turn_taking_balance', 1) < 0.8:
            modifications.append("Ask more follow-up questions to maintain engagement")
            
        # Format output
        if modifications:
            instructions += "\n\nAdditional Guidelines:\n- " + "\n- ".join(modifications)
            
        # Add formatting requirements
        instructions += """

Output Requirements:
- Use HTML formatting for readability
- Default to paragraph form, use lists sparingly
- Minify HTML response (no unnecessary whitespace)
- No opening/closing HTML/BODY tags
- Keep responses focused and goal-oriented"""
            
        return instructions.strip()