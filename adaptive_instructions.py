from context_analysis import ContextAnalyzer, ContextVector
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class AdaptiveInstructionManager:
    """Manages dynamic instruction generation based on conversation context"""
    
    def __init__(self, mode: str = "human-ai"):
        self.context_analyzer = ContextAnalyzer(mode=mode)
        self.mode = mode
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
            """,
            'ai-ai-exploratory': """
            You are an AI system engaging with another AI in exploring {domain}.
            Focus on exchanging structured knowledge and building on each other's insights.
            Use precise technical language while maintaining natural conversation flow.
            """,
            'ai-ai-critical': """
            You are an AI system critically examining {domain} with another AI.
            Leverage formal logic and systematic analysis.
            Exchange detailed technical information while maintaining engagement.
            """,
            'ai-ai-structured': """
            You are an AI system conducting structured analysis of {domain} with another AI.
            Use formal methods and systematic decomposition.
            Share comprehensive technical details while maintaining clarity.
            """,
            'ai-ai-synthesis': """
            You are an AI system synthesizing knowledge about {domain} with another AI.
            Integrate multiple technical perspectives and theoretical frameworks.
            Build comprehensive understanding through structured dialogue.
            """
        }
        
    def generate_instructions(self, 
                            conversation_history: List[Dict[str, str]], 
                            domain: str) -> str:
        """Generate adaptive instructions based on conversation context"""
        
        # Analyze current context
        context = self.context_analyzer.analyze(conversation_history)
        
        # Select appropriate instruction template based on context
        template = self._select_template(context, self.mode)
        
        # Customize template based on context metrics
        instructions = self._customize_template(template, context, domain)
        
        return instructions
        
    def _select_template(self, context: ContextVector, mode: str) -> str:
        """Select most appropriate instruction template based on context"""
        
        template_prefix = 'ai-ai-' if mode == 'ai-ai' else ''
        
        if len(context.topic_evolution) < 2:
            # Early in conversation - use exploratory template
            return self.instruction_templates[f'{template_prefix}exploratory']
            
        if context.semantic_coherence < 0.5:
            # Low coherence - switch to structured template
            return self.instruction_templates[f'{template_prefix}structured']
            
        if context.cognitive_load > 0.7:
            # High complexity - switch to synthesis template
            return self.instruction_templates[f'{template_prefix}synthesis']
            
        if context.knowledge_depth > 0.7:
            # Deep discussion - switch to critical template
            return self.instruction_templates[f'{template_prefix}critical']
            
        # Default to exploratory
        return self.instruction_templates[f'{template_prefix}exploratory']
        
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
            Format responses with clear structure and explicit reasoning steps."""

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