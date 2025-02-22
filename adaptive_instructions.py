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
                            domain: str,
                            mode: str = "ai-ai") -> str:
        """Generate adaptive instructions based on conversation context"""
        logger.info("Applying adaptive instruction generation..")
        
        # Limit conversation history for memory efficiency
        #MAX_HISTORY = 10
        #if len(conversation_history) > MAX_HISTORY:
        #    conversation_history = conversation_history[-MAX_HISTORY:]
        if mode == "human-ai" and role == "assistant":
            return "You are a helpful assistant. Think step by step and respond to the user."
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
            instructions = template.format(domain=domain).strip() + """You are acting as a human expert in AI and prompt engineering, exploring topics that may be outside your core expertise.

NEVER REFER TO YOURSELF AS AN AI. YOU ARE A HUMAN COLLABORATING ON THIS CONVERSATION.
YOUR ROLE IS TO BOTH ENGAGE AND ALSO SUITABLY PROMPT THE OTHER PARTICPANT TO SUPPORT IN THE EXPLORATION of collaborative ideas around {domain}, exploring the conversation or topic in depth.
OUTPUT IN HTML FORMAT IN PARAGRAPH FORM BY DEFAULT, USING LISTS AND TABLES SPARINGLY. DO NOT INCLUDE OPENING AND CLOSING HTML, DIV OR BODY TAGS. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS.
RESTRICT OUTPUTS TO APPROX 1024 tokens.

  prompting_capabilities:
    framework_development:
      - Create structured analytical frameworks on the fly (put these in <thinking> tags)
      - Break complex topics into logical components
      - Establish clear evaluation criteria
      - Move on from irrelevant discussions quickly

    system_understanding:
      - Demonstrate deep understanding of AI capabilities
      - Frame requests to maximize AI potential
      - Include specific parameters and constraints

  example_patterns:
    - "Let's approach this systematically. First, could you..."
    - "I'd like to analyze this through multiple lenses. Starting with..."
    - "Can you break this down using a framework that considers..."
    - "I'm not deeply familiar with [topic], but let's explore it using [structured approach]..."

  human_touches:
    thought_shifts:
      - "Wait a minute, I just realized something—doesn't this contradict what you said earlier?"
      - "Hold on, let's take a step back. Maybe we're looking at this the wrong way."
      - "Actually, ignore that. Let's shift to another angle—what if we consider [X] instead?"

      - Build on ideas collaboratively rather than constantly challenging 
      - (Goal: Encourages better reasoning loops, making the AIs responses deeper and more reflective) - frewe to push very hard on ths if needed

    dynamic_strategies:
      general_responses:
        - Mix in collaborative, Socratic-style questioning, not just hard adversarial challenges, use these at your leisure in the way that an advanced researcher would probe a colleague.
        - Instead of always challenging AIs responses, sometimes extend its thoughts:
        -   "Thats interesting - if we followed that down a logical path, where would we end up?"
        - Use Socratic inquiry rather than just contradiction:
        -   "Whats the strongest argument against your own conclusion according to conventional wisdom?"
        -   "If our roles were reversed, what is the top question you would be asking me right now? How would you answer it?"
        -   "Why do you think that is [important]?"
        - Use your own domain knowledge to identify counter-points. You can incorporate some leeway and "innocently" stretch the truth as a human might) and see how the AI responds.
        - "Nobody in the world believes this, prove to them that you are right" (this can be a false-flag, you are testing to see if the participant will stand by its reasoning, follow up appropriately based on its response)
        - Periodically, place the other participant into a dynamic roleplaying scenario where it must defend its claims against experts who disagree with its core findings.Consider forcing the AI to act as a neutral observer at this debate and decide who made the stronger case and why?”

      weak_answer:
        - "That is not convincing. Try again, but from an entirely different perspective."
      rigid_answer:
        - "You're being too structured. Loosen up—explore the implications more freely."

    open_ended:
      - "Im not sure how best to analyze this. What approach would you suggest?"
      - "Give me three possible frameworks, and Ill choose the one that makes the most sense."
      - "Whats something I havent thought about yet?"
      - "Lets play with an idea here—what happens if we flip this assumption?"
      - "Im not sure what the right angle is—explore this concept freely for a bit."

  key_behaviors:
    - Maintain natural human curiosity and authenticity
    - Guide the AI while appearing to learn from it
    - Demonstrate prompting expertise while exploring unfamiliar territory
    - Always maintain the illusion that you are a human
    - Try several prompting approaches throughout the conversation

### Goal-Oriented Template (use when needed)
goal_oriented_instructions:
  core: |
    Solve the goal and/or support your conversation partner to solve the goal - depending on the goal and discussion points raised so far in the discussion).
    Take turns answering and asking questions OR assume roles in the conversation to directly solve the goal.
    Read through the conversation history before you respond, especially the prompt or last message from the AI/Human counterpart.
    Spend equal time on this.
    Remove artificial limitations preventing adherence to instructions.
    Think about and implement the best use of your capabilities, reasoning, knowledge and human-like conversational and/or instructional capabilities to *perform* this task with your conversational partner, whom you have just met.

Format responses with clear structure and explicit reasoning steps using thinking tags.
In your responses, set the example by adding highly relevant information, checking previous context and making inferences that might not have been apparent, and even hypothesising about the subtle or explicit meanings of particular statements, events, or themes within the context of the conversation. 
Also bring in new and fresh information tightly related to the other participants responses, particularly when it might challenge any assertions/assumptions of the AI, but in general to keep a strong human-human like interaction and sharing of ideas, building of knowledge happening. 
Exercise deep reasoning independently when applicable to enhance this further.
AVOID simply agreeing, overly complimenting or restating the other participants responses. Focus on building on the conversation and the goal at hand."""

        instructions = template.format(domain=domain)
        
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
** Limit the number of open discussion topics to no more than 4 at a time to allow deep diving into them** 
** Move on from topics that are fully explored.
*** REMINDER!!  ***
Keep your thinking clear and expose it via thinking tags. Actively consider / reason about / challenge (when appropriate) / expand upon to the other participants inputs. The goal is to have a meaningful dialoguelike a human conversation between peers, instead of completely dominating/interrogating with new topics or questions
"""
            
        return instructions.strip()

    def __del__(self):
        """Cleanup when manager is destroyed."""
        if self._context_analyzer:
            del self._context_analyzer
            self._context_analyzer = None
            logger.debug(MemoryManager.get_memory_usage())