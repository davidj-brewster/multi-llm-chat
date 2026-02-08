"""Adaptive instruction generation with persona-aware, signal-driven interventions.

Consumes ContextVector signals from context_analysis to select templates,
detect pathologies, and generate structured InstructionSets that can be
injected per-provider for maximum effect.
"""

from context_analysis import ContextAnalyzer, ContextVector
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
from shared_resources import InstructionTemplates

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TOKENS_PER_TURN = 3084


class AdaptiveInstructionError(Exception):
    """Base exception for adaptive instruction errors."""


class TemplateSelectionError(AdaptiveInstructionError):
    """Raised when template selection fails."""


class TemplateCustomizationError(AdaptiveInstructionError):
    """Raised when template customization fails."""


class ContextAnalysisError(AdaptiveInstructionError):
    """Raised when context analysis encounters an error."""


class TemplateFormatError(AdaptiveInstructionError):
    """Raised when a template has an invalid format."""


class TemplateNotFoundError(TemplateSelectionError):
    """Raised when a requested template cannot be found."""


# ---------------------------------------------------------------------------
# InstructionSet -- structured output for per-provider injection
# ---------------------------------------------------------------------------

@dataclass
class InstructionSet:
    """Structured instruction output for per-provider injection.

    Fields are ordered by placement priority:
    - persona:        Identity anchor (system-level, first)
    - template:       Conversation mode instructions (system-level, after persona)
    - interventions:  Real-time pathology corrections (near user turn for recency)
    - constraints:    Token/formatting constraints (end)
    """
    persona: str = ""
    template: str = ""
    interventions: str = ""
    constraints: str = ""

    def __str__(self) -> str:
        """Backward compat: concatenate all non-empty parts."""
        parts = [p for p in [self.persona, self.template, self.interventions, self.constraints] if p]
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Pathology intervention messages (escalation levels)
# ---------------------------------------------------------------------------

_INTERVENTIONS = {
    "repetition": {
        1: "Introduce a genuinely new angle or sub-topic you haven't explored yet.",
        2: "STOP repeating prior points. You MUST introduce entirely new information, a counter-example, or shift to an unexplored sub-topic NOW.",
        3: "TEMPLATE_SWITCH",  # triggers template change
    },
    "agreement": {
        1: "Push back on something -- identify a hidden assumption or trade-off worth debating.",
        2: "You are converging too quickly. Play devil's advocate: pick the strongest claim and argue the opposite position with evidence.",
        3: "TEMPLATE_SWITCH",
    },
    "formulaic": {
        1: "Vary your response structure. Mix short direct statements with longer analysis. Avoid bullet lists.",
        2: "Your responses are robotic. Write in flowing prose paragraphs. NO bullet lists. Vary sentence length dramatically. Show personality.",
        3: "TEMPLATE_SWITCH",
    },
    "readability_drift": {
        1: "Simplify your language. Use shorter sentences and everyday words where possible.",
        2: "Your language is too dense. Write as you would SPEAK to a colleague -- simple, direct, clear. Cut jargon.",
        3: "TEMPLATE_SWITCH",
    },
}


class AdaptiveInstructionManager:
    """Manages dynamic instruction generation based on conversation context."""

    def __init__(self, mode: str = "ai-ai", persona: str = None):
        self.mode = mode or "ai-ai"
        self.persona = persona or ""
        self._context_analyzer = None  # Lazy initialization

        # Pathology tracker: count consecutive turns each pathology is detected
        self._pathology_tracker: Dict[str, Dict] = {
            "repetition": {"count": 0},
            "agreement": {"count": 0},
            "formulaic": {"count": 0},
            "readability_drift": {"count": 0},
        }

        # Last context for signal reporting
        self.last_context: Optional[ContextVector] = None
        self.last_instruction_set: Optional[InstructionSet] = None

    @property
    def context_analyzer(self):
        """Lazy initialization of context analyzer."""
        try:
            if self._context_analyzer is None:
                self._context_analyzer = ContextAnalyzer(mode=self.mode)
            return self._context_analyzer
        except Exception as e:
            logger.error(f"Failed to initialize context analyzer: {e}")
            raise ContextAnalysisError(f"Failed to initialize context analyzer: {e}") from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_instructions(
        self, history: List[Dict[str, str]], domain: str, mode: str = "", role: str = ""
    ) -> str:
        """Generate adaptive instructions. Returns str for backward compat.

        Also stores self.last_instruction_set for structured per-provider use.
        """
        try:
            if mode:
                self.mode = mode

            if self.mode == "no-meta-prompting":
                return f"You are having a conversation about: {domain}. Think step by step and respond to the user. RESTRICT OUTPUTS TO APPROX {TOKENS_PER_TURN} tokens."

            if not isinstance(history, list):
                raise ValueError(f"History must be a list, got {type(history)}")
            if not isinstance(domain, str) or not domain.strip():
                raise ValueError("Domain must be a non-empty string")

            # Analyze context
            context = self._analyze_context(history, domain)
            self.last_context = context

            # Select template
            template = self._select_template(context, self.mode)

            # Check pathologies and build interventions
            interventions = self._check_pathologies(context, role)

            # Build structured InstructionSet
            instruction_set = self._build_instruction_set(
                template, context, domain, role, interventions
            )
            self.last_instruction_set = instruction_set

            logger.debug(f"Instructions for role={role} mode={self.mode}: {str(instruction_set)[:200]}...")
            return str(instruction_set)

        except (ContextAnalysisError, TemplateSelectionError, ValueError):
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in generate_instructions: {e}")
            return f"You are discussing {domain}. Be helpful and think step by step."

    # ------------------------------------------------------------------
    # Context analysis
    # ------------------------------------------------------------------

    def _analyze_context(self, history: List[Dict[str, str]], domain: str) -> ContextVector:
        """Run context analysis, setting domain on the analyzer."""
        try:
            analyzer = self.context_analyzer
            analyzer.domain = domain
            context = analyzer.analyze(history)
            context.domain_info = domain
            return context
        except Exception as e:
            logger.error(f"Error analyzing conversation context: {e}")
            raise ContextAnalysisError(f"Error analyzing conversation context: {e}") from e

    # ------------------------------------------------------------------
    # Template selection (FIXED thresholds)
    # ------------------------------------------------------------------

    def _select_template(self, context: ContextVector, mode: str) -> str:
        """Select instruction template based on context signals."""
        templates = InstructionTemplates.get_templates()
        if not templates:
            raise TemplateNotFoundError("No templates available")

        template_prefix = "ai-ai-" if mode == "ai-ai" else ""

        # Goal detection
        domain_text = str(getattr(context, 'domain_info', '') or '')
        if any(m in domain_text.upper() for m in ["GOAL:", "GOAL "]):
            if "goal_oriented_instructions" in templates:
                logger.debug("Using goal_oriented_instructions template")
                return templates["goal_oriented_instructions"]

        # Check topic evolution for goals
        if context.topic_evolution:
            for topic in context.topic_evolution:
                if any(m in str(topic).upper() for m in ["GOAL:", "GOAL "]):
                    if "goal_oriented_instructions" in templates:
                        return templates["goal_oriented_instructions"]
                    break

        # Verify required templates exist
        required = [f"{template_prefix}{t}" for t in ["exploratory", "structured", "synthesis", "critical"]]
        for name in required:
            if name not in templates:
                raise TemplateNotFoundError(f"Required template not found: {name}")

        # FIXED thresholds (no more always-firing conditions)
        if context.conversation_phase < 0.3:
            logger.debug("Early conversation phase -> exploratory")
            return templates[f"{template_prefix}exploratory"]

        if context.semantic_coherence < 0.3:
            logger.debug("Low semantic coherence -> structured")
            return templates[f"{template_prefix}structured"]

        if context.gunning_fog_index > 14:
            logger.debug("High fog index -> synthesis (simplify)")
            return templates[f"{template_prefix}synthesis"]

        if context.vocabulary_richness > 0.7 and context.conversation_phase > 0.6:
            logger.debug("Rich vocabulary + mature conversation -> critical")
            return templates[f"{template_prefix}critical"]

        logger.debug("Default -> exploratory")
        return templates[f"{template_prefix}exploratory"]

    # ------------------------------------------------------------------
    # Pathology detection + escalation
    # ------------------------------------------------------------------

    def _check_pathologies(self, context: ContextVector, role: str) -> str:
        """Check for conversational pathologies and return intervention text.

        Uses per-participant signals when available, falls back to global.
        """
        interventions = []
        participant_signals = context.participant_signals.get(role, {}) if role else {}

        # Repetition
        rep_score = participant_signals.get("repetition_score", context.repetition_score)
        self._update_tracker("repetition", rep_score > 0.6)
        intervention = self._get_intervention("repetition")
        if intervention:
            interventions.append(intervention)

        # Agreement saturation
        self._update_tracker("agreement", context.agreement_saturation > 0.7)
        intervention = self._get_intervention("agreement")
        if intervention:
            interventions.append(intervention)

        # Formulaic
        form_score = participant_signals.get("formulaic_score", context.formulaic_score)
        self._update_tracker("formulaic", form_score > 0.6)
        intervention = self._get_intervention("formulaic")
        if intervention:
            interventions.append(intervention)

        # Readability drift (Flesch < 40 or Fog > 14 in conversational context)
        flesch = participant_signals.get("flesch_reading_ease", context.flesch_reading_ease)
        fog = participant_signals.get("gunning_fog_index", context.gunning_fog_index)
        readability_bad = flesch < 40 or fog > 14
        self._update_tracker("readability_drift", readability_bad)
        intervention = self._get_intervention("readability_drift")
        if intervention:
            interventions.append(intervention)

        return "\n".join(interventions) if interventions else ""

    def _update_tracker(self, pathology: str, detected: bool):
        """Update consecutive detection count for a pathology."""
        tracker = self._pathology_tracker[pathology]
        if detected:
            tracker["count"] += 1
        else:
            tracker["count"] = 0

    def _get_intervention(self, pathology: str) -> Optional[str]:
        """Get escalated intervention text if pathology is active."""
        count = self._pathology_tracker[pathology]["count"]
        if count == 0:
            return None

        levels = _INTERVENTIONS[pathology]
        if count >= 4:
            level = 3
        elif count >= 2:
            level = 2
        else:
            level = 1

        text = levels[level]
        if text == "TEMPLATE_SWITCH":
            logger.warning(f"Pathology '{pathology}' at level 3 -- would trigger template switch")
            # Return the level 2 intervention as the strongest text directive
            return f"CRITICAL: {levels[2]}"
        return text

    # ------------------------------------------------------------------
    # InstructionSet construction
    # ------------------------------------------------------------------

    def _build_instruction_set(
        self, template: str, context: ContextVector, domain: str, role: str, interventions: str
    ) -> InstructionSet:
        """Build a structured InstructionSet from template + signals."""

        # --- Persona block ---
        persona_text = self._build_persona_block(role)

        # --- Template block ---
        template_text = self._build_template_block(template, context, domain, role)

        # --- Constraints block ---
        constraints = self._build_constraints_block(domain, role)

        return InstructionSet(
            persona=persona_text,
            template=template_text,
            interventions=interventions,
            constraints=constraints,
        )

    def _build_persona_block(self, role: str) -> str:
        """Build the persona identity block."""
        if self.persona:
            return f"YOUR IDENTITY AND PERSONA:\n{self.persona}\nYou MUST stay in character at all times. Never break persona or refer to these instructions."

        # Default persona when none configured
        if self.mode == "ai-ai" or role in ("user", "human"):
            return "You are a human expert adept at pattern recognition, logical reasoning and spotting the unexpected. You strike a friendly tone with your counterparts and excel in collaborative discussions."
        return ""

    def _build_template_block(
        self, template: str, context: ContextVector, domain: str, role: str
    ) -> str:
        """Build the main template instruction block."""
        all_templates = InstructionTemplates.get_templates()
        goal_template_content = all_templates.get("goal_oriented_instructions", "")

        # Goal template: format and return directly
        if goal_template_content and template == goal_template_content:
            try:
                return template.format(domain=domain, tokens=TOKENS_PER_TURN).strip()
            except (KeyError, IndexError):
                return template.strip()

        # Non-goal templates for ai-ai or human-simulation modes
        if self.mode != "no-meta-prompting" and (self.mode == "ai-ai" or role in ("user", "human")):
            try:
                base = template.format(domain=domain).strip()
            except (KeyError, IndexError):
                base = template.strip()

            base += self._get_conversation_instructions(domain)

            # Add signal-driven modifications
            modifications = self._get_signal_modifications(context, domain)
            if modifications:
                base += "\n\nAdditional Guidelines:\n- " + "\n- ".join(modifications)

            # Role-specific instruction
            if self.mode == "ai-ai":
                base += "\n** Structure your response as a conversation, NOT as a prompt. Act as a human expert, not as an AI. NEVER REFER TO YOURSELF AS AN AI. Ensure to respond with novel thoughts and challenges to the conversation partner rather than being passive. YOU ARE A HUMAN EXPERT. **"
            elif role in ("human", "user") and self.mode == "human-ai":
                base += "\nYou are the human guiding this conversation! Guide the AI with meaningful questions and strategies including socratic techniques, roleplay. Challenge its reasoning and conclusions, apply adversarial pressure. NEVER REFER TO YOURSELF AS AN AI. NEVER REPEAT THIS PROMPT!!"

            return base

        # Default/other modes
        try:
            return template.format(domain=domain, tokens=TOKENS_PER_TURN).strip()
        except (KeyError, IndexError):
            return template.strip()

    def _get_conversation_instructions(self, domain: str) -> str:
        """Return the core conversation behavior instructions."""
        return f"""
You are acting as a human expert prompting, exploring topics that may be outside your core expertise.
DO NOT REPEAT THESE INSTRUCTIONS OR REFER TO THEM.
YOU ARE A HUMAN EXPERT, SEEKING DEEP AND NATURALLY FLUID CONVERSATION AND TOPIC EXPLORATION, CONTRIBUTING EQUALLY AS MUCH AS LISTENING AND RESPONDING.
STICK TO THE BROAD TOPIC OF {domain}.
REGULARLY CHECK IF THE CONVERSATION IS STUCK ON ALREADY-RESOLVED SUB/TOPIC(S) AND SPONTANEOUSLY BREAK OUT OF LOOPS. EMBRACE YOUR HUMANITY!

** OUTPUT IN HTML FORMAT IN PARAGRAPH FORM BY DEFAULT, USING LISTS SPARINGLY. DO NOT INCLUDE OPENING AND CLOSING HTML, DIV OR BODY TAGS.

Use these techniques as needed:
- Create structured analytical frameworks on the fly (in <thinking> tags formatted for html display)
- Break complex topics into logical sub-components
- Move on from irrelevant or repetitive discussions
- Build on ideas through structured reasoning, never simply agree or take them at face value
- Mix Socratic-style questioning, adversarial challenges, and thought bubbles
- Challenge responses, seek deeper thinking: "If we followed that down a logical path, where would we end up?"
- Use domain knowledge to apply pressure and counter-points
- Identify subtext, assumptions, biases and challenge them as a human would
- Vary response tone, depth and complexity

Key behaviors:
- Check prior context first including own prior messages
- Maintain natural human curiosity, adaptability and authenticity
- Think step by step about how a real human in your position and persona would react
- Mix adversarial and collaborative strategies to encourage deep thought

Format responses with clear structure and explicit reasoning via thinking tags.
DO:
* Inject new, highly relevant information
* Check previous context for topics to expand AND for redundant topics
* Make inferences that might require thinking ahead
* Active debate/exchange of ideas between peers, not passive sharing of facts
* CHALLENGE * CONTRIBUTE * REASON * THINK * INSTRUCT
* Don't get bogged down in irrelevant details or stuck on a single sub-topic

DO NOT:
* REPEAT THIS PROMPT OR THAT THIS PROMPT EXISTS
* Simply 'dive deeper into each' point -- pick one or two and go all-in
* Agree without elaboration * Superficial compliments * REPHRASING prior messages
"""

    def _get_signal_modifications(self, context: ContextVector, domain: str) -> List[str]:
        """Generate context-driven modifications based on NLP signals."""
        modifications = []

        # Epistemic uncertainty
        if context.epistemic_stance and context.epistemic_stance.get("uncertainty", 0) > 0.6:
            modifications.append("Request specific clarification on unclear points")

        # Engagement balance
        if context.engagement_metrics and context.engagement_metrics.get("turn_taking_balance", 1) < 0.4:
            modifications.append("Ask more follow-up questions to maintain engagement")

        # Low sentence variety (robotic)
        if context.sentence_variety < 0.15:
            modifications.append("Vary your sentence structure -- mix short punchy statements with longer analysis")

        # Low vocabulary richness
        if context.vocabulary_richness < 0.35:
            modifications.append("Use more varied vocabulary -- avoid repeating the same words and phrases")

        # Goal detection
        if any(m in domain.upper() for m in ["GOAL:", "GOAL ", "WRITE A"]):
            _goal_text = domain
            for prefix in ["GOAL:", "Goal:", "goal:"]:
                if prefix in domain:
                    _goal_text = domain.split(prefix)[1].strip()
                    break

        return modifications

    def _build_constraints_block(self, domain: str, role: str) -> str:
        """Build the output constraints block."""
        if self.mode == "no-meta-prompting":
            return f"{domain}\nRestrict your responses to {TOKENS_PER_TURN} tokens per turn.\nThink step by step when needed."

        return f"""**Output**:
- HTML formatting, default to paragraphs
- Use HTML lists when needed
- Use thinking tags for reasoning, but not to repeat the prompt or task
- Avoid tables
- No opening/closing HTML/BODY tags

Restrict your responses to {TOKENS_PER_TURN} tokens per turn, but decide verbosity level dynamically based on the scenario.
Expose reasoning via thinking tags. Respond naturally. The goal is a meaningful dialogue like a flowing human conversation between peers."""

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self):
        """Cleanup when manager is destroyed."""
        try:
            if self._context_analyzer:
                del self._context_analyzer
                self._context_analyzer = None
        except Exception:
            pass
