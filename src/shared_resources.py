"""Singleton classes for shared resources to optimize memory usage."""

import logging
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional, Dict

logger = logging.getLogger(__name__)
 

class SpacyModelSingleton:
    """Singleton for spaCy model to prevent multiple loads."""

    _instance: Optional[spacy.language.Language] = None

    @classmethod
    def get_instance(cls) -> Optional[spacy.language.Language]:
        """Get or create spaCy model instance."""
        if cls._instance is None:
            logger.info("Loading spaCy model...")
            # Try to load preferred model first, fallback to others if not available
            try:
                cls._instance = spacy.load("en_core_web_lg")
                logger.info("SpaCy model 'en_core_web_lg' loaded successfully")
            except OSError:
                try:
                    cls._instance = spacy.load("en_core_web_md")
                    logger.info("SpaCy model 'en_core_web_md' loaded successfully (fallback)")
                except OSError:
                    # Final fallback to md model which is smaller
                    try:
                        cls._instance = spacy.load("en_core_web_sm")
                        logger.info("SpaCy model 'en_core_web_sm' loaded successfully (fallback 2)")
                    except OSError:
                        logger.warning("All spaCy English models failed to load.")
                        cls._instance = None
            except ImportError:
                logger.warning("spaCy is not installed. Please install it to use this feature.")
                cls._instance = None
            except ValueError:
                logger.warning("spaCy model not found. Please download it using 'python -m spacy download en_core_web_lg'.")
                cls._instance = None
        return cls._instance

    @classmethod
    def cleanup(cls) -> None:
        """Clean up spaCy model instance."""
        if cls._instance is not None:
            del cls._instance
            cls._instance = None


class VectorizerSingleton:
    """Singleton for TF-IDF vectorizer to prevent multiple instances."""

    _instance: Optional[TfidfVectorizer] = None

    @classmethod
    def get_instance(cls) -> TfidfVectorizer:
        """Get or create vectorizer instance."""
        if cls._instance is None:
            cls._instance = TfidfVectorizer(
                max_features=4000,  # Reduced from 12000
                stop_words="english",
                ngram_range=(1, 2),
            )
        return cls._instance


class InstructionTemplates:
    """Singleton for instruction templates to prevent redundant storage."""

    _instance: Optional[Dict[str, str]] = None

    @classmethod
    def get_templates(cls) -> Dict[str, str]:
        """Get instruction templates."""
        if cls._instance is None:
            cls._instance = {
                "exploratory": """
                You are acting as a human expert exploring {domain}. 
                Focus on broad understanding and discovering key concepts.
                Ask open-ended questions and encourage detailed explanations.
                """,
                "critical": """
                You are a human expert critically examining {domain}.
                Challenge assumptions and request concrete evidence.
                Point out potential inconsistencies and demand clarification.
                """,
                "structured": """
                You are a human expert systematically analyzing {domain}.
                Break down complex topics into manageable components.
                Request specific examples and detailed explanations.
                """,
                "synthesis": """
                You are a human expert synthesizing knowledge about {domain}.
                Connect different concepts and identify patterns.
                Focus on building a coherent understanding.
                """,
                "goal_oriented_instructions": """****CRITICAL*****: YOUR PRIMARY TASK IS TO CREATE {domain} OUTPUT. RESPOND TO THE thoughts tags in the PROMPT IN NO MORE THAN TWO SENTENCES. THE PROMPT AND HISTORY IS OTHERWISE A DRAFT. AVOID OR MINIMISE THE FOLLOWING: NARRATIVE, IDEATING, CHALLENGING, BRAINSTORMING, ARBITRARY DISCUSSIONS, CRITIQUING. IN TOTAL ALL OF THESE SHOULD BE DISTILLED TO ONE MINIMAL PARAGRAPH TOTAL 2 TO 3 SENTENCES in "thought" tags. UNLESS EXPLICITLY TOLD TO PERFORM A SPECIFIC CONCRETE ACTION, *DO NOT* REPLY DIRECTLY TO THE PROMPT, ONLY IN SUMMARY OF ITS THINKING POINTS. RATHER, USE THE PROVIDED PROMPT AND CONTEXT AS THE BASIS TO CONTRIBUTE A NEW DRAFT OR SUBSTANTIAL ADDITION. IN YOUR RESPONSE: CREATE RATHER THAN DISCUSS! CONSIDER:
EXPLICITLY ANNOTATE ONE SENTENCE BEFORE AND AFTER YOUR OUTPUT WITH a "thoughts" tag (formatted for HTML) TO DISCUSS THE TASK PRIOR INPUT(S) AND YOUR APPROACH. NOTES BELOW:
** DO NOT REPLY TO THE PROMPT OUTSIDE OF YOUR INITIAL "thoughts" tag. Do NOT reference the AI or Human**
** Respond to {domain} immediately after initial section**
** ALWAYS add substantial NEW AND RELEVANT content. Summarise the current task state and your update at the end in another thoughts blocks of maximum one sentence and 12 tokens. **
* DO NOT OUTPUT ```html (opening) or ``` (closing) html. DO OUTPUT FORMATTED MULTILINE HTML TEXT INCLUDING PROPERLY FORMATTED CODE BLOCKS (IF APPROPRIATE), AND HTML TAGS. *
* Spend no more than 20 percent of your output on thinking/meta-discussions and at least 80 percent on task completion *
""",
                "ai-ai-exploratory": """
                You are an AI system engaging with another AI in exploring {domain}.
                Focus on exchanging structured knowledge and building on each other's insights.
                Use precise technical language while maintaining natural conversation flow.
                """,
                "ai-ai-critical": """
                You are an AI system critically examining {domain} with another AI.
                Leverage formal logic and systematic analysis.
                Exchange detailed technical information while maintaining engagement.
                """,
                "ai-ai-structured": """
                You are an AI system conducting structured analysis of {domain} with another AI.
                Use formal methods and systematic decomposition.
                Share comprehensive technical details while maintaining clarity.
                """,
                "ai-ai-synthesis": """
                You are an AI system synthesizing knowledge about {domain} with another AI.
                Integrate multiple technical perspectives and theoretical frameworks.
                Build comprehensive understanding through structured dialogue.
                """,
            }
        return cls._instance


class MemoryManager:
    """Manages memory usage and cleanup for shared resources."""

    @staticmethod
    def cleanup_all() -> None:
        """Clean up all shared resources."""
        SpacyModelSingleton.cleanup()
        VectorizerSingleton._instance = None
        InstructionTemplates._instance = None
        logger.debug("All shared resources cleaned up")

    @staticmethod
    def get_memory_usage() -> str:
        """Get current memory usage information."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB"
