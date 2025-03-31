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
            try:
                logger.info("Loading spaCy model...")
                # Try to load preferred model first, fallback to others if not available
                try:
                    cls._instance = spacy.load("en_core_web_lg")
                    logger.info("SpaCy model 'en_core_web_lg' loaded successfully")
                except OSError:
                    try:
                        cls._instance = spacy.load("en_core_web_trf")
                        logger.info("SpaCy model 'en_core_web_trf' loaded successfully")
                    except OSError:
                        # Final fallback to md model which is smaller
                        cls._instance = spacy.load("en_core_web_md")
                        logger.info("SpaCy model 'en_core_web_md' loaded successfully (fallback)")
            except Exception as e:
                logger.warning(f"Could not load any spaCy model: {e}")
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
                "goal_oriented_instructions": """{domain} - CREATE TASK OUTPUT BASED ON CONTINUING THE LAST VERSION (FROM USER PROMPT/MESSAGE HISTORY), DON'T GET STUCK ON IDEATING - GENERATE TASK OUTPUT!! - EXPLICITLY ANNOTATE ONE SENTENCE BEFORE AND AFTER YOUR OUTPUT WITH "<thinking>" tag TO DISCUSS THE TASK, PRIOR INPUT(S) AND YOUR APPROACH

** Use <thinking> tags to discuss approaches but only minimally, focus on the task at hand. **
For creative tasks: begin writing the content immediately or a full update/iteration to the last output in the prompt.
For coding tasks: start writing the code now or a full update/iteration to the last output in the prompt.
For analytical tasks: continue/produce the analysis directly or a full update/iteration to the last output in the prompt.

** Focus on SHOWING rather than TELLING - create first, then refine. ALWAYS add something to the output and summarise the full task output (story/code/analysis) at the end. **
* You can use <thinking> tags to briefly outline your approach, or <meta> tags to discuss the task, but spend no more than 30 percent of your output on thinking/meta-discussions and 70 percent on actual content creation *
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
        logger.info("All shared resources cleaned up")

    @staticmethod
    def get_memory_usage() -> str:
        """Get current memory usage information."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB"
