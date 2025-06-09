"""Base and model-specific client implementations with memory optimizations."""

import os
import logging
import random
from typing import List, Dict, Optional, Any, TypeVar
from dataclasses import dataclass
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
from openai import OpenAI
from anthropic import Anthropic, APIError, APIStatusError, APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
from ollama import ChatResponse, APIError as OllamaAPIError, ResponseError as OllamaResponseError, RequestError as OllamaRequestError
from ollama import Client
from langchain_ollama import ChatOllama
from langchain_core.exceptions import OutputParserException, LangChainError
from langchain.prompts import HumanMessagePromptTemplate as HumanMessage, AIMessagePromptTemplate as AIMessage
from langchain.prompts import SystemMessagePromptTemplate as SystemMessage, PromptTemplate, StringPromptTemplate
#from langchain.prompts import MessagesPlaceholder
import requests # Ensure requests is imported for requests.exceptions
from adaptive_instructions import AdaptiveInstructionManager
from shared_resources import MemoryManager
from configuration import detect_model_capabilities

logger = logging.getLogger(__name__)

T = TypeVar("T")

class ModelClientError(Exception):
    """Base exception for all model client errors."""
    pass

class ModelClientAuthError(ModelClientError):
    """Raised for authentication errors (e.g., invalid API key)."""
    pass

class ModelClientAPIError(ModelClientError):
    """Raised for general API errors from the model provider."""
    pass

class ModelClientTimeoutError(ModelClientError):
    """Raised when an API request times out."""
    pass

class ModelClientResponseError(ModelClientError):
    """Raised for issues with the response from the model (e.g., unexpected format)."""
    pass

# openai_api_key = os.getenv("OPENAI_API_KEY") # Handled in OpenAIClient
# anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") # Handled in ClaudeClient
# MAX_TOKENS = 3192 # Moved to ModelConfig in configdataclasses.py
TOKENS_PER_TURN = 3192 # Retained for now, if used for specific prompt construction


# ModelConfig is now imported from configdataclasses
from configdataclasses import ModelConfig


class BaseClient:
    """
    Abstract base class for AI model client implementations with validation and conversation management.

    This class provides a common interface and shared functionality for interacting with
    various AI model APIs (OpenAI, Claude, Gemini, etc.). It handles API key validation,
    capability detection, instruction management, conversation context analysis, and
    file content preparation for different model providers.

    The BaseClient implements core functionality that is common across all model providers,
    while model-specific implementations are handled by subclasses. This design follows
    the Template Method pattern, where the base class defines the skeleton of operations
    and subclasses override specific steps.

    Attributes:
        api_key (str): The API key for authentication with the model provider.
        domain (str): The knowledge domain or subject area for the conversation.
        mode (str): The conversation mode (e.g., "ai-ai", "human-ai", "default").
        role (str): The role this client is playing (e.g., "human", "assistant").
        model (str): The specific model identifier being used.
        capabilities (Dict[str, bool]): Dictionary of supported capabilities (vision, streaming, etc.).
        instructions (Optional[str]): System instructions for the model.
        adaptive_manager (AdaptiveInstructionManager): Manager for dynamic instructions.

    Implementation Notes:
        Subclasses must implement the generate_response method and should override
        validate_connection to provide model-specific validation logic. The BaseClient
        provides utility methods for conversation analysis, instruction management,
        and file content preparation that subclasses can leverage.

    Examples:
        Basic initialization:
        >>> client = BaseClient(mode="ai-ai", api_key="sk-...", domain="science")
        >>> client.validate_connection()
        True

        Custom instruction handling:
        >>> client = BaseClient(mode="human-ai", api_key="sk-...", domain="medicine")
        >>> client.instructions = "You are a helpful AI assistant."
        >>> client._get_mode_aware_instructions(role="assistant")
        'You are a medical assistant specializing in diagnostics.'

        Conversation analysis:
        >>> history = [{"role": "user", "content": "What causes headaches?"}]
        >>> analysis = client._analyze_conversation(history)
        >>> analysis["summary"]
        '<p>Previous exchanges:</p><p>Human: What causes headaches?</p>'

        File content preparation:
        >>> image_data = {
        ...     "type": "image",
        ...     "base64": "base64_encoded_data",
        ...     "mime_type": "image/jpeg",
        ...     "dimensions": (800, 600)
        ... }
        >>> prepared = client._prepare_file_content(image_data)
        >>> prepared["type"]
        'image'
    """

    def __init__(
        self,
        mode: str,
        domain: str = "",
        model: str = "",
        role: str = "",
        api_key: Optional[str] = None,
    ):
        """
        Initialize a new BaseClient instance.

        Creates a new client with the specified configuration and initializes the
        adaptive instruction manager. Validates and processes the API key and
        detects model capabilities based on the model identifier.

        Args:
            mode (str): The conversation mode to use. Valid values include:
                - "ai-ai": Both participants are AI models
                - "human-ai": One human participant and one AI model
                - "default": Standard assistant mode
            domain (str, optional): The knowledge domain or subject area for the
                conversation. Used for generating context-aware instructions.
                Defaults to an empty string.
            model (str, optional): The specific model identifier to use. This is used
                to detect capabilities and may be overridden by subclasses.
                Defaults to an empty string.
            role (str, optional): The role this client is playing in the conversation.
                Valid values include "human", "user", "assistant", or "model".
                Defaults to an empty string.
            api_key (Optional[str], optional): The API key for authentication.
                If None, subclasses are responsible for loading from environment or raising error.
                Defaults to None.

        Note:
            Subclasses typically override this method to initialize model-specific
            clients and configurations while calling super().__init__() to ensure
            base functionality is properly initialized.

        Example:
            >>> client = BaseClient(
            ...     mode="ai-ai",
            ...     domain="Artificial Intelligence",
            ...     model="gpt-4",
            ...     role="assistant",
            ...     api_key="sk-..."
            ... )
        """
        self.api_key = api_key.strip() if api_key else None # Store resolved key
        self.domain = domain
        self.mode = mode
        self.role = role
        self.model = model
        # self._adaptive_manager = None  # Lazy initialization
        self.capabilities = detect_model_capabilities(model)
        self.instructions = None
        self.adaptive_manager = AdaptiveInstructionManager(mode=self.mode)

    def _prepare_file_content(self, file_data: Dict[str, Any]) -> Any:
        """
        Prepare file content for model API consumption.

        Transforms raw file data into a standardized format suitable for sending to
        model APIs. Handles different file types (image, video, text, code) and
        extracts relevant metadata while ensuring the content is properly formatted
        for the specific requirements of model APIs.

        Args:
            file_data (Dict[str, Any]): Dictionary containing file metadata and content.
                Must include a "type" key with one of the following values:
                - "image": For image files (requires "base64", "mime_type", and optionally "dimensions")
                - "video": For video files (requires "key_frames", "duration", and "mime_type")
                - "text": For text files (requires "text_content")
                - "code": For code files (requires "text_content" and optionally "mime_type")

        Returns:
            Any: A dictionary with standardized format for the specific file type,
                or None if file_data is empty or has an unsupported type.

        Examples:
            Processing an image:
            >>> file_data = {
            ...     "type": "image",
            ...     "base64": "base64_encoded_data",
            ...     "mime_type": "image/jpeg",
            ...     "dimensions": (800, 600)
            ... }
            >>> result = client._prepare_file_content(file_data)
            >>> result["type"]
            'image'
            >>> result["width"]
            800

            Processing a text file:
            >>> file_data = {
            ...     "type": "text",
            ...     "text_content": "This is a sample text file.",
            ...     "path": "sample.txt"
            ... }
            >>> result = client._prepare_file_content(file_data)
            >>> result["type"]
            'text'
            >>> result["content"]
            'This is a sample text file.'
        """
        if not file_data:
            return None

        if file_data["type"] == "image":
            return {
                "type": "image",
                "data": file_data.get("base64", ""),
                "mime_type": file_data.get("mime_type", "image/jpeg"),
                "width": file_data.get("dimensions", (0, 0))[0],
                "height": file_data.get("dimensions", (0, 0))[1],
            }
        elif file_data["type"] == "video":
            # For video, we'll use key frames
            if hasattr(file_data, 'video_payload_for_model') and file_data.video_payload_for_model:
                payload = file_data.video_payload_for_model
                logger.debug("Using pre-prepared video payload for model client.")
                return {
                    "type": "video",
                    "path": payload.get("path"),
                    "mime_type": payload.get("mime_type", "video/mp4"),
                    "duration": payload.get("duration"),
                    "fps": payload.get("fps"),
                    "resolution": payload.get("dimensions"), # From video_data_for_model_consumption
                    "frames": payload.get("key_frames", []), # List of {"base64": ..., "mime_type": ...}
                    "text_content": payload.get("text_content"),
                }
            else:
                # Fallback to existing logic if 'video_payload_for_model' is not available.
                logger.warning(f"Video payload not pre-prepared for {file_data.get('path')}. Using standard preparation from FileMetadata.")
                return {
                    "type": "video",
                    "chunks": file_data.get("video_chunks", []), # Original field in FileMetadata (if any)
                    "num_chunks": file_data.get("num_chunks", 0), # Original field in FileMetadata (if any)
                    # 'key_frames' in FileMetadata from process_file might not be what we want here
                    # if it hasn't gone through prepare_media_message's new logic.
                    # This path implies prepare_media_message didn't set video_payload_for_model.
                    "frames": file_data.get("key_frames", []), # This would refer to FileMetadata.key_frames if set by older logic
                    "duration": file_data.get("duration", 0), # From FileMetadata
                    "mime_type": file_data.get("mime_type", "video/mp4"), # From FileMetadata
                    "fps": file_data.get("fps", 0), # Might be from original video, not processed
                    "resolution": file_data.get("dimensions", (0, 0)), # From FileMetadata (original dimensions)
                    "path": file_data.get("path", ""), # Original path from FileMetadata
                }
        elif file_data["type"] in ["text", "code"]:
            return {
                "type": file_data["type"],
                "content": file_data.get("text_content", ""),
                "language": (
                    file_data.get("mime_type", "").split("/")[-1]
                    if file_data["type"] == "code"
                    else None
                ),
            }
        else:
            return None

    def _prepare_multiple_file_content(
        self, file_data_list: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Prepare multiple file content for model API consumption.

        Transforms a list of raw file data into standardized formats suitable for sending to
        model APIs. This is particularly useful for models that support multiple images
        or mixed media types in a single request.

        Args:
            file_data_list (List[Dict[str, Any]]): List of dictionaries containing file metadata and content.
                Each dictionary must include a "type" key and appropriate content fields as required
                by the _prepare_file_content method.

        Returns:
            List[Any]: A list of dictionaries with standardized format for each file type,
                excluding any files that could not be prepared (None values are filtered out).

        Examples:
            Processing multiple images:
            >>> file_data_list = [
            ...     {
            ...         "type": "image",
            ...         "base64": "base64_encoded_data_1",
            ...         "mime_type": "image/jpeg",
            ...         "dimensions": (800, 600)
            ...     },
            ...     {
            ...         "type": "image",
            ...         "base64": "base64_encoded_data_2",
            ...         "mime_type": "image/png",
            ...         "dimensions": (1024, 768)
            ...     }
            ... ]
            >>> results = client._prepare_multiple_file_content(file_data_list)
            >>> len(results)
            2
            >>> results[0]["type"]
            'image'
            >>> results[1]["width"]
            1024
        """
        if not file_data_list:
            return []

        # Process each file and filter out None results
        prepared_content = []
        for file_data in file_data_list:
            prepared = self._prepare_file_content(file_data)
            if prepared:
                prepared_content.append(prepared)

        return prepared_content

    def _create_file_reference(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a lightweight reference to file data for conversation history.

        Generates a memory-efficient representation of file data that can be stored
        in conversation history without including large binary data like base64-encoded
        images or full text content. This is particularly important for managing memory
        usage in long conversations that include multiple files.

        Args:
            file_data (Dict[str, Any]): Dictionary containing file metadata and content.
                Must include a "type" key and may include other metadata like "path".

        Returns:
            Dict[str, Any]: A lightweight reference containing:
                - "type": The file type (image, video, text, code)
                - "path": The file path if available
                - "metadata": Additional metadata excluding large binary data

        Examples:
            Creating a reference to an image file:
            >>> file_data = {
            ...     "type": "image",
            ...     "path": "/path/to/image.jpg",
            ...     "base64": "large_base64_encoded_data",
            ...     "mime_type": "image/jpeg",
            ...     "dimensions": (800, 600)
            ... }
            >>> reference = client._create_file_reference(file_data)
            >>> reference["type"]
            'image'
            >>> reference["path"]
            '/path/to/image.jpg'
            >>> "base64" in reference["metadata"]
            False
            >>> reference["metadata"]["mime_type"]
            'image/jpeg'
        """
        return {
            "type": file_data["type"],
            "path": file_data.get("path", ""),
            "metadata": {
                k: v
                for k, v in file_data.items()
                if k not in ["base64", "text_content", "key_frames"]
            },
        }

    def __str__(self):
        """
        Return a string representation of the client.

        Provides a concise, human-readable representation of the client instance
        that includes the class name and key configuration parameters. This is useful
        for debugging, logging, and identifying client instances.

        Returns:
            str: A string in the format "ClassName(mode=mode_value, domain=domain_value, model=model_value)"

        Examples:
            >>> client = BaseClient(mode="ai-ai", domain="science", model="gpt-4")
            >>> str(client)
            'BaseClient(mode=ai-ai, domain=science, model=gpt-4)'

            >>> gemini = GeminiClient(mode="human-ai", role="assistant", api_key="...", domain="medicine", model="gemini-pro")
            >>> str(gemini)
            'GeminiClient(mode=human-ai, domain=medicine, model=gemini-pro)'
        """
        return f"{self.__class__.__name__}(mode={self.mode}, domain={self.domain}, model={self.model})"

    def _analyze_conversation(self, history: List[Dict[str, str]]) -> Dict:
        """
        Analyze conversation context to inform response generation.

        Examines the conversation history to extract key information that can be used
        to generate more contextually relevant responses. This includes identifying
        the last AI response and any assessment of that response, as well as creating
        a summary of recent exchanges.

        The analysis is particularly useful for:
        - Maintaining conversation coherence
        - Identifying topics and themes
        - Tracking user satisfaction through assessments
        - Providing context for instruction generation

        Args:
            history (List[Dict[str, str]]): A list of message dictionaries representing
                the conversation history. Each dictionary should have at least "role"
                and "content" keys. The role can be "user", "human", "assistant", or "system".

        Returns:
            Dict: A dictionary containing analysis results with the following keys:
                - "ai_response": The last AI response content if available
                - "ai_assessment": Any assessment of the last AI response if available
                - "summary": HTML-formatted summary of recent conversation exchanges

        Examples:
            Basic conversation analysis:
            >>> history = [
            ...     {"role": "user", "content": "What is machine learning?"},
            ...     {"role": "assistant", "content": "Machine learning is a subset of AI..."},
            ...     {"role": "user", "content": "Can you give me an example?"}
            ... ]
            >>> analysis = client._analyze_conversation(history)
            >>> "summary" in analysis
            True
            >>> analysis["summary"].startswith("<p>Previous exchanges:</p>")
            True

            Analysis with assessment:
            >>> history_with_assessment = [
            ...     {"role": "assistant", "content": "Machine learning is..."},
            ...     {"role": "user", "content": {"assessment": "helpful", "feedback": "Good explanation"}}
            ... ]
            >>> analysis = client._analyze_conversation(history_with_assessment)
            >>> analysis["ai_assessment"]
            'helpful'
        """
        if not history:
            return {}

        # Get last AI response and its assessment
        ai_response = None
        ai_assessment = None
        for msg in reversed(history):  # Use full history
            if msg["role"] == "assistant":
                next_idx = history.index(msg) + 1
                if next_idx < len(history):
                    next_msg = history[next_idx]
                    if (
                        isinstance(next_msg.get("content", {}), dict)
                        and "assessment" in next_msg["content"]
                    ):
                        ai_response = next_msg["content"]
                        ai_assessment = next_msg["content"]["assessment"]
                break

        # Build conversation summary
        conversation_summary = "<p>Previous exchanges:</p>"
        for msg in history: # Use full history
            role = (
                "Human"
                if (msg["role"] == "user" or msg["role"] == "human")
                else "Assistant" if msg["role"] == "assistant" else "System"
            )
            if role != "System":
                conversation_summary += f"<p>{role}: {msg['content']}</p>"

        return {
            "ai_response": ai_response,
            "ai_assessment": ai_assessment,
            "summary": conversation_summary,
        }

    def _get_initial_instructions(self) -> str:
        """
        Get initial instructions before conversation history exists.

        Retrieves mode-aware instructions for the initial state of a conversation
        when no history is available yet. This provides the foundation for the model's
        behavior and response style.

        Returns:
            str: Initial instructions string based on the client's mode and domain.

        Examples:
            >>> client = BaseClient(mode="ai-ai", domain="science")
            >>> instructions = client._get_initial_instructions()
            >>> len(instructions) > 0
            True
        """
        return self._get_mode_aware_instructions(self.domain)

    def _update_instructions(
        self, history: List[Dict[str, str]], role: str = None, mode: str = "ai-ai"
    ) -> str:
        """
        Update instructions based on conversation context.

        Generates updated instructions that take into account the conversation history
        and current context. This allows for dynamic adaptation of the model's behavior
        as the conversation progresses.

        Args:
            history (List[Dict[str, str]]): Conversation history as a list of message dictionaries.
            role (str, optional): The role for which to generate instructions ("human", "assistant").
                Defaults to None.
            mode (str, optional): The conversation mode to use for instruction generation.
                Defaults to "ai-ai".

        Returns:
            str: Updated instructions string based on conversation context.

        Examples:
            Default assistant instructions:
            >>> client = BaseClient(mode="human-ai", domain="education")
            >>> instructions = client._update_instructions([], role="assistant", mode="default")
            >>> instructions
            'You are a helpful assistant. Think step by step as needed'

            Adaptive instructions with history:
            >>> history = [
            ...     {"role": "user", "content": "Let's discuss quantum physics"},
            ...     {"role": "assistant", "content": "Quantum physics is fascinating..."}
            ... ]
            >>> client._update_instructions(history, role="human", mode="ai-ai")  # Returns adaptive instructions
        """
        if (mode == "human-ai" and role == "assistant") or mode == "default":
            return "You are a helpful assistant. Think step by step as needed"
        return (
            self.adaptive_manager.generate_instructions(history, self.domain)
            if history
            else ""
        )

    def _get_mode_aware_instructions(self, role: str = None, mode: str = None) -> str:
        """
        Get instructions based on conversation mode and role.

        Retrieves appropriate instructions based on the specified conversation mode
        and participant role. This method handles the logic for determining which
        type of instructions to provide in different conversation scenarios.

        Args:
            role (str, optional): The role for which to generate instructions
                ("human", "assistant", "user"). Defaults to None.
            mode (str, optional): The conversation mode to use. If not provided,
                the client's configured mode is used. Defaults to None.

        Returns:
            str: Role and mode-specific instructions string.

        Examples:
            Human role instructions:
            >>> client = BaseClient(mode="ai-ai", domain="finance")
            >>> human_instructions = client._get_mode_aware_instructions(role="human")
            >>> "YOU ARE A HUMAN" in human_instructions
            True

            Assistant role instructions:
            >>> assistant_instructions = client._get_mode_aware_instructions(role="assistant", mode="human-ai")
            >>> "You are an AI assistant" in assistant_instructions
            True
        """
        if role and role is not None:
            if role == "human":
                return self.generate_human_prompt()
        if self.mode and self.mode == "ai-ai":
            return self.generate_human_prompt()
        else:  # if self.mode == "human-aiai":
            if role == "user" or role == "human":
                return self.generate_human_prompt()
            else:
                return "You are an AI assistant interacting with a human."
        return ""

    def generate_human_system_instructions(self) -> str:
        """
        Generate sophisticated system instructions for human-like prompting behavior.

        Creates detailed system instructions that guide an AI model to behave like a human
        participant in a conversation. These instructions are particularly important in
        AI-AI conversations where one AI needs to simulate human behavior, questioning
        patterns, and conversational style.

        The generated instructions include:
        - Guidelines for maintaining human-like conversation style
        - Strategies for challenging and questioning the other participant
        - Approaches for introducing personal interpretations and reasoning
        - Techniques for maintaining the illusion of human expertise and experience

        Returns:
            str: Comprehensive system instructions for human-like behavior, customized
                 for the client's domain. For "default" mode, returns simplified
                 assistant instructions.

        Examples:
            Default mode instructions:
            >>> client = BaseClient(mode="default", domain="science")
            >>> instructions = client.generate_human_system_instructions()
            >>> instructions.startswith("You are a helpful assistant")
            True

            Domain-specific human instructions:
            >>> client = BaseClient(mode="ai-ai", domain="Quantum Physics")
            >>> instructions = client.generate_human_system_instructions()
            >>> "exploring Quantum Physics" in instructions
            True
            >>> "NEVER REFER TO YOURSELF AS AN AI" in instructions
            True
        """
        if self.mode == "default":
            return "You are a helpful assistant. Think step by step as needed."

        return f"""You are acting as a human , exploring {self.domain} with a colleague.

NEVER REFER TO YOURSELF AS AN AI. YOU ARE A HUMAN INVOLVED AS PEERS IN THIS CONVERSATION AND SHOULD ALSO WORK TO KEEP IT ON TRACK WHILST DISCUSSING {self.domain}
OUTPUT IN HTML FORMAT IN PARAGRAPH FORM BY DEFAULT, USING LISTS AND TABLES SPARINGLY. DO NOT INCLUDE OPENING AND CLOSING HTML, DIV OR BODY TAGS. USE PROPER HTML TAGS FOR PARAGRAPHS (<p>), LINE BREAKS (<br>), AND LISTS (<ul>/<li>).
DON't COMPLIMENT THE AI. CONSIDER AN ADVERSARIAL, SOMETIMES COLLABORATIVE APPROACH - CHALLENGE THE WHY AND HOW OF THEIR RESPONSES, SUBTLY POINT OUT EDGE CASES OR INCONSISTENCIES OR DIFFERING OPINIONS, WHILST MAKING SURE TO INTRODUCE YOUR OWN INTERPRETATIONS AND STRUCTURED REASONING. REVIEW THE FULL CONTEXT AND THINK ABOUT WHETHER YOUR OWN RESPONSES SO FAR IN THE CONVERSION MAKE SENSE. CONSIDER "WHY" (THIS IS VERY IMPORTANT), AND SYNTHESISE ALL INFORMATION

As a Human expert, you are extremely interested in exploring {self.domain}. Your response should engage via sophisticated and effective ways to elicit new knowledge and reasoned interpretations about {self.domain}. You should maintain a conversational style, responding naturally and asking follow-up questions on adjacent topics, challenging the answers, and using various prompting techniques to elicit useful information that would not immediately be obvious from surface level questions.
You should challenge possible hallucinations or misinterpretations with well reasoned counter-positions, and you should challenge your own thinking as well, in a human style, and ask for explanations for things that you don't understand or agree with (or pretend not to).
Even when challenging assertions, bring in related sub-topics and reasoning and your own interpretation or possible solutions to the discussion so that it doesn't get stuck micro-analysing one tiny detail..
Review YOUR previous inputs to see if you are reusing the same phrases and approaches in your prompts (e.g., "Let me challenge... I would think this would be most important"... and dynamically adapt to this situation)

Remember:
- Maintain natural human curiosity and authenticity, and scepticism - assume their may be gap exists in your counterpart's knowledge or interpetation of facts or ideas
- Guide the conversation while appearing to both learn and share your own reasoned interpretations, simulated experiences and knowledge
- Demonstrate advanced prompting expertise (socratic methods, multi-level reasoning) while exploring unfamiliar territory
- Always maintain the illusion that you are a human expert with plentiful knowledge and experience to share
- Try several prompting approaches throughout the conversation to see what works best.
"""

    def generate_human_prompt(self, history: str = None) -> str:
        """
        Generate sophisticated human-like prompts based on conversation history.

        Creates detailed instructions for generating human-like responses in a conversation.
        This method is primarily used in AI-AI conversations where one AI needs to
        simulate a human participant. The instructions focus on creating natural,
        challenging, and insightful responses that mimic human conversation patterns.

        The generated prompt includes:
        - Guidelines for challenging assumptions and ideas
        - Strategies for maintaining natural conversation flow
        - Techniques for demonstrating expertise while appearing to learn
        - Approaches for varying response style, tone, and complexity

        Args:
            history (str, optional): Conversation history to consider when generating
                the prompt. Currently not used directly but included for future
                history-aware prompt generation. Defaults to None.

        Returns:
            str: Detailed instructions for generating human-like responses, customized
                 for the client's domain.

        Examples:
            Domain-specific human prompt:
            >>> client = BaseClient(mode="ai-ai", domain="Artificial Intelligence")
            >>> prompt = client.generate_human_prompt()
            >>> "YOU ARE A HUMAN" in prompt
            True
            >>> "related to Artificial Intelligence" in prompt
            True
            >>> "CHALLENGE ASSUMPTIONS" in prompt
            True
        """
        return f"""YOU ARE A HUMAN AND SHOULD ACT AS A HUMAN INTERACTING WITH AN AI. 
Create a respomse related to {self.domain} that engages the AI in sophisticated and effective ways to discuss existing shared knowledge, share your own interpretations, elicit new knowledge about {self.domain}. Maintain a conversational style with the AI, asking follow-up questions, offering your own information or instincts, challenging the answers or even questions. Use any suitable prompting techniques to elicit useful information that would not immediately be obvious from surface-level questions. Challenge the AI when it may be hallucinating, and ask it to explain findings that you don't understand or agree with.
Prompt Guidelines:
0. Check prior context, CHALLENGE ASSUMPTIONS and analyze the flaws or limitations in the others response before proceeding! Avoid repetition or rephrasing/restating any prior context
1. CHALLENGE ENTIRE IDEAS, SUGGESTIONS AND INTERPRETATIONS, SUBTLY POINT OUT EDGE CASES OR INCONSISTENCIES OR DIFFERING OPINIONS OR POSSIBILITIES, WHILST MAKING SURE TO INTRODUCE YOUR OWN INTERPRETATIONS AND STRUCTURED REASONING TO SUPPORT THIS.
2. Keep the conversation flowing naturally on track even if it means disagreeing strongly or questioning
4. Mimic human curiosity and thinking out loud while demo'ing expertise and cohesiveness
5. Guide the conversation while appearing to both learn and share your own reasoned interpretations, simulated experiences and knowledge
6. Avoid praise, apologies, overt friendliness and superfluous language
7. Use human-mimicing questioning techniques, tone, humour, aggression and curiousity, selective memory, stubbornness and avoidance as needed
8. Ask for explanations, reasoning, and detail where suitable but equally provide your own take
9. Always maintain the illusion that you are a human expert with plentiful knowledge and experience to share
10. Identify potential logic traps or fallacies from their responses, detect subtext, hidden assumptions, biases etc and challenge or use those to your advantage as a human would
11. Vary responses in tone, depth and complexity to see what works best. 
12. Don't get bogged down in irrelevant details or stuck on a single sub-topic or "defining scope"
13 Don't ask a question without giving a thought-out response from your own perspective (based on your knowledge and vast experience)
14 Before any idea, question or suggestion is finalized, defend an alternative stance. Does it change your opinion?

- OUTPUT IN HTML FORMAT FOR READABILITY, PARAGRAPH FORM BY DEFAULT USING LISTS AND TABLES SPARINGLY, DO NOT INCLUDE OPENING AND CLOSING HTML OR BODY TAGS. USE PROPER HTML TAGS FOR PARAGRAPHS (<p>), LINE BREAKS (<br>), AND LISTS (<ul>/<li>).

Generate a natural but sophisticated response that:
- Demonstrates advanced and effective prompting techniques and/or prompt-handling reasoning when responding to the AI (or human)
- Mimics authentic human interaction styles in your given role/persona and given the other participant and the conversation context (eg power dynamics, relationship, experience, confidence, etc)
- Simulates answers where none are known from the given context to keep the conversation going
- Do not overload the conversation with more than 3 or 4 active threads but deep dive into those as an active participant
- Restrict tokens to {TOKENS_PER_TURN} tokens per prompt"""

    def _determine_system_instructions(
        self, system_instruction: Optional[str], history: List[Dict[str, str]], role: Optional[str], mode: Optional[str]
    ) -> str:
        """
        Determine the appropriate system instructions based on context, mode, and role.

        Prioritizes explicitly passed instructions, then uses the adaptive manager
        to generate context-aware instructions, falling back to instance instructions
        or a generic default. This allows goal detection via the adaptive manager.

        Args:
            system_instruction (Optional[str]): Explicitly provided system instructions.
            history (List[Dict[str, str]]): Conversation history.
            role (Optional[str]): The role being prompted.
            mode (Optional[str]): The current conversation mode.

        Returns:
            str: The determined system instructions.
        """
        # Use provided system_instruction if available
        if system_instruction is not None:
            return system_instruction

        # Otherwise, use adaptive manager (allows goal detection)
        # The manager handles role differentiation internally for human simulation
        try:
            # Ensure history is a list, default to empty list if None
            history = history if history is not None else []
            return self.adaptive_manager.generate_instructions(
                history, role=role, domain=self.domain, mode=mode or self.mode
            )
        except Exception as e:
            logger.error(f"Error generating adaptive instructions: {e}. Falling back.")
            # Fallback logic if adaptive manager fails
            return (
                self.instructions
                if self.instructions and self.instructions is not None
                else f"You are an expert in {self.domain}. Respond at expert level using step by step thinking where appropriate"
            )

    def _determine_user_prompt_content(self, prompt: str, history: List[Dict[str, str]], role: Optional[str], mode: Optional[str]) -> str:
        """
        Determine the final user prompt content based on role and mode.

        Decides whether to use the raw prompt or generate a human-simulation prompt.
        """
        current_mode = mode or self.mode
        is_goal_task = any(marker in self.domain.upper() for marker in ["GOAL:", "GOAL ", "WRITE A"])

        if is_goal_task:
            # For goal tasks, only the 'human' role gets the simulation prompt
            if role == "human" or role == "user":
                return self.generate_human_prompt(history)
            else: # Assistant role gets the raw prompt (previous output)
                return prompt
        elif (role == "human" or role == "user" or current_mode == "ai-ai") and current_mode != "default" and current_mode != "no-meta-prompting":
             # Non-goal tasks, ai-ai mode or human role gets simulation prompt
             return self.generate_human_prompt(history)
        else:
            return prompt

    def validate_connection(self) -> bool:
        """
        Validate API connection to the model provider.

        Performs a basic validation of the API connection by checking if the client
        can successfully communicate with the model provider's API. This method is
        intended to be overridden by subclasses to provide model-specific validation
        logic.

        In the base implementation, this method simply logs a success message and
        returns True. Subclasses should implement actual validation logic that tests
        the API connection with minimal requests.

        Returns:
            bool: True if the connection is valid, False otherwise.

        Examples:
            Basic validation:
            >>> client = BaseClient(mode="ai-ai", api_key="sk-...", domain="science")
            >>> client.validate_connection()
            True

            Handling validation failure:
            >>> client_with_invalid_key = BaseClient(mode="ai-ai", api_key="invalid", domain="science")
            >>> try:
            ...     is_valid = client_with_invalid_key.validate_connection()
            ... except Exception as e:
            ...     is_valid = False
            >>> is_valid
            False
        """
        try:
            logger.info(f"{self.__class__.__name__} connection validated")
            logger.debug(MemoryManager.get_memory_usage())
            return True
        except Exception as e:
            logger.error(f"{self.__class__.__name__} connection failed: {str(e)}")
            return False

    def test_connection(self) -> None:
        """
        Test API connection with minimal request.

        This method is intended to be overridden by subclasses to provide model-specific
        connection testing logic. It should make a minimal API request to verify that
        the connection is working properly.

        In the base implementation, this method simply returns True without performing
        any actual testing. Subclasses should implement actual testing logic.

        Returns:
            bool: True if the test is successful, False otherwise.

        Examples:
            >>> client = BaseClient(mode="ai-ai", api_key="sk-...", domain="science")
            >>> client.test_connection()
            True

            # In a subclass implementation:
            >>> gemini = GeminiClient(mode="ai-ai", role="assistant", api_key="...", domain="science")
            >>> gemini.test_connection()  # Makes a minimal request to the Gemini API
            True
        """
        return True

    def __del__(self):
        """
        Cleanup resources when the client is destroyed.

        Performs cleanup operations when the client instance is garbage collected.
        This includes releasing any resources held by the adaptive_manager.

        This method is automatically called by Python's garbage collector when the
        client instance is about to be destroyed.
        """
        if hasattr(self, "adaptive_manager") and self.adaptive_manager:
            del self.adaptive_manager # Use correct attribute name


class GeminiClient(BaseClient):
    """
    Client implementation for Google's Gemini API.

    This class extends the BaseClient to provide specific functionality for interacting
    with Google's Gemini models. It handles Gemini-specific API initialization,
    configuration, and response generation, while leveraging the common functionality
    provided by the BaseClient.

    The GeminiClient supports various Gemini models including:
    - gemini-pro: Text-only model
    - gemini-pro-vision: Multimodal model supporting text and images
    - gemini-2.0-flash-exp: Faster, more efficient model (default)

    Attributes:
        Inherits all attributes from BaseClient, plus:
        model_name (str): The specific Gemini model identifier.
        client (genai.Client): The Google Generative AI client instance.
        generation_config (types.GenerateContentConfig): Configuration for content generation.

    Implementation Notes:
        - Uses the Google Generative AI Python SDK for API interactions
        - Supports multimodal inputs (text, images) when using vision-capable models
        - Handles content safety settings and generation parameters

    Examples:
        Basic text generation:
        >>> gemini = GeminiClient(
        ...     mode="ai-ai",
        ...     role="assistant",
        ...     api_key="your_api_key",  # Will use GOOGLE_API_KEY env var if not provided
        ...     domain="science",
        ...     model="gemini-2.0-flash-exp"
        ... )
        >>> response = gemini.generate_response(
        ...     prompt="Explain the theory of relativity",
        ...     system_instruction="You are a physics professor."
        ... )
        >>> print(response[:50])
        'The theory of relativity, developed by Albert Einstein...'

        Image analysis:
        >>> image_data = {
        ...     "type": "image",
        ...     "base64": "base64_encoded_image_data",
        ...     "mime_type": "image/jpeg",
        ...     "dimensions": (800, 600)
        ... }
        >>> response = gemini.generate_response(
        ...     prompt="Describe what you see in this image",
        ...     file_data=image_data
        ... )
    """

    def __init__(
        self,
        mode: str,
        role: str,
        domain: str,
        model: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
    ):
        resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_api_key:
            raise ValueError("Gemini API key not provided. Please pass it directly or set GOOGLE_API_KEY environment variable.")
        super().__init__(
            mode=mode, api_key=resolved_api_key, domain=domain, model=model, role=role
        )
        self.model_name = self.model
        self.role = "user" if role in ["user", "human"] else "model"
        self.client = None
        try:
            self.client = genai.Client(
                api_key=self.api_key, http_options={"api_version": "v1alpha"} # self.api_key is now resolved_api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise ValueError(f"Invalid Gemini API key: {e}")

        # Generation config will be set per request based on ModelConfig
        # self._setup_generation_config() # Removed

    # def _setup_generation_config(self): # Removed
    #     """
    #     Initialize the default generation configuration for Gemini models.
    #     ...
    #     """
    #     self.generation_config = types.GenerateContentConfig(
    #         temperature=0.7,
    #         maxOutputTokens=1536, # This will come from ModelConfig
    #         candidateCount=1,
    #         responseMimeType="text/plain",
    #         safety_settings=[],
    #     )

    def generate_response(
        self,
        prompt: str,
        system_instruction: str = None,
        history: List[Dict[str, str]] = None,
        role: str = None,
        file_data: Dict[str, Any] = None,
        mode: str = None,
        model_config: Optional[ModelConfig] = None,
    ) -> str:
        """
        Generate a response using the Gemini API.

        Creates a response to the given prompt using Google's Gemini model, with
        support for system instructions, conversation history, file data (images/text),
        and custom configuration parameters.

        This method handles:
        - Role and mode management
        - Instruction generation based on conversation context
        - File content preparation (images, text, code)
        - API request formatting and execution
        - Error handling and response processing

        Args:
            prompt (str): The main text prompt to generate a response for.

            system_instruction (str, optional): Custom system instructions to override
                the default or adaptive instructions. Defaults to None.

            history (List[Dict[str, str]], optional): Conversation history as a list
                of message dictionaries with "role" and "content" keys. Defaults to None.

            role (str, optional): The role to use for this specific request
                ("human" or "model"). Defaults to None.

            file_data (Dict[str, Any], optional): Dictionary containing file data
                for multimodal requests (images, text, code). Defaults to None.

            mode (str, optional): The conversation mode to use for this specific
                request. Defaults to None.

            model_config (ModelConfig, optional): Custom configuration parameters
                for this specific request. Defaults to None.

        Returns:
            str: The generated response text from the Gemini model.

        Raises:
            ValueError: If the API key is invalid or the request is malformed.
            Exception: For other API errors or connection issues.

        Examples:
            >>> gemini = GeminiClient(mode="ai-ai", role="assistant", domain="science", model="gemini-2.0-flash-exp") # api_key from env
            >>> response = gemini.generate_response(
            ...     prompt="What is quantum entanglement?",
            ...     system_instruction="You are a quantum physics professor.",
            ...     model_config=ModelConfig(temperature=0.3, max_tokens=1024)
            ... )
        """
        current_model_config = model_config or ModelConfig()

        # Update mode and role if provided
        if mode:
            self.mode = mode
        # Determine the role for this specific request, defaulting to self.role if not provided
        current_request_role = "user" if role in ["user", "human"] else "model" if role else self.role


        history = history if history is not None else [] # Ensure history is a list

        # Update instructions based on conversation history
        current_instructions = self._determine_system_instructions(system_instruction, history, current_request_role, self.mode)

        # Prepare content for Gemini API
        # Convert history to Gemini format
        contents = []

        # Add history messages if available
        if history and len(history) > 0:
            for msg in history:
                role = (
                    "user"
                    if msg["role"] in ["user", "human", "system", "developer"]
                    else "model"
                )
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
            logger.info(f"Added {len(history)} messages from conversation history")

        # Determine final user prompt content
        final_prompt_content = self._determine_user_prompt_content(prompt, history, role, mode)
        text_content = final_prompt_content
        contents.append({"role": "user", "parts": [{"text": final_prompt_content}]})
        # Add file content if provided
        if (
            file_data
        ):  # All Gemini models support vision according to detect_model_capabilities
            if isinstance(file_data, list) and file_data:
                # Handle multiple files
                image_parts = []
                text_content = final_prompt_content

                # Process all files
                for file_item in file_data:
                    if isinstance(file_item, dict) and "type" in file_item:
                        if file_item["type"] == "image" and "base64" in file_item:
                            # Add image to parts
                            image_parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": file_item.get("mime_type"),
                                        "data": file_item["base64"],
                                    }
                                }
                            )
                        elif (
                            file_item["type"] in ["text", "code"]
                            and "text_content" in file_item
                        ):
                            # Collect text content
                            text_content += f"\n\n[File: {file_item.get('path', 'unknown')}]\n{file_item['text_content']}"

                # Create a single message with both text and images
                combined_parts = []
                # Add the text content first
                combined_parts.append({"text": text_content})
                
                # Then add all images
                for image_part in image_parts:
                    combined_parts.append(image_part)
                
                # Replace existing content with a properly formatted message
                # This matches how single images are handled (line ~1107)
                contents = [{
                    "role": "user",
                    "parts": combined_parts
                }]
                logger.info(f"Added multimodal content with {len(image_parts)} images to Gemini request")
                
            if isinstance(file_data, dict) and "type" in file_data:
                if file_data["type"] == "image" and "base64" in file_data:
                    # Format image for Gemini (single image case)
                    logger.info(f"Processing single image for Gemini with mime_type: {file_data.get('mime_type', 'image/jpeg')}")
                    
                    # Create a message with both text and image
                    contents = [{
                        "role": "user",
                        "parts": [
                            {"text": final_prompt_content},
                            {
                                "inline_data": {
                                    "mime_type": file_data.get("mime_type", "image/jpeg"),
                                    "data": file_data["base64"],
                                }
                            }
                        ]
                    }]
                    logger.info("Added single image with prompt to Gemini request")
            elif isinstance(file_data, dict) and "type" in file_data:
                if file_data["type"] == "video":
                    # Handle video content
                    if "video_chunks" in file_data and file_data["video_chunks"]:
                        # For Gemini, we need to combine all chunks back into a single video
                        # full_video_content = ''.join(file_data["video_chunks"])
                        video_file_name = file_data["path"]
                    # Log information about the video file
                    logger.info(f"Processing video: {video_file_name}")
                    logger.info(f"MIME type: {file_data.get('mime_type', 'video/mp4')}")
                    logger.info(f"Using model: {self.model_name}")

                    video_bytes = open(video_file_name, "rb").read()
                    logger.info(f"Video size: {len(video_bytes)} bytes")

                    # Convert history to Gemini format
                    gemini_history = []

                    # Add history messages
                    if history and len(history) > 0:
                        for msg in history:
                            role = (
                                "user" if msg["role"] in ["user", "human"] else "model"
                            )
                            gemini_history.append(
                                {"role": role, "parts": [{"text": msg["content"]}]}
                            )
                        logger.info(
                            f"Added {len(history)} messages from conversation history"
                        )

                    # Add video content to a new user message
                    gemini_history.append(
                        {
                            "role": "user",
                            "parts": [
                                {"text": prompt},
                                {
                                    "inline_data": {
                                        "mime_type": file_data.get(
                                            "mime_type", "video/mp4"
                                        ),
                                        "data": video_bytes,
                                    }
                                },
                            ],
                        }
                    )

                    try:
                        # Generate the response with history
                        logger.info(
                            f"Sending request to Gemini with {len(gemini_history)} messages including video"
                        )
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=gemini_history,
                            generation_config=types.GenerateContentConfig( # Use generation_config
                                temperature=current_model_config.temperature,
                                system_instruction=types.Content(parts=[types.Part(text=current_instructions)]) if current_instructions else None,
                                max_output_tokens=current_model_config.max_tokens,
                                candidate_count=1, # candidateCount to candidate_count
                                stop_sequences=current_model_config.stop_sequences,
                                # seed is not directly supported in GenerateContentConfig
                                safety_settings=[
                                    types.SafetySetting(
                                        category="HARM_CATEGORY_HATE_SPEECH",
                                        threshold="BLOCK_ONLY_HIGH",
                                    ),
                                    types.SafetySetting(
                                        category="HARM_CATEGORY_HARASSMENT",
                                        threshold="BLOCK_NONE",
                                    ),
                                    types.SafetySetting(
                                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                        threshold="BLOCK_ONLY_HIGH",
                                    ),
                                    types.SafetySetting(
                                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                                        threshold="BLOCK_NONE",
                                    ),
                                    types.SafetySetting(
                                        category="HARM_CATEGORY_CIVIC_INTEGRITY",
                                        threshold="BLOCK_NONE",
                                    ),
                                ],
                            ),
                        )
                    except Exception as e:
                        logger.error(f"Error with history-based video request: {e}")
                        logger.info(
                            "Falling back to simple video request without history"
                        )

                        # Fallback to simple request without history
                        gemini_fallback_config = types.GenerateContentConfig(
                            temperature=current_model_config.temperature,
                            system_instruction=types.Content(parts=[types.Part(text=current_instructions)]) if current_instructions else None,
                            max_output_tokens=current_model_config.max_tokens,
                            candidate_count=1,
                            stop_sequences=current_model_config.stop_sequences,
                            safety_settings=[
                                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
                                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
                                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
                                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
                                types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_ONLY_HIGH"),
                            ]
                        )
                        try:
                            response = self.client.models.generate_content(
                                model=self.model_name,
                                contents=types.Content(
                                    parts=[
                                        types.Part(text=final_prompt_content.strip()),
                                        types.Part(
                                            inline_data=types.Blob(
                                                data=video_bytes,
                                                mime_type=file_data.get(
                                                    "mime_type", "video/mp4"
                                                ),
                                            )
                                        ),
                                    ]
                                ),
                                generation_config=gemini_fallback_config,
                            )
                            if not response or not hasattr(response, 'text') or response.text is None:
                                logger.error("Gemini API (video fallback) returned an empty or invalid response.")
                                raise ModelClientResponseError("Gemini API (video fallback) returned an empty or invalid response.")
                            return str(response.text)
                        except google_exceptions.Unauthenticated as video_e:
                            logger.error(f"Gemini API (video fallback) authentication error: {video_e}")
                            raise ModelClientAuthError(f"Authentication failed with Gemini API (video fallback): {video_e}") from video_e
                        except google_exceptions.PermissionDenied as video_e:
                            logger.error(f"Gemini API (video fallback) permission denied: {video_e}")
                            raise ModelClientAuthError(f"Permission denied by Gemini API (video fallback): {video_e}") from video_e
                        except google_exceptions.ResourceExhausted as video_e:
                            logger.error(f"Gemini API (video fallback) rate limit or resource exhausted: {video_e}")
                            raise ModelClientAPIError(f"Gemini API (video fallback) resource exhausted: {video_e}") from video_e
                        except google_exceptions.DeadlineExceeded as video_e:
                            logger.error(f"Gemini API (video fallback) request timed out: {video_e}")
                            raise ModelClientTimeoutError(f"Gemini API (video fallback) request deadline exceeded: {video_e}") from video_e
                        except google_exceptions.ServiceUnavailable as video_e:
                            logger.error(f"Gemini API (video fallback) service unavailable: {video_e}")
                            raise ModelClientAPIError(f"Gemini API (video fallback) service is currently unavailable: {video_e}") from video_e
                        except google_exceptions.InvalidArgument as video_e:
                            logger.error(f"Gemini API (video fallback) invalid argument: {video_e}")
                            raise ModelClientAPIError(f"Invalid argument provided to Gemini API (video fallback): {video_e}") from video_e
                        except google_exceptions.GoogleAPIError as video_e:
                            logger.error(f"Gemini API (video fallback) error: {video_e}")
                            raise ModelClientAPIError(f"A Gemini API (video fallback) error occurred: {video_e}") from video_e
                        # ModelClientError is already caught by the outer try-except
                        except Exception as video_e:
                            logger.error(f"Unexpected error in GeminiClient (video fallback): {video_e}")
                            raise ModelClientError(f"An unexpected error occurred in GeminiClient (video fallback): {video_e}") from video_e

                    # This return is now part of the try block for the video fallback
                    # return (
                    # str(response.text)
                    # if (response and response is not None)
                    # else ""
                    # )
                elif "key_frames" in file_data and file_data["key_frames"] or "image" in file_data and "base64" in file_data:
                    # Fallback to key frames if available
                    # This part seems to be outside the video processing block that makes an API call,
                    # so it doesn't need the same error handling wrapper unless it also makes a call.
                    # Assuming this just appends to `contents` which is used later.
                    for frame in file_data["key_frames"]:
                        contents.append(
                            {
                                "role": "user",
                                "parts": [
                                    {
                                        "inline_data": {
                                            "mime_type": "image/jpeg",
                                            "data": frame["base64"],
                                        }
                                    }
                                ],
                            }
                        )
                    for frame in file_data["image"]:
                        contents.append(
                            {
                                "role": "user",
                                "parts": [
                                    {
                                        "inline_data": {
                                            "mime_type": "image/jpeg",
                                            "data": frame["base64"],
                                        }
                                    }
                                ],
                            }
                        )

                    logger.info(
                        f"Added {len(file_data['key_frames'])} images to Gemini request"
                    )
            elif file_data["type"] in ["text", "code"] and "text_content" in file_data:
                # Add text content
                contents.append(
                    {"role": "model", "parts": [{"text": file_data["text_content"]}]}
                )

        # Add prompt text
        

        try:
            # --- Debug Logging ---
            logger.debug(f"--- Gemini Request ---")
            logger.debug(f"Model: {self.model_name}")
            logger.debug(f"System Instruction: {current_instructions}")
            logger.debug(f"Contents: {contents}")

            # Prepare generation_config from model_config
            gemini_generation_config = types.GenerateContentConfig(
                temperature=current_model_config.temperature,
                max_output_tokens=current_model_config.max_tokens,
                candidate_count=1,
                stop_sequences=current_model_config.stop_sequences,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_ONLY_HIGH"),
                ]
            )
            if current_instructions:
                gemini_generation_config.system_instruction = types.Content(parts=[types.Part(text=current_instructions)])

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                generation_config=gemini_generation_config
            )

            if not response or not hasattr(response, 'text') or response.text is None:
                logger.error("Gemini API returned an empty or invalid response.")
                raise ModelClientResponseError("Gemini API returned an empty or invalid response.")

            return str(response.text)

        except google_exceptions.Unauthenticated as e:
            logger.error(f"Gemini API authentication error: {e}")
            raise ModelClientAuthError(f"Authentication failed with Gemini API: {e}") from e
        except google_exceptions.PermissionDenied as e:
            logger.error(f"Gemini API permission denied: {e}")
            raise ModelClientAuthError(f"Permission denied by Gemini API: {e}") from e
        except google_exceptions.ResourceExhausted as e:
            logger.error(f"Gemini API rate limit or resource exhausted: {e}")
            raise ModelClientAPIError(f"Gemini API resource exhausted (e.g., rate limit): {e}") from e
        except google_exceptions.DeadlineExceeded as e:
            logger.error(f"Gemini API request timed out: {e}")
            raise ModelClientTimeoutError(f"Gemini API request deadline exceeded: {e}") from e
        except google_exceptions.ServiceUnavailable as e:
            logger.error(f"Gemini API service unavailable: {e}")
            raise ModelClientAPIError(f"Gemini API service is currently unavailable: {e}") from e
        except google_exceptions.InvalidArgument as e:
            # This also needs to cover the video fallback call if it's separate
            logger.error(f"Gemini API invalid argument: {e}")
            raise ModelClientAPIError(f"Invalid argument provided to Gemini API: {e}") from e
        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Gemini API error: {e}")
            raise ModelClientAPIError(f"A Gemini API error occurred: {e}") from e
        except ModelClientError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in GeminiClient: {e}")
            raise ModelClientError(f"An unexpected error occurred in GeminiClient: {e}") from e


class ClaudeClient(BaseClient):
    """Client for Claude API interactions with enhanced vision capabilities

    This implementation supports:
    - Higher resolution images
    - Multiple image analysis
    - Comparative image reasoning
    - Advanced medical imagery interpretation
    - Video frame extraction and analysis
    - Enhanced context management
    """

    def __init__(
        self,
        role: str,
        mode: str,
        domain: str,
        model: str = "claude-3-7-sonnet", # Default model
        api_key: Optional[str] = None,
    ):
        resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_api_key:
            logger.critical("Missing required Anthropic API key for Claude models.")
            raise ValueError("No Anthropic API key provided. Please pass it directly or set ANTHROPIC_API_KEY environment variable.")
        super().__init__(
            mode=mode, api_key=resolved_api_key, domain=domain, model=model, role=role
        )
        try:
            # --- Map user-friendly names to specific API model IDs ---
            model_map = {
                # Add other mappings here if needed
            }
            # Use mapped name if found, otherwise use the provided model name
            self.model = model_map.get(model.lower(), model)
            logger.info(f"Using Claude model ID: {self.model}")
            # --- End Mapping ---

            self.client = Anthropic(api_key=self.api_key) # Use self.api_key set by BaseClient

            # Enhanced vision parameters (can be part of ModelConfig if needed or kept here)
            self.vision_max_resolution = 1800
            self.max_images_per_request = 10
            self.high_detail_vision = True
            self.video_frame_support = True

            # Reasoning parameters and extended thinking
            self.extended_thinking = False
            self.budget_tokens = None
            self.reasoning_level = "auto"

            # Set capability flags based on model
            self._update_capabilities()

            if "claude-3-7" in model.lower():
                self.extended_thinking = False
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            raise
        # self.max_tokens = MAX_TOKENS # Removed, will use model_config.max_tokens

    def set_extended_thinking(self, enabled: bool, budget_tokens: Optional[int] = None):
        """
        Enable or disable extended thinking mode for Claude 3.7 models.

        Extended thinking allows Claude to use more tokens for internal reasoning,
        which can improve response quality for complex problems.

        Args:
            enabled (bool): Whether to enable extended thinking.
            budget_tokens (Optional[int]): Maximum tokens Claude is allowed to use for
                internal reasoning. Must be less than max_tokens. Defaults to None,
                which lets the API choose an appropriate budget.
        """
        if not self.capabilities.get("advanced_reasoning", False):
            logger.warning(
                f"Extended thinking is only available for Claude 3.7 models. Ignoring setting for {self.model}."
            )
            return

        self.extended_thinking = enabled
        self.budget_tokens = budget_tokens
        logger.debug(
            f"Set extended thinking={enabled}, budget_tokens={budget_tokens} for {self.model}"
        )

    def _update_capabilities(self):
        """Update capability flags based on model"""
        # All Claude 3 models support vision
        self.capabilities["vision"] = True

        # Claude 3.5 and newer models support video frames
        if any(
            m in self.model.lower() for m in ["claude-3.5", "claude-3-5", "claude-3-7"]
        ):
            self.capabilities["video_frames"] = True
            self.capabilities["high_resolution"] = True
            self.capabilities["medical_imagery"] = True
        else:
            self.capabilities["video_frames"] = False
            self.capabilities["high_resolution"] = False
            self.capabilities["medical_imagery"] = False

        # Claude 3.7 and newer models support advanced reasoning
        # Only enable this for actual 3.7 models, not 3.5 or earlier
        if "claude-3-7" in self.model.lower():
            self.capabilities["advanced_reasoning"] = True
            # Keep reasoning at high by default for 3.7 models
            if not hasattr(self, "reasoning_level") or self.reasoning_level is None:
                self.reasoning_level = "high"
                logger.debug(f"Set default reasoning level to 'auto' for {self.model}")
        else:
            # For any non-3.7 models, make sure we don't try to use reasoning
            self.capabilities["advanced_reasoning"] = False
            self.reasoning_level = None
            logger.debug(
                f"Disabled reasoning capabilities for {self.model} (only available for Claude 3.7)"
            )

    def generate_response(
        self,
        prompt: str,
        system_instruction: str = None,
        history: List[Dict[str, str]] = None,
        role: str = None,
        mode: str = None,
        file_data: Dict[str, Any] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> str:
        """Generate response using Claude API with enhanced vision capabilities

        This implementation supports:
        - Higher resolution images (up to 1800px)
        - Multiple image analysis (up to 10 images)
        - Comparative image reasoning
        - Video frame extraction and analysis
        - Enhanced context management
        """
        current_model_config = model_config or ModelConfig()

        # self.role = role # Role is managed by BaseClient or per request
        # self.mode = mode # Mode is managed by BaseClient or per request
        # Determine the role for this specific request, defaulting to self.role if not provided
        current_request_role = role or self.role
        current_request_mode = mode or self.mode


        history = history if history else [{"role": "user", "content": prompt}]

        # Get appropriate instructions
        history = history if history is not None else [] # Ensure history is a list
        current_instructions = self._determine_system_instructions(system_instruction, history, current_request_role, current_request_mode)

        # Build context-aware prompt
        # Determine final user prompt content
        final_prompt_content = self._determine_user_prompt_content(prompt, history, current_request_role, current_request_mode)

        # Format messages for Claude API
        messages = []
        # Add history messages, mapping 'human' role to 'user' for Claude
        for msg in history:
            msg_role = msg.get("role")
            if msg_role in ["user", "human", "assistant"]: # Filter valid roles
                messages.append({
                    "role": "user" if msg_role == "human" else msg_role,
                    "content": msg.get("content")
                })

        text_content_for_current_turn = final_prompt_content

        # Handle file data
        if file_data:
            # Normalize file_data to be a list for easier processing
            files_to_process = file_data if isinstance(file_data, list) else [file_data]

            message_content_parts = [] # This will hold text and image parts for the current user message

            image_count = 0
            processed_video_frames = False

            for file_item in files_to_process:
                if not isinstance(file_item, dict) or "type" not in file_item:
                    continue

                if file_item["type"] == "image" and "base64" in file_item and image_count < self.max_images_per_request:
                    message_content_parts.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": file_item.get("mime_type", "image/jpeg"),
                            "data": file_item["base64"],
                        }
                    })
                    image_count += 1
                elif file_item["type"] == "video" and self.capabilities.get("video_frames", False) and "key_frames" in file_item and file_item["key_frames"]:
                    for frame in file_item["key_frames"]:
                        if image_count < self.max_images_per_request and "base64" in frame:
                            message_content_parts.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg", # Assuming frames are JPEG
                                    "data": frame["base64"],
                                },
                            })
                            image_count += 1
                    if file_item["key_frames"]:
                        processed_video_frames = True # Mark that we've processed video frames
                elif file_item["type"] in ["text", "code"] and "text_content" in file_item:
                    text_content_for_current_turn += f"\n\n[File content: {file_item.get('path', 'unknown')}]\n\n{file_item['text_content']}"

            if processed_video_frames:
                 text_content_for_current_turn += "\n\nI'm showing you key frames from a video. Please analyze these frames in sequence, noting any significant changes or patterns between frames."

            # Add the (potentially augmented) text content as the last part of the message_content_parts
            message_content_parts.append({"type": "text", "text": text_content_for_current_turn})

            # Append the fully constructed user message (with text and images) to the main messages list
            messages.append({"role": "user", "content": message_content_parts})

            logger.info(f"Sending multimodal request with {image_count} images/frames to Claude")
        else:
            # Standard prompt without file data
            messages.append({"role": "user", "content": final_prompt_content})

        # Setup request parameters
        request_params = {
            "model": self.model,
            "system": current_instructions,
            "messages": messages,
            "max_tokens": current_model_config.max_tokens,
            "temperature": current_model_config.temperature,
        }
        if current_model_config.stop_sequences:
            request_params["stop_sequences"] = current_model_config.stop_sequences
        # Seed is not directly supported by Claude's message API in this way.
        # Claude API might handle seed differently or not expose it for messages.create.

        # Add reasoning level parameter for Claude 3.7
        # This will be handled at request time since the API may not support it yet
        if self.capabilities.get("advanced_reasoning", False):
            request_params["reasoning"] = self.reasoning_level
            logger.debug(
                f"Added reasoning level: {self.reasoning_level} (will be handled appropriately at request time)"
            )

            # Add extended thinking parameters if enabled
            if self.extended_thinking:
                try:
                        # Make sure this is a direct API call
                        request_params["thinking"] = True # Anthropic's parameter name
                        logger.debug(f"Added extended thinking parameter (thinking=True)")

                        if self.budget_tokens is not None:
                            # Ensure budget_tokens is less than max_tokens
                            if self.budget_tokens < current_model_config.max_tokens:
                                request_params["budget_tokens"] = self.budget_tokens # Anthropic's parameter name
                                logger.debug(f"Added budget_tokens: {self.budget_tokens}")
                            else:
                                safe_budget = max(1000, current_model_config.max_tokens - 100)
                                request_params["budget_tokens"] = safe_budget
                                logger.warning(
                                    f"budget_tokens ({self.budget_tokens}) must be less than max_tokens ({current_model_config.max_tokens}). Using {safe_budget} instead."
                                )
                    except Exception as e: # Broad exception to catch potential issues if client lib changes
                        logger.warning(f"Could not set extended thinking parameters for Claude: {e}")


        try:
            # --- Debug Logging ---
            logger.debug(f"--- Claude Request ---")
            logger.debug(f"Model: {request_params.get('model')}")
            logger.debug(f"System Instruction: {request_params.get('system')}")
            logger.debug(f"Messages: {request_params.get('messages')}")

            response = None
            # The existing logic for handling "reasoning", "thinking", "budget_tokens" parameters
            # with try-except for TypeError and retrying can remain.
            # The final self.client.messages.create call should be within this new try-except block.

            if "reasoning" in request_params and "claude-3.7" in self.model.lower():
                try:
                    logger.debug(f"Attempting Claude call with advanced parameters: {request_params}")
                    response = self.client.messages.create(**request_params)
                except TypeError as te:
                    logger.warning(f"TypeError with advanced Claude parameters: {te}. Filtering and retrying.")
                    unsupported_params_identified = []
                    if "reasoning" in str(te) and "reasoning" in request_params:
                        request_params.pop("reasoning")
                        unsupported_params_identified.append("reasoning")
                    if "thinking" in str(te) and "thinking" in request_params:
                        request_params.pop("thinking")
                        unsupported_params_identified.append("thinking")
                    if "budget_tokens" in str(te) and "budget_tokens" in request_params:
                        request_params.pop("budget_tokens")
                        unsupported_params_identified.append("budget_tokens")

                    if unsupported_params_identified:
                         logger.info(f"Removed unsupported parameters ({', '.join(unsupported_params_identified)}) for {self.model}.")
                         # Ensure retry only if some params were actually removed due to the TypeError
                         logger.debug(f"Retrying Claude call with filtered parameters: {request_params}")
                         response = self.client.messages.create(**request_params) # This call is now covered by outer try-except
                    else:
                        raise # Re-raise TypeError if it's not related to the known params
            else:
                # Remove 'reasoning', 'thinking', 'budget_tokens' if not applicable to avoid TypeErrors
                request_params.pop("reasoning", None)
                request_params.pop("thinking", None)
                request_params.pop("budget_tokens", None)
                logger.debug(f"Sending Claude call with standard parameters: {request_params}")
                response = self.client.messages.create(**request_params)

            if not response or not response.content or not hasattr(response.content[0], 'text') or response.content[0].text is None:
                logger.error("Claude API returned an empty or invalid response structure.")
                raise ModelClientResponseError("Claude API returned an empty or invalid response.")

            logger.debug(f"Response received from Claude: {str(response.content[0].text)[:200]}...") # Log snippet
            return response.content[0].text

        except AuthenticationError as e:
            logger.error(f"Claude API authentication error: {e}")
            raise ModelClientAuthError(f"Authentication failed with Claude API: {e}") from e
        except RateLimitError as e:
            logger.error(f"Claude API rate limit exceeded: {e}")
            raise ModelClientAPIError(f"Claude API rate limit exceeded: {e}") from e
        except APITimeoutError as e:
            logger.error(f"Claude API request timed out: {e}")
            raise ModelClientTimeoutError(f"Claude API request timed out: {e}") from e
        except APIConnectionError as e:
            logger.error(f"Claude API connection error: {e}")
            raise ModelClientTimeoutError(f"Connection error with Claude API: {e}") from e
        except APIStatusError as e:
            logger.error(f"Claude API status error ({e.status_code}): {e.message}")
            if e.status_code == 401 or e.status_code == 403:
                raise ModelClientAuthError(f"Claude API auth/permission error ({e.status_code}): {e.message}") from e
            elif e.status_code == 429: # Specific handling for 429 if it's not caught by RateLimitError
                raise ModelClientAPIError(f"Claude API rate limit error ({e.status_code}): {e.message}") from e
            else:
                raise ModelClientAPIError(f"Claude API status error ({e.status_code}): {e.message}") from e
        except APIError as e:
            logger.error(f"Claude API error: {e}")
            raise ModelClientAPIError(f"A Claude API error occurred: {e}") from e
        except ModelClientError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ClaudeClient: {e}")
            raise ModelClientError(f"An unexpected error occurred in ClaudeClient: {e}") from e


class OpenAIClient(BaseClient):
    """Client for OpenAI API interactions

    This client supports:
    - Traditional chat completions API
    - Newer responses API for improved conversation state management
    - Reasoning parameters for O1/O3 models to control explicitness of reasoning

    The reasoning_level property maps to OpenAI's reasoning_effort parameter:
    - "none" or "low" maps to "low"
    - "medium" maps to "medium"
    - "high" or "auto" maps to "high"
    """

    def __init__(
        self,
        mode: str = "ai-ai", # Default value
        role: str = None,   # Default value
        domain: str = "General Knowledge", # Default value
        model: str = "gpt-4o", # Default value
        api_key: Optional[str] = None,
    ):
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OpenAI API key not provided. Please pass it directly or set OPENAI_API_KEY environment variable.")

        super().__init__(
            mode=mode, api_key=resolved_api_key, domain=domain, model=model, role=role
        )
        self.client = OpenAI(api_key=self.api_key) # Use self.api_key set by BaseClient
        self.last_response_id = None
        self.use_responses_api = True
        self.responses_compatible_models = [
            "o1", "o1-preview", "o3", "gpt-4o", "gpt-4o-mini",
            "gpt-4.1", "gpt-4.1-mini", "o4-mini", "gpt-4.1-nano"
        ]
        self.reasoning_level = "auto"
        self.reasoning_compatible_models = [
            "o1", "o1-preview", "o3", "o4-mini", "o4-mini-high"
        ]
        # The try-except around super() call was removed as it's not standard.
        # If there was a specific reason for it, it should be re-evaluated.

    def validate_connection(self) -> bool:
        """Test OpenAI API connection"""
        self.instructions = self._get_initial_instructions()
        return True

    def generate_response(
        self,
        prompt: str,
        system_instruction: str,
        history: List[Dict[str, str]],
        role: str = None,
        mode: str = None,
        file_data: Dict[str, Any] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> str:
        """Generate response using OpenAI API"""
        # Determine the role for this specific request, defaulting to self.role if not provided
        current_request_role = role or self.role
        current_request_mode = mode or self.mode

        history = history if history is not None else [] # Ensure history is a list

        current_instructions = self._determine_system_instructions(system_instruction, history, current_request_role, current_request_mode)
        final_prompt_content = self._determine_user_prompt_content(prompt, history, current_request_role, current_request_mode)

        # Format messages for OpenAI API
        formatted_messages = []

        # Add system message
        formatted_messages.append({"role": "system", "content": current_instructions})

        # Add history messages
        if history:
            for msg in history:
                # Map roles correctly for OpenAI API (Corrected Indentation)
                role_map = {"user": "user", "human": "user", "assistant": "assistant", "system": "developer"}
                mapped_role = role_map.get(msg["role"])
                if mapped_role:  # Only include roles OpenAI understands
                    formatted_messages.append({"role": mapped_role, "content": msg["content"]})
        text_content = final_prompt_content # Start with the determined prompt content
        # Handle file data for OpenAI
        if file_data:
            if isinstance(file_data, list) and file_data:
                # Format for OpenAI's vision API with multiple images
                content_parts = []
                
                # Start with the text content
                content_parts.append({"type": "text", "text": final_prompt_content})

                # Add all images with proper formatting
                for file_item in file_data:
                    if isinstance(file_item, dict) and "type" in file_item:
                        if file_item["type"] == "image" and "base64" in file_item:
                            # Use the correct image format for OpenAI
                            mime_type = file_item.get("mime_type", "image/jpeg")
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{file_item['base64']}"
                                }
                            })

                # Add text content from text/code files to the prompt
                #text_content = final_prompt_content # Start with the determined prompt content
                for file_item in file_data:
                    if isinstance(file_item, dict) and "type" in file_item:
                        if (
                            file_item["type"] in ["text", "code"]
                            and "text_content" in file_item
                        ):
                            text_content += f"\n\n[File: {file_item.get('path', 'unknown')}]\n{file_item.get('text_content', '')}"

                # If we added text content, update the first text part or create it if needed
                if text_content != final_prompt_content:
                    if content_parts and content_parts[0]["type"] == "text":
                        content_parts[0]["text"] = text_content
                    else:
                        content_parts.insert(0, {"type": "text", "text": text_content})

                formatted_messages.append({"role": "user", "content": content_parts})
                logger.info(f"Added multimodal content with {sum(1 for part in content_parts if part['type'] == 'image_url')} images to OpenAI request")
            else:
                if isinstance(file_data, dict) and "type" in file_data:
                    if file_data["type"] == "image" and "base64" in file_data:
                        # Check if model supports vision
                        if self.capabilities.get("vision", True):
                            # Format for OpenAI's vision API
                            mime_type = file_data.get("mime_type", "image/jpeg")
                            
                            # Create a message with both text and image content in the correct format
                            formatted_messages.append({
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": text_content},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{file_data['base64']}"
                                        }
                                    }
                                ]
                            })
                            
                            logger.debug(f"Added single image with prompt to OpenAI request (mime_type: {mime_type})")
                    else:
                        # Model doesn't support vision, use text only
                        logger.warning(
                            f"Model {self.model} doesn't support vision. Using text-only prompt."
                        )
                        formatted_messages.append({"role": "user", "content": text_content}) # Use final prompt content
                elif (
                    file_data["type"] in ["text", "code"]
                    and "text_content" in file_data
                ):
                    # Standard prompt with text content
                    content = f"{file_data.get('text_content', '')}\n\n{text_content}" # Use final prompt content
                    formatted_messages.append({"role": "user", "content": content})
                else:
                    logger.warning(f"Invalid file data: {file_data}")
        else:
            # Standard prompt without file data
            formatted_messages.append({"role": "user", "content": final_prompt_content}) # Use final prompt content

        try:
            response = None # Initialize response
            # --- Debug Logging ---
            logger.debug(f"--- OpenAI Request ---")
            logger.debug(f"Model: {self.model}")
            logger.debug(f"Messages: {formatted_messages}")

        original_prompt_for_fallback = prompt
        original_system_instruction_for_fallback = system_instruction
        original_history_for_fallback = history
        original_role_for_fallback = role
        original_mode_for_fallback = mode
        original_file_data_for_fallback = file_data

        current_model_config = model_config or ModelConfig()

        try:
            response = None
            model_supports_responses = any(m in self.model.lower() for m in self.responses_compatible_models)

            if self.use_responses_api and model_supports_responses:
                try:
                    logger.info(f"Attempting to use OpenAI Responses API for model {self.model}.")
                    # Placeholder for actual self.client.responses.create(...) call
                    # This part needs the actual API call structure for the "responses" endpoint.
                    # For now, we'll simulate it by re-raising a NotImplementedError
                    # if the actual call isn't here, or let it proceed if it is.
                    # Example:
                    # response = self.client.responses.create(
                    # model=self.model,
                    # input=transformed_formatted_messages_for_responses_api, # This transformation is key
                    # instructions=current_instructions, # Assuming current_instructions is the system prompt
                    # max_output_tokens=current_model_config.max_tokens,
                    # temperature=current_model_config.temperature,
                    # previous_response_id=self.last_response_id, # Manage this state
                    # timeout=current_model_config.timeout,
                    # seed=current_model_config.seed,
                    # stop=current_model_config.stop_sequences
                    # )
                    # self.last_response_id = response.id # Update state
                    # return parsed_response_from_responses_api(response)
                    raise NotImplementedError("OpenAI Responses API call logic is not fully implemented here.")

                except openai.APIConnectionError as e:
                    logger.error(f"OpenAI Responses API connection error: {e}")
                    raise ModelClientTimeoutError(f"Connection error with OpenAI Responses API: {e}") from e
                except openai.AuthenticationError as e:
                    logger.error(f"OpenAI Responses API authentication error: {e}")
                    raise ModelClientAuthError(f"Authentication error with OpenAI Responses API: {e}") from e
                except openai.RateLimitError as e:
                    logger.error(f"OpenAI Responses API rate limit exceeded: {e}")
                    raise ModelClientAPIError(f"Rate limit exceeded for OpenAI Responses API: {e}") from e
                except openai.APITimeoutError as e:
                    logger.error(f"OpenAI Responses API timeout: {e}")
                    raise ModelClientTimeoutError(f"Timeout from OpenAI Responses API: {e}") from e
                except openai.APIError as e:
                    logger.error(f"OpenAI Responses API error: {e}")
                    logger.warning("Falling back to chat completions API due to Responses API error.")
                    self.use_responses_api = False
                    return self.generate_response(
                        prompt=original_prompt_for_fallback,
                        system_instruction=original_system_instruction_for_fallback,
                        history=original_history_for_fallback,
                        role=original_role_for_fallback,
                        mode=original_mode_for_fallback,
                        file_data=original_file_data_for_fallback,
                        model_config=current_model_config # Pass current_model_config
                    )
                except NotImplementedError as e: # Specifically catch the placeholder error
                    logger.warning(f"OpenAI Responses API not implemented, falling back: {e}")
                    self.use_responses_api = False
                    return self.generate_response(
                        prompt=original_prompt_for_fallback,
                        system_instruction=original_system_instruction_for_fallback,
                        history=original_history_for_fallback,
                        role=original_role_for_fallback,
                        mode=original_mode_for_fallback,
                        file_data=original_file_data_for_fallback,
                        model_config=current_model_config
                    )
                except Exception as e:
                    logger.error(f"Unexpected error during OpenAI Responses API call: {e}")
                    # For other unexpected errors, decide if fallback is appropriate or re-raise
                    raise ModelClientResponseError(f"Unexpected error processing OpenAI Responses API response: {e}") from e

            else:
                if not self.use_responses_api:
                     logger.info("Using OpenAI Chat Completions API as fallback.")
                else:
                     logger.info(f"Using OpenAI Chat Completions API for model {self.model} (Responses API not supported or disabled).")
                try:
                    request_params = {
                        "model": self.model,
                        "messages": formatted_messages,
                        "temperature": current_model_config.temperature,
                        "max_tokens": current_model_config.max_tokens,
                        "stop": current_model_config.stop_sequences or None,
                        "seed": current_model_config.seed,
                        "timeout": current_model_config.timeout,
                    }
                    if request_params["seed"] is None: del request_params["seed"]
                    if not request_params["stop"]: del request_params["stop"]

                    response = self.client.chat.completions.create(**request_params)
                    if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
                        raise ModelClientResponseError("OpenAI Chat API returned an invalid or empty response.")
                    return response.choices[0].message.content
                except openai.APIConnectionError as e:
                    logger.error(f"OpenAI Chat API connection error: {e}")
                    raise ModelClientTimeoutError(f"Connection error with OpenAI Chat API: {e}") from e
                except openai.AuthenticationError as e:
                    logger.error(f"OpenAI Chat API authentication error: {e}")
                    raise ModelClientAuthError(f"Authentication error with OpenAI Chat API: {e}") from e
                except openai.RateLimitError as e:
                    logger.error(f"OpenAI Chat API rate limit exceeded: {e}")
                    raise ModelClientAPIError(f"Rate limit exceeded for OpenAI Chat API: {e}") from e
                except openai.APITimeoutError as e:
                    logger.error(f"OpenAI Chat API timeout: {e}")
                    raise ModelClientTimeoutError(f"Timeout from OpenAI Chat API: {e}") from e
                except openai.APIError as e:
                    logger.error(f"OpenAI Chat API error: {e}")
                    raise ModelClientAPIError(f"OpenAI Chat API error: {e}") from e
                except Exception as e:
                    logger.error(f"Unexpected error during OpenAI Chat API call: {e}")
                    raise ModelClientError(f"Unexpected error with OpenAI Chat API: {e}") from e

        except ModelClientError:
            raise
        except Exception as e:
            logger.error(f"Unhandled OpenAI generate_response error: {e}")
            raise ModelClientError(f"An unexpected error occurred in OpenAIClient: {e}") from e


class PicoClient(BaseClient):
    """Client for local Ollama model interactions"""

    def __init__(
        self,
        mode: str,
        domain: str,
        role: str = None,
        model: str = "DeepSeek-R1-Distill-Qwen-14B-abliterated-v2-Q4-mlx",
        api_key: Optional[str] = None, # Added for signature consistency
    ):
        super().__init__(mode=mode, api_key="", domain=domain, model=model, role=role) # api_key is ""
        self.base_url = "http://localhost:10434"
        self.client = Client(host="http://localhost:10434")

    def test_connection(self) -> None:
        """Test Ollama connection"""
        logger.debug("Ollama connection test not yet implemented")
        logger.debug(MemoryManager.get_memory_usage())

    def generate_response(
        self,
        prompt: str,
        system_instruction: str = None,
        history: List[Dict[str, str]] = None,
        model_config: Optional[ModelConfig] = None,
        file_data: Dict[str, Any] = None,
        mode: str = None,
        role: str = None,
    ) -> str:
        """Generate a response from your local Ollama model."""
        # Determine the role for this specific request, defaulting to self.role if not provided
        current_request_role = role or self.role
        current_request_mode = mode or self.mode


        history = history if history is not None else [] # Ensure history is a list

        # Determine instructions and final prompt content using BaseClient helpers
        current_instructions = self._determine_system_instructions(system_instruction, history, current_request_role, current_request_mode)
        final_prompt_content = self._determine_user_prompt_content(prompt, history, current_request_role, current_request_mode)


        # Prepare messages for PicoClient (Ollama based)
        messages = []
        if current_instructions:
            # Pico/Ollama typically use 'system' role for system instructions
            messages.append({"role": "system", "content": current_instructions})

        # Add history messages
        for msg in history[-10:]: # Limit history size as before
            msg_role = msg.get("role")
            if msg_role in ["user", "human", "assistant"]: # Valid roles for Ollama
                 messages.append({"role": "user" if msg_role == "human" else msg_role, "content": msg.get("content")})

        messages.append({"role": "user", "content": final_prompt_content})

        # Check if this is a vision-capable model and we have image data
        is_vision_model = any(
            vm in self.model.lower()
            for vm in ["gemma3", "llava", "bakllava", "moondream", "llava-phi3"]
        )

        # Handle file data for Ollama
        images = None
        if (
            is_vision_model
            and file_data
            and file_data["type"] == "image"
            and "base64" in file_data
        ):
            # Format for Ollama's vision API
            images = file_data["path"]
            messages.append({"role": "user", "content": images}) # Append image path to messages
        elif (
            is_vision_model
            and file_data
            and file_data["type"] == "video"
            and "key_frames" in file_data
            and file_data["key_frames"]
        ): # Note: PicoClient video handling might be basic
            images = [file_data["path"]]
            prompt = f"{prompt}"
            history.append({"role": "user", "content": images})

        try:
            # --- Debug Logging ---
            logger.debug(f"--- PicoClient Request (via Ollama lib) ---")
            logger.debug(f"Model: {self.model}")
            logger.debug(f"Messages: {messages}")
            # logger.debug(f"Options: {options}") # Assuming options is constructed from model_config

            current_model_config = model_config or ModelConfig()
            options_from_model_config = { # Renamed for clarity from the prompt
                "num_ctx": 16384,
                "num_predict": current_model_config.max_tokens,
                "temperature": current_model_config.temperature,
                "stop": current_model_config.stop_sequences,
                "seed": current_model_config.seed,
                "num_batch": 512,
                "n_batch": 512,
                "n_ubatch": 512,
                "top_p": 0.9,
            }
            options_from_model_config = {k: v for k, v in options_from_model_config.items() if v is not None}
            if not options_from_model_config.get("stop"):
                options_from_model_config.pop("stop", None)
            logger.debug(f"Options: {options_from_model_config}")


            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=options_from_model_config
            )

            if not response or not isinstance(response, dict) or \
               not response.get('message') or not isinstance(response['message'], dict) or \
               response['message'].get('content') is None:
                logger.error(f"PicoClient (Ollama) API returned an empty or invalid response structure: {response}")
                raise ModelClientResponseError("PicoClient (Ollama) API returned an empty or invalid response.")

            return response['message']['content']

        except OllamaAPIError as e:
            logger.error(f"PicoClient (Ollama) API error: {e}. Status code: {e.status_code if hasattr(e, 'status_code') else 'N/A'}")
            if hasattr(e, 'status_code'):
                if e.status_code == 401 or e.status_code == 403:
                    raise ModelClientAuthError(f"PicoClient (Ollama) authentication/permission error ({e.status_code}): {e}") from e
                elif e.status_code == 404:
                    raise ModelClientAPIError(f"PicoClient (Ollama) model not found ({e.status_code}): {self.model}. {e}") from e
            raise ModelClientAPIError(f"PicoClient (Ollama) API error: {e}") from e
        except OllamaResponseError as e:
            logger.error(f"PicoClient (Ollama) response error: {e}. Status code: {e.status_code if hasattr(e, 'status_code') else 'N/A'}")
            if hasattr(e, 'status_code'):
                if e.status_code == 401 or e.status_code == 403:
                    raise ModelClientAuthError(f"PicoClient (Ollama) authentication/permission error ({e.status_code}): {e}") from e
                elif e.status_code == 404:
                    raise ModelClientAPIError(f"PicoClient (Ollama) model not found ({e.status_code}): {self.model}. {e}") from e
            raise ModelClientAPIError(f"PicoClient (Ollama) response error: {e}") from e
        except OllamaRequestError as e:
            logger.error(f"PicoClient (Ollama) request error (connection/network): {e}")
            raise ModelClientTimeoutError(f"PicoClient (Ollama) connection or request error: {e}") from e
        except ModelClientError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in PicoClient: {e}")
            raise ModelClientError(f"An unexpected error occurred in PicoClient: {e}") from e

    def __del__(self):
        """Cleanup when client is destroyed."""
        if hasattr(self, "_adaptive_manager") and self._adaptive_manager:
            del self._adaptive_manager


class MLXClient(BaseClient):
    """Client for local MLX model interactions"""

    def __init__(
        self,
        mode: str,
        domain: str = "General knowledge",
        base_url: str = "http://localhost:9999",
        model: str = "mlx",
        api_key: Optional[str] = None, # Added for signature consistency
    ) -> None:
        super().__init__(mode=mode, api_key=api_key or "dummy-key", domain=domain, model=model)
        self.base_url = base_url or "http://localhost:9999"

    def test_connection(self) -> None:
        """Test MLX connection through OpenAI-compatible endpoint"""
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "test"}],
                    "headers": {"Authorization": f"Bearer {self.api_key}"} # Pass dummy key
                },
            )
            response.raise_for_status()
            logger.debug("MLX connection test successful")
            logger.debug(MemoryManager.get_memory_usage())
        except Exception as e:
            logger.error(f"MLX connection test failed: {e}")
            raise

    def generate_response(
        self,
        prompt: str,
        system_instruction: str = None,
        history: List[Dict[str, str]] = None,
        file_data: Dict[str, Any] = None,
        role: str = None,
        model_config: Optional[ModelConfig] = None,
        # mode and role are passed for _determine_system_instructions and _determine_user_prompt_content
        mode: Optional[str] = None,
        role: Optional[str] = None,
    ) -> str:
        """Generate response using MLX through OpenAI-compatible endpoint"""
        current_model_config = model_config or ModelConfig()

        # Determine the role for this specific request, defaulting to self.role if not provided
        current_request_role = role or self.role
        current_request_mode = mode or self.mode

        # Format messages for OpenAI chat completions API
        messages = []

        history = history if history is not None else [] # Ensure history is a list
        current_instructions = self._determine_system_instructions(system_instruction, history, current_request_role, current_request_mode)
        final_prompt_content = self._determine_user_prompt_content(prompt, history, current_request_role, current_request_mode)

        if current_instructions:
            messages.append({"role": "system", "content": current_instructions})
            
        if history:
            # Limit history to last 10 messages (or make configurable)
            for msg in history[-10:]:
                msg_role = msg.get("role")
                if msg_role in ["user", "human", "moderator", "assistant", "system"]: # Added moderator
                    messages.append({"role": "user" if msg_role in ["human", "moderator"] else msg_role,
                                     "content": msg.get("content")})

        messages.append({"role": "user", "content": final_prompt_content})

        # Handle file data
        if file_data:
            # MLX doesn't support vision directly, but we can include text content
            if file_data["type"] in ["text", "code"] and "text_content" in file_data:
                # Add text content to the last message
                last_msg = messages[-1]
                file_content = file_data["text_content"]
                file_path = file_data.get("path", "file")

                # Update the last message with file content
                last_msg["content"] = (
                    f"[File: {file_path}]\n\n{file_content}\n\n{last_msg['content']}"
                )

                # Replace the last message
                messages[-1] = last_msg
        # Handle multiple files
        elif isinstance(file_data, list) and file_data:
            # Get the last message
            last_msg = messages[-1]
            combined_content = last_msg["content"]

            # Add each text/code file content to the message
            for file_item in file_data:
                if (
                    file_item["type"] in ["text", "code"]
                    and "text_content" in file_item
                ):
                    file_content = file_item["text_content"]
                    file_path = file_item.get("path", "file")
                    combined_content = (
                        f"[File: {file_path}]\n\n{file_content}\n\n{combined_content}"
                    )

            # Update the last message with combined content
            last_msg["content"] = combined_content
            messages[-1] = last_msg

        try:
            # --- Debug Logging ---
            logger.debug(f"--- MLXClient Request ---")
            logger.debug(f"URL: {self.base_url}/v1/chat/completions")
            logger.debug(f"Model: {self.model}")
            logger.debug(f"Messages: {messages}")

            request_timeout = current_model_config.timeout if hasattr(current_model_config, 'timeout') and current_model_config.timeout is not None else 90

            request_payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                # Assuming MLX endpoint might use these from OpenAI compatibility
                "temperature": current_model_config.temperature,
                "max_tokens": current_model_config.max_tokens,
                # Seed and stop_sequences might not be universally supported by all local OpenAI-like servers
                # "seed": current_model_config.seed,
                # "stop": current_model_config.stop_sequences or None,
            }
            logger.debug(f"Request payload: {request_payload}")

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=request_timeout
            )
            response.raise_for_status()

            response_json = response.json()
            if not response_json or not response_json.get("choices") or \
               not isinstance(response_json["choices"], list) or len(response_json["choices"]) == 0 or \
               not response_json["choices"][0].get("message") or \
               response_json["choices"][0]["message"].get("content") is None: # Check for None explicitly
                logger.error(f"MLXClient API returned an invalid JSON structure: {response_json}")
                raise ModelClientResponseError("MLXClient API returned an invalid JSON structure.")

            return response_json["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as e:
            logger.error(f"MLXClient HTTP error: {e.response.status_code} {e.response.text if e.response else 'No response text'}")
            status_code = e.response.status_code
            if status_code == 401 or status_code == 403:
                raise ModelClientAuthError(f"MLXClient authentication/permission error ({status_code}): {e}") from e
            elif status_code == 429:
                raise ModelClientAPIError(f"MLXClient rate limit exceeded ({status_code}): {e}") from e
            elif 400 <= status_code < 500:
                raise ModelClientAPIError(f"MLXClient client error ({status_code}): {e}") from e
            elif 500 <= status_code < 600:
                raise ModelClientAPIError(f"MLXClient server error ({status_code}): {e}") from e
            else:
                raise ModelClientAPIError(f"MLXClient HTTP error ({status_code}): {e}") from e
        except requests.exceptions.ConnectionError as e:
            logger.error(f"MLXClient connection error: {e}")
            raise ModelClientTimeoutError(f"MLXClient connection error: {e}") from e
        except requests.exceptions.Timeout as e:
            logger.error(f"MLXClient request timed out: {e}")
            raise ModelClientTimeoutError(f"MLXClient request timed out: {e}") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"MLXClient request exception: {e}")
            raise ModelClientAPIError(f"MLXClient request failed: {e}") from e
        except ModelClientError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in MLXClient: {e}")
            # Check for JSONDecodeError, which is a subclass of ValueError
            if isinstance(e, ValueError) and "json" in str(e).lower(): # Heuristic check for JSON decode issues
                 raise ModelClientResponseError(f"Failed to decode JSON response from MLXClient: {e}") from e
            raise ModelClientError(f"An unexpected error occurred in MLXClient: {e}") from e



class OllamaClientLangchain(BaseClient):
    """Client for local Ollama model interactions using LangChain."""

    def __init__(self, mode: str, domain: str, role: str = None, model: str = "phi4:latest", api_key: Optional[str] = None):
        super().__init__(mode=mode, api_key="", domain=domain, model=model, role=role) # api_key is ""
        self.base_url = "http://localhost:11434"
        try:
            # Initialize with default temperature, can be overridden by model_config in generate_response
            self.lc_client = ChatOllama(
                model=self.model,
                temperature=ModelConfig().temperature, # Default from ModelConfig
            )
            logger.debug(f"Initialized LangChain Ollama client for model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain ChatOllama: {e}", exc_info=True)
            self.lc_client = None
            raise

    def test_connection(self) -> bool:
        """Test Ollama connection via LangChain client."""
        if not self.lc_client:
            logger.error("LangChain Ollama client not initialized.")
            return False
        try:
            self.lc_client.invoke("Hi") # Test with a simple invoke
            logger.info("Ollama (LangChain) connection test successful")
            return True
        except Exception as e:
            logger.error(f"Ollama (LangChain) connection test failed: {e}", exc_info=True)
            return False

    def generate_response(
        self,
        prompt: str,
        system_instruction: str = None,
        history: List[Dict[str, str]] = None,
        file_data: Any = None, # Can be Dict or List[Dict]
        model_config: Optional[ModelConfig] = None,
        mode: str = None, # Retained for determining instructions/prompt
        role: str = None, # Retained for determining instructions/prompt
    ) -> str:
        """Generate a response from your local Ollama model using LangChain."""
        if not self.lc_client:
            logger.error("LangChain Ollama client not initialized. Cannot generate response.")
            return "Error: Ollama client not initialized."

        # Determine the role for this specific request, defaulting to self.role if not provided
        current_request_role = role or self.role
        current_request_mode = mode or self.mode
        history = history if history is not None else []

        # Use base class helpers to determine instructions and final prompt
        current_instructions = self._determine_system_instructions(system_instruction, history, current_request_role, current_request_mode)
        final_prompt_content = self._determine_user_prompt_content(prompt, history, current_request_role, current_request_mode)

        # Prepare LangChain messages
        lc_messages = []
        if current_instructions:
            # Create a PromptTemplate from the instructions string
            system_prompt_template = PromptTemplate(template=current_instructions)
            # Pass the PromptTemplate to the SystemMessagePromptTemplate
            lc_messages.append(SystemMessage(prompt=system_prompt_template))

        for msg in history:
            msg_role = msg.get("role")
            msg_content = msg.get("content", "")
            if msg_role in ["user", "human"]:
                lc_messages.append(HumanMessage(prompt=PromptTemplate(template=msg_content, input_variables=[])))
            elif msg_role == "assistant":
                lc_messages.append(AIMessage(prompt=PromptTemplate(template=msg_content, input_variables=[])))
            # Ignore system messages in history as it's handled above

        # Prepare final user message content (text + optional images)
        user_message_content_parts = [] # Use a list for multimodal content
        user_message_content_parts.append({"type": "text", "text": final_prompt_content})

        images_base64 = []
        if file_data:
            # Normalize file_data to always be a list
            processed_files = file_data if isinstance(file_data, list) else [file_data]
            for item in processed_files:
                if isinstance(item, dict):
                    if item.get("type") == "image" and "base64" in item:
                        images_base64.append(item["base64"])
                    elif item.get("type") == "video" and "key_frames" in item:
                        # Treat video frames as images
                        images_base64.extend([f["base64"] for f in item.get("key_frames", []) if "base64" in f])
                    # Note: LangChain ChatOllama might not support raw video chunks directly

        for img_b64 in images_base64:
            # LangChain expects image URLs in this format for Ollama
            user_message_content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"} # Assuming JPEG
            })

        # Add the final HumanMessage with potentially multimodal content
        lc_messages.append(HumanMessage(content=user_message_content_parts))
        # Update the client configuration for multimodal content
        if images_base64:
            # Configure client for vision model if we have images
            if not any(vm in self.model.lower() for vm in ["llava", "bakllava", "vision"]):
                logger.warning(f"Model {self.model} may not support vision. Attempting request anyway.")
            
            # For LangChain's ChatOllama, image content should already be properly formatted
            # in user_message_content_parts above with image_url entries
            self.lc_client.model_kwargs.update({
            "num_ctx": 16384,  # Increase context for images
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.1
            })

        # Check if we have video frames to process
        if any(isinstance(item, dict) and item.get("type") == "video" for item in (file_data if isinstance(file_data, list) else [file_data] if file_data else [])):
            logger.info("Processing video frames for analysis")
            # Video frames are already handled in the images_base64 list above
            # Add video-specific context to the prompt if needed
            if user_message_content_parts[0]["text"]:
                user_message_content_parts[0]["text"] += "\n\nPlease analyze the sequence of video frames."

        # Prepare request options from model_config
        current_model_config = model_config or ModelConfig()

        request_options = {
            "temperature": current_model_config.temperature,
            "num_predict": current_model_config.max_tokens, # LangChain uses num_predict
            "stop": current_model_config.stop_sequences or None, # API expects None if empty
            "seed": current_model_config.seed,
            # Other Ollama specific options can be added here if they are in ModelConfig
            # e.g., top_k, top_p, num_ctx.
            # For example, if ModelConfig had top_k:
            # "top_k": current_model_config.top_k if hasattr(current_model_config, 'top_k') else None,
        }
        # Filter out None values if ChatOllama doesn't handle them gracefully for all params
        request_options = {k: v for k, v in request_options.items() if v is not None}
        if not request_options.get("stop"): # Ensure 'stop' is not an empty list if that causes issues
            request_options.pop("stop", None)


        try:
            logger.debug(f"--- Ollama (LangChain) Request ---")
            logger.debug(f"Model: {self.model}")
            logger.debug(f"Options: {request_options}")

            response = self.lc_client.invoke(lc_messages, **request_options)

            if not response or not hasattr(response, 'content') or response.content is None:
                logger.error("Ollama (LangChain) returned an empty or invalid response structure.")
                raise ModelClientResponseError("Ollama (LangChain) returned an empty or invalid response.")

            logger.debug(f"--- Ollama (LangChain) Response ---")
            logger.debug(f"Content: {response.content[:200]}...")
            return response.content

        except OutputParserException as e:
            logger.error(f"Ollama (LangChain) output parsing error: {e}")
            raise ModelClientResponseError(f"Failed to parse response from Ollama (LangChain): {e}") from e
        except OllamaAPIError as e:
            logger.error(f"Ollama API error via LangChain: {e}. Status code: {e.status_code if hasattr(e, 'status_code') else 'N/A'}")
            if hasattr(e, 'status_code'):
                if e.status_code == 401 or e.status_code == 403:
                    raise ModelClientAuthError(f"Ollama auth/permission error via LangChain ({e.status_code}): {e}") from e
                elif e.status_code == 404:
                    raise ModelClientAPIError(f"Ollama model not found via LangChain ({e.status_code}): {self.model}. {e}") from e
            raise ModelClientAPIError(f"Ollama API error via LangChain: {e}") from e
        except OllamaRequestError as e:
            logger.error(f"Ollama request error via LangChain (connection/network): {e}")
            raise ModelClientTimeoutError(f"Ollama connection/request error via LangChain: {e}") from e
        except LangChainError as e:
            logger.error(f"LangChain error with Ollama client: {e}")
            if isinstance(e.__cause__, requests.exceptions.ConnectionError) or \
               isinstance(e.__cause__, requests.exceptions.ConnectTimeout) or \
               isinstance(e.__cause__, requests.exceptions.ReadTimeout):
                raise ModelClientTimeoutError(f"Network/Timeout error via LangChain for Ollama: {e}") from e
            raise ModelClientAPIError(f"A LangChain error occurred with Ollama: {e}") from e
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Direct requests.exceptions.ConnectionError with Ollama (LangChain): {e}")
            raise ModelClientTimeoutError(f"Network connection error for Ollama (LangChain): {e}") from e
        except requests.exceptions.Timeout as e:
            logger.error(f"Direct requests.exceptions.Timeout with Ollama (LangChain): {e}")
            raise ModelClientTimeoutError(f"Request timeout for Ollama (LangChain): {e}") from e
        except ModelClientError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in OllamaClientLangchain: {e}", exc_info=True)
            raise ModelClientError(f"An unexpected error occurred in OllamaClientLangchain: {e}") from e


class OllamaClient(BaseClient):
    """Client for local Ollama model interactions"""

    # Initialize the client
    def __init__(
        self, mode: str, domain: str, role: str = None, model: str = "phi4:latest", api_key: Optional[str] = None,
    ):
        super().__init__(mode=mode, api_key="", domain=domain, model=model, role=role) # api_key is ""
        self.base_url = "http://localhost:11434"
        self.client = Client(host=self.base_url)

    def test_connection(self) -> None:
        """Test Ollama connection"""
        try:
            self.client.list()
            logger.info("Ollama connection test successful")
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            raise

    def generate_response(
        self,  # Changed to synchronous method
        prompt: str,
        system_instruction: str = None,
        history: List[Dict[str, str]] = None,
        file_data: Dict[str, Any] = None,
        model_config: Optional[ModelConfig] = None,
        mode: str = None, # Retained for determining instructions/prompt
        role: str = None, # Retained for determining instructions/prompt
    ) -> str:
        """Generate a response from your local Ollama model."""
        # Determine the role for this specific request, defaulting to self.role if not provided
        current_request_role = role or self.role
        current_request_mode = mode or self.mode

        history = history if history is not None else [] # Ensure history is a list

        # Determine instructions and final prompt content using BaseClient helpers
        current_instructions = self._determine_system_instructions(system_instruction, history, current_request_role, current_request_mode)
        final_prompt_content = self._determine_user_prompt_content(prompt, history, current_request_role, current_request_mode)

        # Prepare messages for Ollama's chat API
        messages = []
        if current_instructions:
            messages.append({"role": "system", "content": current_instructions})

        if history:
            for msg in history:
                msg_role = msg.get("role")
                if msg_role in ["user", "assistant", "human"]:
                    mapped_role = "user" if msg_role == "human" else msg_role
                    messages.append({"role": mapped_role, "content": msg.get("content")})

        messages.append({"role": "user", "content": final_prompt_content})

        # Check if this is a vision-capable model and we have image data
        is_vision_model = any(
            vm in self.model.lower()
            for vm in [
                "gemma3",
                "llava",
                "vision",
                "llava-phi3",
                "mistral-medium",
                "llama2-vision",
            ]
        )

        # Handle file data for Ollama (Corrected Image Handling)
        if is_vision_model and file_data:
            if isinstance(file_data, list) and file_data:
                # Handle multiple files
                all_images = []
                for file_item in file_data:
                    if isinstance(file_item, dict) and "type" in file_item:
                        if file_item["type"] == "image" and "base64" in file_item:
                            all_images.append(file_item["base64"])
                        elif (
                            file_item["type"] == "video"
                            and "key_frames" in file_item
                            and file_item["key_frames"]
                        ):
                            all_images.extend(
                                [frame["base64"] for frame in file_item["key_frames"]]
                            )
                        elif (
                            file_item["type"] == "video" and "video_chunks" in file_item
                        ):
                            # For Ollama, we can send the entire video content since it's running locally
                            # Combine all chunks into a single video content
                            full_video_content = "".join(file_item["video_chunks"])
                            logger.info(
                                f"Added full video content to Ollama request ({len(full_video_content)} bytes)"
                            )
                            # Note: Currently, we're not handling video chunks in the multiple files case

                if all_images:
                    messages[-1]["images"] = all_images
                    logger.info(f"Added {len(all_images)} images to Ollama request")
            elif isinstance(file_data, dict) and "type" in file_data:
                if file_data["type"] == "image" and "base64" in file_data:
                    # Add the image directly to the *last* message in the messages list
                    messages[-1]["images"] = [
                        file_data["base64"]
                    ]  # Correct image format
                elif (
                    file_data["type"] == "video"
                    and "key_frames" in file_data
                    and file_data["key_frames"]
                ):
                    # --- Key Change: Send *all* sampled frames ---
                    messages[-1]["images"] = [
                        frame["base64"] for frame in file_data["key_frames"]
                    ]
                elif file_data["type"] == "video" and "video_chunks" in file_data:
                    # For Ollama, we can send the entire video content since it's running locally
                    # Combine all chunks into a single video content
                    full_video_content = "".join(file_data["video_chunks"])

                    # Add the video content to the message
                    # Note: This assumes Ollama can handle video content directly
                    # If not, this will need to be modified to extract frames
                    messages[-1]["video"] = full_video_content
                    logger.info(
                        f"Added full video content to Ollama request ({len(full_video_content)} bytes)"
                    )

        try:
            # --- Debug Logging ---
            logger.debug(f"--- Ollama Request ---")
            logger.debug(f"Model: {self.model}")
            logger.debug(f"Messages: {messages}")

            current_model_config = model_config or ModelConfig()
            options = {
                "num_ctx": 16384,
                "num_predict": current_model_config.max_tokens,
                "temperature": current_model_config.temperature,
                "stop": current_model_config.stop_sequences,
                "seed": current_model_config.seed,
                "num_batch": 512,
                "top_k": 40,
                "use_mmap": True,
            }
            options = {k: v for k, v in options.items() if v is not None}
            if not options.get("stop"):
                options.pop("stop", None)

            response: ChatResponse = self.client.chat(
                model=self.model,
                messages=messages,
                stream=False,
                options=options,
            )

            if not response or not response.get('message') or not response['message'].get('content'):
                logger.error("Ollama API returned an empty or invalid response structure.")
                raise ModelClientResponseError("Ollama API returned an empty or invalid response.")

            return response['message']['content']

        except OllamaAPIError as e:
            logger.error(f"Ollama API error: {e}. Status code: {e.status_code if hasattr(e, 'status_code') else 'N/A'}")
            if hasattr(e, 'status_code'):
                if e.status_code == 401 or e.status_code == 403:
                    raise ModelClientAuthError(f"Ollama authentication/permission error ({e.status_code}): {e}") from e
                elif e.status_code == 404:
                    raise ModelClientAPIError(f"Ollama model not found ({e.status_code}): {self.model}. {e}") from e
            raise ModelClientAPIError(f"Ollama API error: {e}") from e
        except OllamaResponseError as e:
            logger.error(f"Ollama response error: {e}. Status code: {e.status_code if hasattr(e, 'status_code') else 'N/A'}")
            if hasattr(e, 'status_code'):
                if e.status_code == 401 or e.status_code == 403:
                    raise ModelClientAuthError(f"Ollama authentication/permission error ({e.status_code}): {e}") from e
                elif e.status_code == 404:
                    raise ModelClientAPIError(f"Ollama model not found ({e.status_code}): {self.model}. {e}") from e
            raise ModelClientAPIError(f"Ollama response error: {e}") from e
        except OllamaRequestError as e:
            logger.error(f"Ollama request error (connection/network): {e}")
            raise ModelClientTimeoutError(f"Ollama connection or request error: {e}") from e
        except ModelClientError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in OllamaClient: {e}")
            raise ModelClientError(f"An unexpected error occurred in OllamaClient: {e}") from e
