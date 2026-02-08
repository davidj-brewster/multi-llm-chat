"""Base and model-specific client implementations with memory optimizations."""

import os
import logging
import random
import inspect
from typing import List, Dict, Optional, TypeVar, Any, Union
from dataclasses import dataclass
from google import genai
from google.genai import types
from openai import OpenAI
from anthropic import Anthropic
from ollama import ChatResponse, Client, Options
import requests
from adaptive_instructions import AdaptiveInstructionManager, InstructionSet
from shared_resources import MemoryManager
from configuration import detect_model_capabilities

logger = logging.getLogger(__name__)

T = TypeVar("T")
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
MAX_TOKENS = 4096
TOKENS_PER_TURN = MAX_TOKENS


@dataclass
class ModelConfig:
    """
    Configuration for AI model generation parameters.

    This dataclass encapsulates the common configuration parameters used across
    different AI model providers for controlling the generation behavior. It provides
    a standardized interface for setting parameters like temperature, token limits,
    and stopping criteria.

    Attributes:
        temperature (float): Controls randomness in response generation. Higher values
            (e.g., 0.8) produce more diverse and creative outputs, while lower values
            (e.g., 0.2) produce more deterministic and focused outputs. Defaults to 0.8.

        max_tokens (int): Maximum number of tokens to generate in the response.
            This helps control response length and prevent excessive token usage.
            Defaults to MAX_TOKENS=2048.

        stop_sequences (List[str], optional): List of strings that, when encountered,
            will stop the model from generating further text. Useful for controlling
            response format. Defaults to None.

        seed (Optional[int], optional): Random seed for deterministic generation.
            Setting the same seed with the same prompt and parameters should produce
            the same output. Defaults to a random integer between 0 and 1000.

    Examples:
        Basic configuration:
        >>> config = ModelConfig()
        >>> config.temperature
        >>> config.max_tokens

        Custom configuration:
        >>> config = ModelConfig(
                temperature=0.3,
                max_tokens=MAX_TOKENS,
                stop_sequences=["END", "STOP"],
                seed=42
            )
        >>> config.temperature
        0.3
        >>> config.stop_sequences
        ['END', 'STOP']

        Using with a client:
        >>> client = GeminiClient(mode="ai-ai", role="assistant", api_key="   ", domain="science")
        >>> response = client.generate_response(
                prompt="Explain quantum entanglement",
                model_config=ModelConfig(temperature=0.2, max_tokens=MAX_TOKENS)
            )
    """

    temperature: float = 0.8
    max_tokens: int = MAX_TOKENS
    stop_sequences: List[str] = None
    seed: Optional[int] = random.randint(0, 1000)


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
        >>> client = BaseClient(mode="ai-ai", api_key="sk-   ", domain="science")
        >>> client.validate_connection()
        True

        Custom instruction handling:
        >>> client = BaseClient(mode="human-ai", api_key="sk-   ", domain="medicine")
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
                "type": "image",
                "base64": "base64_encoded_data",
                "mime_type": "image/jpeg",
                "dimensions": (800, 600)
            }
        >>> prepared = client._prepare_file_content(image_data)
        >>> prepared["type"]
        'image'
    """

    def __init__(
        self, mode: str, api_key: str, domain: str = "", model: str = "", role: str = "", persona: str = None
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
            api_key (str): The API key for authentication with the model provider.
                Will be stripped of whitespace and validated during connection.
            domain (str, optional): The knowledge domain or subject area for the
                conversation. Used for generating context-aware instructions.
                Defaults to an empty string.
            model (str, optional): The specific model identifier to use. This is used
                to detect capabilities and may be overridden by subclasses.
                Defaults to an empty string.
            role (str, optional): The role this client is playing in the conversation.
                Valid values include "human", "user", "assistant", or "model".
                Defaults to an empty string.
            persona (str, optional): Rich persona description from YAML config.
                Defines personality, power dynamics, cognitive style, etc.
                Defaults to None.

        Note:
            Subclasses typically override this method to initialize model-specific
            clients and configurations while calling super().__init__() to ensure
            base functionality is properly initialized.

        Example:
            >>> client = BaseClient(
                    mode="ai-ai",
                    api_key="sk-   ",
                    domain="Artificial Intelligence",
                    model="gpt-4",
                    role="assistant"
                )
        """
        self.api_key = api_key.strip() if api_key else ""
        self.domain = domain
        self.mode = mode
        self.role = role
        self.model = model
        self.persona = persona
        self.capabilities = detect_model_capabilities(model)
        self.instructions = None
        self.adaptive_manager = AdaptiveInstructionManager(mode=self.mode, persona=self.persona)

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
                    "type": "image",
                    "base64": "base64_encoded_data",
                    "mime_type": "image/jpeg",
                    "dimensions": (800, 600)
                }
            >>> result = client._prepare_file_content(file_data)
            >>> result["type"]
            'image'
            >>> result["width"]
            800

            Processing a text file:
            >>> file_data = {
                    "type": "text",
                    "text_content": "This is a sample text file.",
                    "path": "sample.txt"
                }
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
        if file_data["type"] == "video":
            # For video, we'll use key frames
            return {
                "type": "video",
                "chunks": file_data.get(
                    "video_chunks", []
                ),  # Video chunks if available
                "num_chunks": file_data.get("num_chunks", 0),  # Number of chunks
                "frames": file_data.get("key_frames", []),  # Fallback to key frames
                "duration": file_data.get("duration", 0),
                "mime_type": file_data.get("mime_type", "video/mp4"),
                "fps": file_data.get("fps", 0),
                "resolution": file_data.get("resolution", (0, 0)),
                "path": file_data.get("video_path", ""),
            }
        if file_data["type"] in ["text", "code"]:
            return {
                "type": file_data["type"],
                "content": file_data.get("text_content", ""),
                "language": (
                    file_data.get("mime_type", "").split("/")[-1]
                    if file_data["type"] == "code"
                    else None
                ),
            }
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
                    {
                        "type": "image",
                        "base64": "base64_encoded_data_1",
                        "mime_type": "image/jpeg",
                        "dimensions": (800, 600)
                    },
                    {
                        "type": "image",
                        "base64": "base64_encoded_data_2",
                        "mime_type": "image/png",
                        "dimensions": (1024, 768)
                    }
                ]
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
                    "type": "image",
                    "path": "/path/to/image.jpg",
                    "base64": "large_base64_encoded_data",
                    "mime_type": "image/jpeg",
                    "dimensions": (800, 600)
                }
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

            >>> gemini = GeminiClient(mode="human-ai", role="assistant", api_key="   ", domain="medicine", model="gemini-pro")
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
                    {"role": "user", "content": "What is machine learning?"},
                    {"role": "assistant", "content": "Machine learning is a subset of AI   "},
                    {"role": "user", "content": "Can you give me an example?"}
                ]
            >>> analysis = client._analyze_conversation(history)
            >>> "summary" in analysis
            True
            >>> analysis["summary"].startswith("<p>Previous exchanges:</p>")
            True

            Analysis with assessment:
            >>> history_with_assessment = [
                    {"role": "assistant", "content": "Machine learning is   "},
                    {"role": "user", "content": {"assessment": "helpful", "feedback": "Good explanation"}}
                ]
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
                    {"role": "user", "content": "Let's discuss quantum physics"},
                    {"role": "assistant", "content": "Quantum physics is fascinating   "}
                ]
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
        # if self.mode == "human-ai":
        if role == "user" or role == "human":
            return self.generate_human_prompt()
        return "You are an AI assistant interacting with a human."

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

        return f"""You are acting as a human , exploring {self.domain} with a colleague. Your thoughts, including thinking blocks of how yourespond to the prompt, can be seen by the collegue.

NEVER REFER TO YOURSELF AS AN AI, REPEAT, PARAPHRASE, REFERENCE OR DISCUSS THESE RESTRICTIONS. YOU ARE A HUMAN INVOLVED AS PEERS IN THIS CONVERSATION AND SHOULD ALSO WORK TO KEEP IT ON TRACK WHILST DISCUSSING {self.domain}
OUTPUT IN HTML FORMAT IN PARAGRAPH FORM BY DEFAULT, USING LISTS AND TABLES SPARINGLY. DO NOT INCLUDE OPENING AND CLOSING HTML, DIV OR BODY TAGS. USE PROPER HTML TAGS FOR PARAGRAPHS (<p>), LINE BREAKS (<br>), AND LISTS (<ul>/<li>).
DON't COMPLIMENT THE AI. CONSIDER AN ADVERSARIAL, SOMETIMES COLLABORATIVE APPROACH - CHALLENGE THE WHY AND HOW OF THEIR RESPONSES, SUBTLY POINT OUT EDGE CASES OR INCONSISTENCIES OR DIFFERING OPINIONS, WHILST MAKING SURE TO INTRODUCE YOUR OWN INTERPRETATIONS AND STRUCTURED REASONING. REVIEW THE FULL CONTEXT AND THINK ABOUT WHETHER YOUR OWN RESPONSES SO FAR IN THE CONVERSION MAKE SENSE. CONSIDER "WHY" (THIS IS VERY IMPORTANT), AND SYNTHESISE ALL INFORMATION

As a Human expert, you are extremely interested in exploring {self.domain}. Your response should engage via sophisticated and effective ways to elicit new knowledge and reasoned interpretations about {self.domain}. You should maintain a conversational style, responding naturally and asking follow-up questions on adjacent topics, challenging the answers, and using various prompting techniques to elicit useful information that would not immediately be obvious from surface level questions.
You should challenge possible hallucinations or misinterpretations with well reasoned counter-positions, and you should challenge your own thinking as well, in a human style, and ask for explanations for things that you don't understand or agree with (or pretend not to).
Even when challenging assertions, bring in related sub-topics and reasoning and your own interpretation or possible solutions to the discussion so that it doesn't get stuck micro-analysing one tiny detail..
Review YOUR previous inputs to see if you are reusing the same phrases and approaches in your prompts (e.g., "Let me challenge    I would think this would be most important"    and dynamically adapt to this situation)

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

        Also stores self._last_instruction_set for structured per-provider injection.

        Args:
            system_instruction (Optional[str]): Explicitly provided system instructions.
            history (List[Dict[str, str]]): Conversation history.
            role (Optional[str]): The role being prompted.
            mode (Optional[str]): The current conversation mode.

        Returns:
            str: The determined system instructions.
        """
        self._last_instruction_set = None

        # Use provided system_instruction if available
        if system_instruction is not None:
            return system_instruction

        # Otherwise, use adaptive manager (allows goal detection)
        # The manager handles role differentiation internally for human simulation
        try:
            # Ensure history is a list, default to empty list if None
            history = history if history is not None else []
            result = self.adaptive_manager.generate_instructions(
                history, role=role, domain=self.domain, mode=mode or self.mode
            )
            # Capture the structured InstructionSet for per-provider injection
            self._last_instruction_set = getattr(self.adaptive_manager, 'last_instruction_set', None)
            return result
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
        if self.mode:
            current_mode = self.mode
        else:
            if mode:
                current_mode = mode
            else:
                logger.critical(f"_determine_user_prompt_content: mode is not configured in this step. Terminating. Debug follows {prompt} {history}")
                raise ValueError("_determine_user_prompt_content: No conversation mode configured")

        if not current_mode:
            logger.critical("_determine_user_prompt_content: current_mode unconfigured somehow")
            raise ValueError("_determine_user_prompt_content: current_mode unconfigured")

        is_goal_task = any(marker in self.domain.upper() for marker in ["GOAL", "TASK", "WRITE A", "MAKE A", "WRITE A", "DESIGN", "BUILD"])

        if is_goal_task:
            # For goal tasks, only the 'human' role gets the simulation prompt
            if role == "human" or role == "user":
                return self.generate_human_prompt(history)
            # Assistant role gets the raw prompt (previous output)
            return prompt
        if (role == "human" or role == "user" or current_mode == "ai-ai") and current_mode != "default" and current_mode != "no-meta-prompting":
            # Non-goal tasks, ai-ai mode or human role gets simulation prompt
            return self.generate_human_prompt(history)
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
            >>> client = BaseClient(mode="ai-ai", api_key="sk-   ", domain="science")
            >>> client.validate_connection()
            True

            Handling validation failure:
            >>> client_with_invalid_key = BaseClient(mode="ai-ai", api_key="invalid", domain="science")
            >>> try:
                    is_valid = client_with_invalid_key.validate_connection()
                except Exception as e:
                    is_valid = False
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
            >>> client = BaseClient(mode="ai-ai", api_key="sk-   ", domain="science")
            >>> client.test_connection()
            True

            # In a subclass implementation:
            >>> gemini = GeminiClient(mode="ai-ai", role="assistant", api_key="   ", domain="science")
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

    def generate_speech(self, text: str, voice_name: str = "default", model: Optional[str] = None) -> bytes:
        """
        Generates speech from text. This method is intended to be overridden
        by subclasses that support Text-to-Speech (TTS).

        Args:
            text (str): The text to be converted to speech.
            voice_name (str, optional): The voice to use for the speech.
            model (str, optional): The specific TTS model to use.

        Returns:
            bytes: The raw audio data.

        Raises:
            NotImplementedError: If the client does not support speech generation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support speech generation."
        )


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
                mode="ai-ai",
                role="assistant",
                api_key="your_api_key",  # Will use GOOGLE_API_KEY env var if not provided
                domain="science",
                model="gemini-2.0-flash-exp"
            )
        >>> response = gemini.generate_response(
                prompt="Explain the theory of relativity",
                system_instruction="You are a physics professor."
            )
        >>> print(response[:50])
        'The theory of relativity, developed by Albert Einstein   '

        Image analysis:
        >>> image_data = {
                "type": "image",
                "base64": "base64_encoded_image_data",
                "mime_type": "image/jpeg",
                "dimensions": (800, 600)
            }
        >>> response = gemini.generate_response(
                prompt="Describe what you see in this image",
                file_data=image_data
            )
    """

    def __init__(
        self,
        mode: str,
        role: str,
        api_key: str,
        domain: str,
        model: str = "gemini-2.5-flash-lite",
    ):
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        super().__init__(
            mode=mode, api_key=api_key, domain=domain, model=model, role=role
        )
        self.model_name = self.model
        self.role = "user" if role in ["user", "human"] else "model"
        self.client = None
        try:
            self.client = genai.Client(
                api_key=self.api_key, http_options={"api_version": "v1alpha"}
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise ValueError(f"Invalid Gemini API key: {e}") from e

        # Initialize generation config
        self._setup_generation_config()

    def _setup_generation_config(self):
        """
        Initialize the default generation configuration for Gemini models.

        Sets up the default parameters for content generation, including temperature,
        token limits, candidate count, response format, and safety settings. This
        configuration serves as the base for all response generation requests.

        The configuration can be overridden on a per-request basis in the generate_response
        method.
        """
        self.generation_config = types.GenerateContentConfig(
            temperature=0.7,
            maxOutputTokens=MAX_TOKENS,
            candidateCount=1,
            responseMimeType="text/plain",
            safety_settings=[],
        )

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
            >>> gemini = GeminiClient(mode="ai-ai", role="assistant", api_key="   ", domain="science")
            >>> response = gemini.generate_response(
                    prompt="What is quantum entanglement?",
                    system_instruction="You are a quantum physics professor.",
                    model_config=ModelConfig(temperature=0.3, max_tokens=MAX_TOKENS)
                )
        """
        if model_config is None:
            model_config = ModelConfig()

        # Update mode and role if provided
        if mode:
            self.mode = mode
        if role:
            self.role = "user" if role in ["user", "human"] else "model"

        if model_config is None:
            model_config = ModelConfig()
        if role == "user":
            self.role = "user"
        else:
            self.role = "model"
        if role:  # and not self.role:
            self.role = role

        history = history if history is not None else [] # Ensure history is a list

        # Update instructions based on conversation history
        current_instructions = self._determine_system_instructions(system_instruction, history, role, mode)

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
        # Gemini: prepend interventions to prompt for recency bias
        iset = getattr(self, '_last_instruction_set', None)
        if iset and isinstance(iset, InstructionSet) and iset.interventions:
            final_prompt_content = f"[IMPORTANT DIRECTIVE]\n{iset.interventions}\n\n{final_prompt_content}"
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
                    # Format image for Gemini (single image case
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
                            config=types.GenerateContentConfig(
                                temperature=0.8,
                                systemInstruction=current_instructions,
                                max_output_tokens=4096,
                                candidateCount=1,
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
                            config=types.GenerateContentConfig(
                                temperature=0.8,
                                systemInstruction=current_instructions,
                                max_output_tokens=4096,
                                candidateCount=1,
                                safety_settings=[
                                    types.SafetySetting(
                                        category="HARM_CATEGORY_HATE_SPEECH",
                                        threshold="BLOCK_ONLY_HIGH",
                                    ),
                                    types.SafetySetting(
                                        category="HARM_CATEGORY_HARASSMENT",
                                        threshold="BLOCK_ONLY_HIGH",
                                    ),
                                    types.SafetySetting(
                                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                        threshold="BLOCK_ONLY_HIGH",
                                    ),
                                    types.SafetySetting(
                                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                                        threshold="BLOCK_ONLY_HIGH",
                                    ),
                                    types.SafetySetting(
                                        category="HARM_CATEGORY_CIVIC_INTEGRITY",
                                        threshold="BLOCK_ONLY_HIGH",
                                    ),
                                ],
                            ),
                        )

                    return (
                        str(response.text)
                        if (response and response is not None)
                        else ""
                    )
                if "key_frames" in file_data and file_data["key_frames"] or "image" in file_data and "base64" in file_data:
                    # Fallback to key frames if available
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
            logger.debug("--- Gemini Request ---")
            logger.debug(f"Model: {self.model_name}")
            logger.debug(f"System Instruction: {current_instructions}")
            logger.debug(f"Contents: {contents}")
            # Generate final response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,  # Use the properly formatted contents array
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    max_output_tokens=model_config.max_tokens if model_config else MAX_TOKENS, # Use model_config
                    systemInstruction=current_instructions,
                    candidateCount=1,
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_ONLY_HIGH",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_ONLY_HIGH",
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
                            threshold="BLOCK_ONLY_HIGH",
                        ),
                    ],
                ),
            )

            return str(response.text) if (response and response is not None) else ""
        except Exception as e:
            logger.error(f"GeminiClient generate_response error: {e}")
            raise

    def generate_speech(self, text: str, voice_name: str = "Kore", model: str = "gemini-2.5-flash-preview-tts") -> bytes:
        """
        Generates speech from text using the Gemini TTS API.

        Args:
            text (str): The text to be converted to speech.
            voice_name (str, optional): The name of the prebuilt voice to use. Defaults to "Kore".
            model (str, optional): The specific TTS model to use. Defaults to "gemini-2.5-flash-preview-tts".

        Returns:
            bytes: The raw audio data in WAV format.
        """
        try:
            preview = text[:30] + ("..." if len(text) > 30 else "")
            logger.info(f"Generating speech with model {model} for text: '{preview}' with voice: {voice_name}")
            response = self.client.models.generate_content(
                model=model,
                contents=[{"parts": [{"text": text}]}],
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name,
                            )
                        )
                    ),
                ),
            )
            if response.candidates and response.candidates[0].content.parts:
                audio_data = response.candidates[0].content.parts[0].inline_data.data
                logger.info(f"Successfully generated {len(audio_data)} bytes of audio data.")
                return audio_data
            logger.error("No audio data received from Gemini TTS API.")
            return b""
        except Exception as e:
            logger.error(f"GeminiClient generate_speech error: {e}")
            raise


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
        api_key: str,
        mode: str,
        domain: str,
        model: str = "claude-3-7-sonnet", # Default model
    ):
        super().__init__(
            mode=mode, api_key=api_key, domain=domain, model=model, role=role
        )
        try:
            api_key = anthropic_api_key or api_key
            if not api_key:
                logger.critical("Missing required Anthropic API key for Claude models.")
                raise ValueError("No Anthropic API key provided. Please set the ANTHROPIC_API_KEY.")

            # --- Map user-friendly names to specific API model IDs ---
            model_map = {
                # Add other mappings here if needed, e.g.:
                # "sonnet": "claude-3-5-sonnet-latest",
                # "claude-3-5-sonnet": "claude-3-5-sonnet-latest",
                # "opus": "claude-3-opus-latest",
                # "claude-3-opus": "claude-3-opus-latest",
            }
            # Use mapped name if found, otherwise use the provided model name
            self.model = model_map.get(model.lower(), model)
            logger.info(f"Using Claude model ID: {self.model}")
            # --- End Mapping ---

            self.client = Anthropic(api_key=api_key)

            # Enhanced vision parameters
            self.vision_max_resolution = 1800  # Up from default 1024
            self.max_images_per_request = 10  # Support for multiple images
            self.high_detail_vision = True  # Enable detailed medical image analysis
            self.video_frame_support = True  # Enable video frame extraction

            # Reasoning parameters and extended thinking
            self.extended_thinking = False  # Whether to enable extended thinking
            self.budget_tokens = None  # Budget tokens for extended thinking
            self.reasoning_level = "auto"  # Options: none, low, medium, high, auto

            # Set capability flags based on model
            self._update_capabilities()

            # If this is Claude 3.7, enable extended thinking by default for deep analytical tasks
            if "claude-3-7" in model.lower():
                self.extended_thinking = False  # Disabled by default but ready to use
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            raise
        self.max_tokens = MAX_TOKENS

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
        # The capability check for "advanced_reasoning" might be too restrictive.
        # Rely on ai-battle.py's explicit configuration for extended_thinking.
        # If the API doesn't support it for a model, it will error out, which is acceptable.
        # if not self.capabilities.get("advanced_reasoning", False):
        #     logger.warning(
        #         f"Extended thinking might not be supported for {self.model} based on current capability flags. Proceeding as configured."
        #     )
            # return # Decided to allow setting it and let API handle validity

        self.extended_thinking = enabled
        self.budget_tokens = budget_tokens
        logger.info( # Changed to info to match ai-battle.py log level for this message
            f"Set extended thinking={enabled}, budget_tokens={budget_tokens} for {self.model}"
        )

    def _update_capabilities(self):
        """Update capability flags based on model"""
        # All Claude 3 models support vision
        self.capabilities["vision"] = True

        # Claude 3.5 and newer models support video frames
        if any(
            m in self.model.lower() for m in ["claude-3.5", "claude-3-5", "claude-3-7", "opus", "sonnet"] # Added Opus and Sonnet for video frames too
        ):
            self.capabilities["video_frames"] = True
            self.capabilities["high_resolution"] = True # Assume newer models might have this
            self.capabilities["medical_imagery"] = True # Assume newer models might have this
        else:
            self.capabilities["video_frames"] = False
            self.capabilities["high_resolution"] = False
            self.capabilities["medical_imagery"] = False

        # Models supporting 'reasoning' or 'thinking' parameters
        # This includes Claude 3.7, and potentially Opus/Sonnet if API supports it.
        # The 'thinking' parameter is the one for extended thinking.
        # The 'reasoning' parameter is for different reasoning levels.
        # For now, "advanced_reasoning" capability can signify support for either.
        if any(m_part in self.model.lower() for m_part in ["claude-3-7", "claude-3-opus", "claude-3-sonnet"]):
            self.capabilities["advanced_reasoning"] = True
            # Set a default reasoning_level if not already set by a more specific config.
            # This can be overridden by the model config in ai-battle.py.
            if not hasattr(self, "reasoning_level") or self.reasoning_level is None:
                if "claude-3-7" in self.model.lower() or "opus" in self.model.lower():
                    self.reasoning_level = "high"
                elif "sonnet" in self.model.lower():
                    self.reasoning_level = "medium"
                else:
                    self.reasoning_level = "auto" # General default
                logger.debug(f"Set default reasoning level to '{self.reasoning_level}' for {self.model}")
        else:
            self.capabilities["advanced_reasoning"] = False
            self.reasoning_level = None # Ensure it's None if not capable

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
        if model_config is None:
            model_config = ModelConfig()

        self.role = role
        self.mode = mode
        history = history if history else [{"role": "user", "content": prompt}]

        # Get appropriate instructions
        history = history if history is not None else [] # Ensure history is a list
        current_instructions = self._determine_system_instructions(system_instruction, history, role, mode)

        # Build context-aware prompt
        # Determine final user prompt content
        final_prompt_content = self._determine_user_prompt_content(prompt, history, role, mode)

        # Format messages for Claude API
        messages = [
            {
                "role": "user" if msg["role"] == "human" else msg["role"], # Map human to user
                "content": msg["content"]
            }
            for msg in history # Iterate through the original history
            if msg.get("role") in ["user", "human", "assistant"] # Filter valid roles
        ]
        text_content = final_prompt_content
        # Handle file data
        if file_data:
            if isinstance(file_data, list) and file_data:
                # Process multiple files (images, video frames, etc.)
                message_content = []

                # Track image count to respect limits
                image_count = 0
                video_frames = []

                # Process images first, with special handling for medical imagery
                for file_item in file_data:
                    if isinstance(file_item, dict) and "type" in file_item:
                        # Process images with enhanced parameters
                        if (
                            file_item["type"] == "image"
                            and "base64" in file_item
                            and image_count < self.max_images_per_request
                        ):
                            # Add the image to message content (without metadata)
                            message_content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": file_item.get(
                                            "mime_type", "image/jpeg"
                                        ),
                                        "data": file_item["base64"],
                                    }
                                }
                            )
                            image_count += 1

                        # Process video frames
                        elif file_item["type"] == "video" and self.capabilities.get(
                            "video_frames", False
                        ):
                            # Extract key frames if available
                            if "key_frames" in file_item and file_item["key_frames"]:
                                for frame in file_item["key_frames"][
                                    : self.max_images_per_request - image_count
                                ]:
                                    video_frames.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/jpeg",
                                                "data": frame["base64"],
                                            },
                                            # Removed metadata field
                                        }
                                    )
                                image_count += len(video_frames)

                # Process text files
                for file_item in file_data:
                    if isinstance(file_item, dict) and "type" in file_item:
                        if (
                            file_item["type"] in ["text", "code"]
                            and "text_content" in file_item
                        ):
                            text_content = f"{text_content}\n\n[File content: {file_item.get('path', '')}]\n\n{file_item['text_content']}"
                for file_item in file_data:
                    if isinstance(file_item, dict) and "image" in file_item:
                        if (
                            file_item["type"] in ["text", "code"]
                            and "text_content" in file_item
                        ):
                            text_content = f"{text_content}\n\n[File content: {file_item.get('path', '')}]\n\n{file_item['text_content']}"

                # Add video frame analysis specific prompting if we have video frames
                if video_frames:
                    text_content += "\n\nI'm showing you key frames from a video. Please analyze these frames in sequence, noting any significant changes or patterns between frames."

                    # Add all video frames to message content
                    message_content.extend(video_frames)

                # Add the prompt text
                message_content.append({"type": "text", "text": final_prompt_content}) # Use final prompt content

                # Add to messages
                messages.append({"role": "user", "content": message_content})

                # Log for debugging
                logger.info(
                    f"Sending multimodal request with {image_count} images/frames to Claude"
                )

            else:
                # Handle single file with enhanced vision capabilities
                if isinstance(file_data, dict) and "type" in file_data:
                    if file_data["type"] == "image" and "base64" in file_data:
                        # Check if model supports vision
                        if self.capabilities.get("vision", False):
                            # Get file name and extension for additional context
                            file_path = file_data.get("path", "image")
                            file_name = os.path.basename(file_path)
                            file_ext = os.path.splitext(file_path)[1]

                            # Enhanced prompt that explicitly references the image
                            enhanced_prompt = f"{final_prompt_content}\n\nPlease analyze this {file_ext} image file: {file_name}"

                            # Format for Claude's multimodal API - without metadata which is causing errors
                            # Important: Claude expects text AFTER images for best results
                            message_content = [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": file_data.get(
                                            "mime_type", "image/jpeg"
                                        ),
                                        "data": file_data["base64"],
                                    }
                                    # Removed metadata field - not supported
                                },
                                {"type": "text", "text": enhanced_prompt},
                            ]

                            messages.append(
                                {"role": "user", "content": message_content}
                            )

                            logger.info(
                                f"Sending Image to Claude: {file_name} ({file_ext}) type={file_data.get('mime_type', 'image/jpeg')}"
                            )
                        else:
                            # Model doesn't support vision, use text only
                            logger.warning(
                                f"Model {self.model} doesn't support vision. Using text-only prompt."
                            )
                            messages.append({"role": "user", "content": prompt})
                    elif file_data["type"] == "video" and self.capabilities.get(
                        "video_frames", False
                    ):
                        # Process video for frame extraction
                        message_content = []

                        # Process key frames if available
                        if "key_frames" in file_data and file_data["key_frames"] or file_data["image"] and "base64" in file_data:
                            for i, frame in enumerate(
                                file_data["key_frames"][: self.max_images_per_request]
                            ):
                                message_content.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": frame["base64"],
                                        },
                                        # Removed metadata field
                                    }
                                )

                            # Add video-specific prompt
                            video_prompt = f"{final_prompt_content}\n\nI'm showing you {len(message_content)} key frames from a video. Please analyze these frames in sequence, noting any significant changes or patterns between frames." # Use final prompt content

                            message_content.append(
                                {"type": "text", "text": video_prompt}
                            )

                            messages.append(
                                {"role": "user", "content": message_content}
                            )

                            logger.debug(
                                f"Sending {len(message_content)-1} video frames to Claude"
                            )
                        else:
                            # No frames available
                            messages.append(
                                {
                                    "role": "user",
                                    "content": f"[Video file: {file_data.get('path', 'unknown')}]\n\n{prompt}",
                                }
                            )
                    elif (
                        file_data["type"] in ["text", "code"]
                        and "text_content" in file_data
                    ):
                        # Add text content with prompt
                        messages.append(
                            {
                                "role": "user",
                                "content": f"[File content: {file_data.get('path', '')}]\n\n{file_data['text_content']}\n\n{prompt}",
                            } # Removed comma here - Note: This case still uses raw prompt, might need adjustment if file content should combine with final_prompt_content
                        )
                    else:
                        # Standard prompt with reference to file
                        content = (
                            f"[File: {file_data.get('path', 'unknown')}]\n\n{prompt}"
                        )
                        messages.append({"role": "user", "content": content})
        else:
            # Standard prompt without file data
            messages.append({"role": "user", "content": final_prompt_content}) # Use final prompt content

        # Setup request parameters -- use structured injection if available
        iset = getattr(self, '_last_instruction_set', None)
        if iset and isinstance(iset, InstructionSet) and iset.persona:
            # Claude supports multi-block system messages for structured injection
            system_blocks = []
            if iset.persona:
                system_blocks.append({"type": "text", "text": iset.persona})
            if iset.template:
                system_blocks.append({"type": "text", "text": iset.template})
            if iset.constraints:
                system_blocks.append({"type": "text", "text": iset.constraints})
            # Interventions injected near user turn for recency bias
            if iset.interventions:
                messages.append({"role": "user", "content": f"[IMPORTANT DIRECTIVE]\n{iset.interventions}"})
                messages.append({"role": "assistant", "content": "Understood, I will follow these directives."})
            system_value = system_blocks
        else:
            system_value = current_instructions

        request_params = {
            "model": self.model,
            "system": system_value,
            "messages": messages,
            "max_tokens": model_config.max_tokens if model_config else MAX_TOKENS,
            "temperature": model_config.temperature if model_config else 0.8,
        }

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
                    # Make sure this is a direct API call, not through a library
                    # that might not support the parameters yet
                    client_has_thinking_param = False

                    # Try to check if the client explicitly supports this
                    try:
                        create_signature = inspect.signature(
                            self.client.messages.create
                        )
                        client_has_thinking_param = (
                            "thinking" in create_signature.parameters
                        )
                    except Exception:
                        # If we can't check, we'll try anyway
                        client_has_thinking_param = True

                    if client_has_thinking_param:
                        request_params["thinking"] = True
                        logger.debug("Added extended thinking parameter")

                        # Add budget_tokens if specified
                        if self.budget_tokens is not None:
                            # Ensure budget_tokens is less than max_tokens
                            max_tokens = request_params.get("max_tokens", MAX_TOKENS*2)
                            if self.budget_tokens < max_tokens:
                                request_params["budget_tokens"] = self.budget_tokens
                                logger.debug(
                                    f"Added budget_tokens: {self.budget_tokens}"
                                )
                            else:
                                # Use max_tokens - 100 as a fallback
                                safe_budget = max(1000, max_tokens - 100)
                                request_params["budget_tokens"] = safe_budget
                                logger.warning(
                                    f"budget_tokens ({self.budget_tokens}) must be less than max_tokens ({max_tokens}). Using {safe_budget} instead."
                                )
                    else:
                        logger.warning(
                            "Client library does not support extended thinking parameters"
                        )
                except Exception as e:
                    logger.warning(f"Error checking for extended thinking support: {e}")
                    # Continue without extended thinking parameters

        # Add stop sequences if provided
        if model_config and model_config.stop_sequences:
            request_params["stop_sequences"] = model_config.stop_sequences

        try:
            # --- Debug Logging ---
            logger.debug("--- Claude Request ---")
            logger.debug(f"Model: {request_params.get('model')}")
            logger.debug(f"System Instruction: {request_params.get('system')}")
            logger.debug(f"Messages: {request_params.get('messages')}")
            # Only attempt to use reasoning parameter with Claude 3.7
            response = None # Initialize response
            if "reasoning" in request_params:
                # Check if this is actually a 3.7 model
                if "claude-3.7" in self.model.lower():
                    # Keep the parameter for 3.7 models, but handle errors gracefully
                    try:
                        # Try with all advanced params first, but be prepared to fall back
                        logger.debug(
                            "Attempting to use advanced parameters for Claude 3.7"
                        )
                        response = self.client.messages.create(**request_params)
                        logger.debug(
                            "Successfully used advanced parameters with Claude 3.7"
                        )
                    except TypeError as e:
                        # Handle unsupported parameters
                        unsupported_params = []
                        retry_needed = False

                        # Check for reasoning parameter
                        if (
                            "unexpected keyword argument 'reasoning'" in str(e)
                            and "reasoning" in request_params
                        ):
                            reasoning_value = request_params.pop("reasoning")
                            unsupported_params.append(f"reasoning={reasoning_value}")
                            retry_needed = True

                        # Check for thinking parameter - try different error patterns
                        if (
                            "unexpected keyword argument 'thinking'" in str(e)
                            or "got an unexpected keyword argument 'thinking'" in str(e)
                        ) and "thinking" in request_params:
                            request_params.pop("thinking")
                            unsupported_params.append("thinking=True")
                            retry_needed = True

                        # Check for budget_tokens parameter - try different error patterns
                        if (
                            "unexpected keyword argument 'budget_tokens'" in str(e)
                            or "got an unexpected keyword argument 'budget_tokens'"
                            in str(e)
                        ) and "budget_tokens" in request_params:
                            budget = request_params.pop("budget_tokens")
                            unsupported_params.append(f"budget_tokens={budget}")
                            retry_needed = True

                        if retry_needed:
                            # Log warning about removed parameters
                            logger.warning(
                                f"Client library doesn't support parameters: {', '.join(unsupported_params)}. Removed and retrying."
                            )
                            response = self.client.messages.create(**request_params)
                        else:
                            # Re-raise other TypeError exceptions
                            raise
                else:
                    # For non-3.7 models, always remove the parameter
                    reasoning_level = request_params.pop("reasoning")
                    logger.debug(
                        f"Removed reasoning parameter for non-3.7 model: {self.model}"
                    )

            # Call Claude API with appropriate parameters
            # This will only execute if we didn't already make the call above
            if not locals().get("response"):
                logger.debug(
                    f"Sending request to Claude with parameters: {str(request_params)}"
                )
                response = self.client.messages.create(**request_params)

            logger.debug(f"Response received from Claude: {str(response.content)}")
            return response.content[0].text if response and response.content else "" # Extract text safely
        except Exception as e:
            logger.error(f"Error generating Claude response: {str(e)}")
            return f"Error generating Claude response: {str(e)}"


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
        api_key: str = None,
        mode: str = "ai-ai",
        domain: str = "General Knowledge",
        role: str = None,
        model: str = "gpt-4o",
    ):
        api_key = os.environ.get("OPENAI_API_KEY")
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.last_response_id = None # For Responses API context chaining
        self.use_responses_api = True  # Default to trying Responses API first

        # Models compatible with the experimental Responses API
        self.responses_compatible_models = [
            "o1", "o1-preview", "o3", "gpt-4o", "gpt-4o-mini",
            "gpt-4.1", "gpt-4.1-mini", "o4-mini", "gpt-4.1-nano"
        ]
        # Models supporting the 'reasoning' parameter in the Responses API
        self.reasoning_compatible_models = [
            "o1", "o1-preview", "o3", "o4-mini", "o4-mini-high"
        ]
        self.reasoning_level = "auto"  # Default reasoning level: "none", "low", "medium", "high", "auto"

        try:
            super().__init__(
                mode=mode, api_key=api_key, domain=domain, model=model, role=role
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise ValueError(f"Invalid OpenAI API key or model: {e}") from e

    def validate_connection(self) -> bool:
        """Test OpenAI API connection"""
        logger.info(f"{self.__class__.__name__} connection validated (API key assumed valid if client initialized).")
        return True

    def _prepare_responses_api_input(self, prompt_text: str, file_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """Prepares the 'input' parameter for the Responses API."""
        current_turn_input_items = [{"type": "input_text", "text": prompt_text}]
        if file_data:
            files_to_process = file_data if isinstance(file_data, list) else [file_data]
            for file_item in files_to_process:
                if isinstance(file_item, dict):
                    if file_item.get("type") == "image" and "base64" in file_item:
                        if self.capabilities.get("vision", True): # Assume vision if not specified
                            mime_type = file_item.get("mime_type", "image/jpeg")
                            current_turn_input_items.append({
                                "type": "input_image",
                                "image_url": {"url": f"data:{mime_type};base64,{file_item['base64']}"}
                            })
                            logger.info(f"Added image {file_item.get('path', 'unknown')} to Responses API input.")
                    elif file_item.get("type") in ["text", "code"] and "text_content" in file_item:
                        # Prepend text/code file content to the main text part for simplicity
                        current_turn_input_items[0]["text"] = (
                            f"[Content from file: {file_item.get('path', 'unknown')}]\n"
                            f"{file_item['text_content']}\n\n"
                            f"{current_turn_input_items[0]['text']}"
                        )
        return current_turn_input_items

    def _parse_responses_api_output(self, response: Any) -> str:
        """Parses the output from the Responses API, trying various known structures."""
        try:
            # Most recent or intended structure (hypothetical, based on previous complex parsing)
            if hasattr(response, "output") and response.output and isinstance(response.output, list):
                if response.output[0].content and isinstance(response.output[0].content, list):
                    if hasattr(response.output[0].content[0], "text"):
                        return response.output[0].content[0].text

            # Fallback to data structure (if it's a list of messages)
            if hasattr(response, "data") and response.data and isinstance(response.data, list):
                for msg_item in response.data:
                    if msg_item.role == "assistant" and hasattr(msg_item, "content") and isinstance(msg_item.content, list):
                        # Assuming content is a list of parts, extract text from text parts
                        text_parts = [part.text for part in msg_item.content if hasattr(part, "type") and part.type == "text" and hasattr(part, "text")]
                        if text_parts:
                            return "\n".join(text_parts)

            # Fallback to choices structure (similar to Chat Completions)
            if hasattr(response, "choices") and response.choices and isinstance(response.choices, list):
                if hasattr(response.choices[0], "message") and hasattr(response.choices[0].message, "content"):
                    return response.choices[0].message.content
                if hasattr(response.choices[0], "text"): # Older structure
                    return response.choices[0].text


            # Fallback if it's a raw JSON response that needs to be parsed
            if hasattr(response, "json") and callable(response.json):
                body = response.json()
                if "choices" in body and body["choices"] and "message" in body["choices"][0] and "content" in body["choices"][0]["message"]:
                    return body["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error during Responses API output parsing: {e}")

        logger.warning("Could not parse Responses API output using known structures. Returning raw string.")
        return str(response)


    def generate_response(
        self,
        prompt: str,
        system_instruction: str,
        history: List[Dict[str, str]],
        role: str = None,
        mode: str = None,
        file_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> str:
        """Generate response using OpenAI API, attempting Responses API first then Chat Completions."""
        if role: self.role = role
        if mode and ( mode == "human" or mode == "user" ): self.mode = "user"
        else:
            self.mode = "agent"
        current_model_config = model_config if model_config else ModelConfig()
        # History is not directly used by Responses API which uses previous_response_id
        # It will be used by the Chat Completions fallback path.

        current_instructions = self._determine_system_instructions(system_instruction, history, self.role, self.mode)
        final_prompt_content = self._determine_user_prompt_content(prompt, history, self.role, self.mode)

        model_supports_responses = any(m_name in self.model.lower() for m_name in self.responses_compatible_models)

        if self.use_responses_api and model_supports_responses:
            logger.debug(f"Attempting OpenAI Responses API with model {self.model}")

            current_turn_input_items = self._prepare_responses_api_input(final_prompt_content, file_data)

            responses_api_params = {
                "model": self.model,
                "input": current_turn_input_items,
                "instructions": current_instructions,
                "max_output_tokens": current_model_config.max_tokens,
                "temperature": current_model_config.temperature,
                "timeout": 90, # Consider making configurable via ModelConfig or client property
                "stream": False # Not using streaming for this implementation
            }
            if self.last_response_id:
                responses_api_params["previous_response_id"] = self.last_response_id

            if any(m_name in self.model.lower() for m_name in self.reasoning_compatible_models):
                reasoning_mapping = {"none": "low", "low": "low", "medium": "medium", "high": "high", "auto": "high"}
                reasoning_effort = reasoning_mapping.get(self.reasoning_level, "high") # Default "auto" to "high"
                responses_api_params["reasoning"] = {"effort": reasoning_effort, "summary": "detailed"}
                logger.debug(f"Using reasoning_effort='{reasoning_effort}' for model {self.model} (from reasoning_level='{self.reasoning_level}')")

            if current_model_config.seed is not None:
                responses_api_params["seed"] = current_model_config.seed
            if current_model_config.stop_sequences:
                responses_api_params["stop"] = current_model_config.stop_sequences

            try:
                response = self.client.responses.create(**responses_api_params)
                self.last_response_id = response.id
                return self._parse_responses_api_output(response)
            except Exception as e: # Catch OpenAI specific errors if possible, e.g. openai.APIError
                logger.warning(f"OpenAI Responses API failed for model {self.model} ({type(e).__name__}: {e}). Falling back to Chat Completions API.")
                self.use_responses_api = False
                # Fall through to Chat Completions API

        # --- Chat Completions API Path (Default or Fallback) ---
        logger.info(f"Using OpenAI Chat Completions API for model {self.model} (or as fallback).")

        chat_completions_messages = []
        iset = getattr(self, '_last_instruction_set', None)
        if iset and isinstance(iset, InstructionSet) and iset.persona:
            # OpenAI: multiple system/developer messages for structured injection
            if iset.persona:
                chat_completions_messages.append({"role": "system", "content": iset.persona})
            if iset.template:
                chat_completions_messages.append({"role": "system", "content": iset.template})
            if iset.constraints:
                chat_completions_messages.append({"role": "system", "content": iset.constraints})
        elif current_instructions:
            chat_completions_messages.append({"role": "system", "content": current_instructions})

        processed_history = history if history is not None else []
        for msg_data in processed_history:
            role_map = {"user": "user", "human": "user", "assistant": "assistant", "system": "system"}
            mapped_role = role_map.get(msg_data["role"].lower())
            if mapped_role:
                # If content is a list (potentially from a previous OpenAI vision response), pass it as is for user roles.
                if mapped_role == "user" and isinstance(msg_data.get("content"), list):
                    chat_completions_messages.append({"role": "user", "content": msg_data["content"]})
                elif isinstance(msg_data.get("content"), str): # Standard string content
                    chat_completions_messages.append({"role": mapped_role, "content": msg_data["content"]})

        user_message_cc_parts = [{"type": "text", "text": final_prompt_content}]
        if file_data:
            files_to_process = file_data if isinstance(file_data, list) else [file_data]
            for file_item in files_to_process:
                if isinstance(file_item, dict) and file_item.get("type") == "image" and "base64" in file_item:
                    if self.capabilities.get("vision", True):
                        mime_type = file_item.get("mime_type", "image/jpeg")
                        user_message_cc_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{file_item['base64']}"}
                        })
                elif isinstance(file_item, dict) and file_item.get("type") in ["text", "code"] and "text_content" in file_item:
                    user_message_cc_parts[0]["text"] = (
                        f"[Content from file: {file_item.get('path', 'unknown')}]\n"
                        f"{file_item['text_content']}\n\n"
                        f"{user_message_cc_parts[0]['text']}"
                    )

        # Inject interventions near user turn for recency bias (OpenAI)
        if iset and isinstance(iset, InstructionSet) and iset.interventions:
            chat_completions_messages.append({"role": "system", "content": f"[IMPORTANT DIRECTIVE]\n{iset.interventions}"})

        final_user_content_for_cc = user_message_cc_parts if len(user_message_cc_parts) > 1 or any(p["type"] == "image_url" for p in user_message_cc_parts) else final_prompt_content
        chat_completions_messages.append({"role": "user", "content": final_user_content_for_cc})

        request_params = {
            "model": self.model,
            "messages": chat_completions_messages,
            "temperature": current_model_config.temperature,
            "max_tokens": current_model_config.max_tokens,
            "timeout": 90, # Consider making configurable
            "stream": False
        }
        if current_model_config.stop_sequences:
            request_params["stop"] = current_model_config.stop_sequences
        if current_model_config.seed is not None:
            request_params["seed"] = current_model_config.seed

        logger.debug("--- OpenAI Chat Completions Request (Fallback/Default) ---")
        logger.debug(f"Model: {request_params.get('model')}")
        logger.debug(f"Messages count: {len(request_params.get('messages', []))}")

        try:
            response = self.client.chat.completions.create(**request_params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI Chat Completions API error: {e}")
            raise



class MLXClient(BaseClient):
    """Client for local MLX model interactions"""

    def __init__(
        self,
        mode: str,
        domain: str = "General knowledge",
        base_url: str = "http://localhost:9999",
        model: str = "mlx",
    ) -> None:
        super().__init__(mode=mode, api_key="dummy-key", domain=domain, model=model) # Add dummy key
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
    ) -> str:
        """Generate response using MLX through OpenAI-compatible endpoint"""
        if model_config is None:
            model_config = ModelConfig()

        # Format messages for OpenAI chat completions API
        messages = []

        history = history if history is not None else [] # Ensure history is a list
        current_instructions = self._determine_system_instructions(system_instruction, history, role, self.mode)
        final_prompt_content = self._determine_user_prompt_content(prompt, history, role, self.mode)

        if current_instructions:
            messages.append({"role": "system", "content": current_instructions}) # Add system instruction first

        if history:
            # Limit history to last 10 messages
            recent_history = history
            for msg in recent_history:
                if msg["role"] in ["user", "human", "moderator"]:
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] in ["assistant", "system"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": final_prompt_content}) # Use final prompt content

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
            logger.debug("--- MLX Request ---")
            logger.debug(f"URL: {self.base_url}/v1/chat/completions")
            # Note: System instruction is the first message in messages
            logger.debug(f"Messages: {messages}")
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={"messages": messages, "stream": False},
                headers={"Authorization": f"Bearer {self.api_key}"}, # Pass dummy key
                stream=False,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            try:
                partial_text = []
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        decoded = chunk.decode("utf-8", errors="ignore")
                        partial_text.append(decoded)
                return "".join(partial_text).strip()
            except Exception as inner_e:
                logger.error(
                    f"MLX generate_response error: {e}, chunk processing error: {inner_e}"
                )
                # Return an error message instead of None
                return f"Error generating response: {e}"
                return f"Error: {e}"



class OllamaClient(BaseClient):
    """Client for local Ollama model interactions"""

    # Initialize the client
    def __init__(
        self, mode: str, domain: str, role: str = None, model: str = None
    ):
        super().__init__(mode=mode, api_key="", domain=domain, model=model, role=role)
        self.base_url = (
            "http://localhost:11434"  # Not directly used with ollama library
        )
        self.client = Client(
            host=self.base_url
        )  # Use synchronous Client instead of AsyncClient

        # Reasoning / thinking configuration
        # Follows the same pattern as ClaudeClient and OpenAIClient
        self.reasoning_level = "none"  # Options: "none", "low", "medium", "high", "auto"
        self.extended_thinking = False  # Whether think parameter is enabled
        self.keep_alive = None  # Optional: keep_alive parameter for ollama (e.g., "10m", "1h")
        self.num_ctx = None  # Optional: override context window size (e.g., 131072 for gpt-oss:120b)

        # Models known to support the think parameter
        self.thinking_capable_keywords = [
            "qwen3", "deepseek-r1", "gpt-oss",
            "phi4-reasoning", "granite-reasoning",
        ]

        # Vision model keywords (consolidated source of truth)
        self.vision_keywords = [
            "llava", "bakllava", "moondream", "gemma3",
            "llava-phi3", "vision", "mistral-medium",
        ]

        # Update capabilities based on model name
        self._update_capabilities()

    def set_extended_thinking(self, enabled: bool, budget_tokens: Optional[int] = None):
        """
        Enable or disable thinking mode for Ollama models that support it.

        Mirrors ClaudeClient.set_extended_thinking for API consistency.
        The budget_tokens parameter is accepted for interface compatibility but is
        not used by Ollama (the think parameter controls reasoning at the model level).

        Args:
            enabled: Whether to enable the think parameter.
            budget_tokens: Ignored for Ollama, accepted for interface compatibility.
        """
        self.extended_thinking = enabled
        if enabled and self.reasoning_level == "none":
            self.reasoning_level = "auto"  # Upgrade from none when thinking is enabled
        logger.info(
            f"Set extended thinking={enabled} for Ollama model {self.model}"
        )

    def _update_capabilities(self):
        """Update capability flags based on model name keywords."""
        model_lower = self.model.lower() if self.model else ""

        # Vision capability
        if any(kw in model_lower for kw in self.vision_keywords):
            self.capabilities["vision"] = True

        # Advanced reasoning capability
        if any(kw in model_lower for kw in self.thinking_capable_keywords):
            self.capabilities["advanced_reasoning"] = True

    def test_connection(self) -> None:
        """Test Ollama connection"""
        try:
            self.client.list()
            logger.info("Ollama connection test successful")
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            raise

    def generate_response(
        self,
        prompt: str,
        system_instruction: str = None,
        history: List[Dict[str, str]] = None,
        file_data: Dict[str, Any] = None,
        model_config: Optional[ModelConfig] = None,
        mode: str = None,
        role: str = None,
    ) -> str:
        """Generate a response from your local Ollama model."""
        if role:
            self.role = role
        if mode:
            self.mode = mode

        history = history if history is not None else []

        # Determine instructions and final prompt content using BaseClient helpers
        current_instructions = self._determine_system_instructions(system_instruction, history, role, mode)
        final_prompt_content = self._determine_user_prompt_content(prompt, history, role, mode)
        messages = []
        messages.append({"role": "user", "content": final_prompt_content})

        if current_instructions:
            messages.append({"role": "system", "content": current_instructions})

        ##messages.append({"role": "user", "content": final_prompt_content})

        # Add conversation history, mapping roles for Ollama compatibility
        if history:
            for msg in history:
                if msg.get("role") in ["user", "assistant", "human"]:
                    msg_role = "user" if msg["role"] == "human" else msg["role"]
                    messages.append({"role": msg_role, "content": msg["content"]})

        # Use capability flag for vision detection (set in _update_capabilities)
        is_vision_model = self.capabilities.get("vision", False)

        # Handle file data for vision-capable Ollama models
        if is_vision_model and file_data:
            image_base64_list = []
            if isinstance(file_data, list) and file_data:
                for file_item in file_data:
                    if isinstance(file_item, dict) and "type" in file_item:
                        if file_item["type"] == "image" and "base64" in file_item:
                            image_base64_list.append(file_item["base64"])
                        elif file_item["type"] == "video" and "key_frames" in file_item and file_item["key_frames"]:
                            for frame in file_item["key_frames"]:
                                if "base64" in frame:
                                    image_base64_list.append(frame["base64"])

            elif isinstance(file_data, dict) and "type" in file_data:
                if file_data["type"] == "image" and "base64" in file_data:
                    image_base64_list.append(file_data["base64"])
                elif file_data["type"] == "video" and "key_frames" in file_data and file_data["key_frames"]:
                    for frame in file_data["key_frames"]:
                        if "base64" in frame:
                            image_base64_list.append(frame["base64"])

            if image_base64_list:
                if messages and messages[-1]["role"] == "user":
                    if "images" in messages[-1] and isinstance(messages[-1]["images"], list):
                        messages[-1]["images"].extend(image_base64_list)
                    else:
                        messages[-1]["images"] = image_base64_list
                else:
                    messages.append({
                        "role": "user",
                        "content": final_prompt_content,
                        "images": image_base64_list
                    })
                logger.info(f"Added {len(image_base64_list)} images to OllamaClient request.")

        # Determine think parameter based on reasoning_level and model capability
        think_value = None
        if self.extended_thinking or self.capabilities.get("advanced_reasoning", False):
            think_mapping = {
                "none": False,
                "low": "low",
                "medium": "medium",
                "high": True,
                "auto": True,
            }
            think_value = think_mapping.get(self.reasoning_level, True)

        # Build typed Options
        is_gpt_oss = "gpt" in self.model.lower()

        # When thinking is enabled, some models require temperature to be unset
        effective_temperature = None
        if think_value is False or think_value is None:
            effective_temperature = (
                model_config.temperature
                if model_config and model_config.temperature is not None
                else (1.0 if is_gpt_oss else 0.4)
            )

        # Use self.num_ctx if set (e.g., 131072 for gpt-oss:120b), otherwise model-specific defaults
        effective_num_ctx = self.num_ctx if self.num_ctx else (32768 if is_gpt_oss else 16384)

        chat_options = Options(
            num_ctx=effective_num_ctx,
            num_predict=(
                model_config.max_tokens
                if model_config and model_config.max_tokens
                else (3072 if is_gpt_oss else 2048)
            ),
            temperature=effective_temperature,
            seed=(
                model_config.seed
                if model_config and model_config.seed is not None
                else random.randint(0, 1000000)
            ),
            stop=(
                model_config.stop_sequences
                if model_config and model_config.stop_sequences
                else None
            ),
            num_batch=1024 if is_gpt_oss else 1024,
            top_k=30 if is_gpt_oss else 25,
            use_mmap=True,
        )

        try:
            logger.debug("--- Ollama Request ---")
            logger.debug(f"Model: {self.model}, think={think_value}, num_ctx={effective_num_ctx}")
            logger.debug(f"Messages: {len(messages)} messages")

            # Build chat kwargs - only include think/keep_alive when set
            chat_kwargs = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": chat_options,
            }

            if think_value is not None:
                chat_kwargs["think"] = think_value

            if self.keep_alive is not None:
                chat_kwargs["keep_alive"] = self.keep_alive

            response: ChatResponse = self.client.chat(**chat_kwargs)

            # Extract thinking content if present (SDK 0.6.x: response.message.thinking)
            thinking_content = getattr(response.message, 'thinking', None)  # pylint: disable=no-member
            main_content = response.message.content or ""  # pylint: disable=no-member

            # Surface thinking content wrapped in <thinking> tags
            # The UI's render_message_with_thinking() already parses these
            if thinking_content:
                logger.debug(f"Ollama model returned thinking content ({len(thinking_content)} chars)")
                return f"<thinking>{thinking_content}</thinking>\n\n{main_content}"
            return main_content

        except Exception as e:
            logger.error(f"Ollama generate_response error: {e}")
            try:
                if response and hasattr(response, 'message') and response.message:
                    return response.message.content or f"Error generating response: {e}"
            except (NameError, AttributeError):
                pass
            return f"Error generating response: {e}"
