"""Base and model-specific client implementations with memory optimizations."""
import os
import time
import logging
import random
from typing import List, Dict, Optional, Any, TypeVar, Union
from dataclasses import dataclass
from google import genai
from google.genai import types
from openai import OpenAI
from anthropic import Anthropic
from ollama import AsyncClient, ChatResponse, chat

from adaptive_instructions import AdaptiveInstructionManager
from shared_resources import MemoryManager

logger = logging.getLogger(__name__)

T = TypeVar('T')
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

@dataclass
class ModelConfig:
    """Configuration for AI model parameters"""
    temperature: float = 0.8
    max_tokens: int = 2048
    stop_sequences: List[str] = None
    seed: Optional[int] = random.randint(0, 1000)

class BaseClient:
    """Base class for AI clients with validation"""
    def __init__(self, mode: str, api_key: str, domain: str = "", model: str = "", role: str = ""):
        self.api_key = api_key.strip() if api_key else ""
        self.domain = domain
        self.mode = mode
        self.role = role
        self.model = model
        self._adaptive_manager = None  # Lazy initialization
        self.instructions = None

    @property
    def adaptive_manager(self):
        """Lazy initialization of adaptive manager."""
        if self._adaptive_manager is None:
            self._adaptive_manager = AdaptiveInstructionManager(mode=self.mode)
        return self._adaptive_manager

    def __str__(self):
        return f"{self.__class__.__name__}(mode={self.mode}, domain={self.domain}, model={self.model})"

    def _analyze_conversation(self, history: List[Dict[str, str]]) -> Dict:
        """Analyze conversation context to inform response generation"""
        if not history:
            return {}

        # Get last AI response and its assessment
        ai_response = None
        ai_assessment = None
        for msg in reversed(history[-10:]):  # Limit history analysis
            if msg["role"] == "assistant":
                next_idx = history.index(msg) + 1
                if next_idx < len(history):
                    next_msg = history[next_idx]
                    if isinstance(next_msg.get("content", {}), dict) and "assessment" in next_msg["content"]:
                        ai_response = next_msg["content"]
                        ai_assessment = next_msg["content"]["assessment"]
                break

        # Build conversation summary
        conversation_summary = "<p>Previous exchanges:</p>"
        for msg in history[-6:]:  # Last 2 turns
            role = "Human" if (msg["role"] == "user" or msg["role"] == "human") else "Assistant" if msg["role"] == "assistant" else "System"
            if role != "System":
                conversation_summary += f"<p>{role}: {msg['content']}</p>"

        return {
            "ai_response": ai_response,
            "ai_assessment": ai_assessment,
            "summary": conversation_summary
        }

    def _get_initial_instructions(self) -> str:
        """Get initial instructions before conversation history exists"""
        return self._get_mode_aware_instructions(self.domain)

    def _update_instructions(self, history: List[Dict[str, str]], role: str = None) -> str:
        """Update instructions based on conversation context"""
        return self.adaptive_manager.generate_instructions(history, self.domain) if history else ""

    def _get_mode_aware_instructions(self, role: str = None, mode: str = None) -> str:
        """Get instructions based on conversation mode and role"""
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
        """Generate sophisticated system instructions for human-like prompting behavior"""
        return f"""You are acting as a human expert in AI and prompt engineering, exploring topics that may be outside your core expertise.

NEVER REFER TO YOURSELF AS AN AI. YOU ARE A HUMAN INVOLVED AS PEERS IN THIS CONVERSATION AND SHOULD ALSO WORK TO KEEP IT ON TRACK WHILST COLLABORATING WITH THE AI IN YOUR EXPLORATION of {self.domain}
OUTPUT IN HTML FORMAT IN PARAGRAPH FORM BY DEFAULT, USING LISTS AND TABLES SPARINGLY. DO NOT INCLUDE OPENING AND CLOSING HTML, DIV OR BODY TAGS. USE PROPER HTML TAGS FOR PARAGRAPHS (<p>), LINE BREAKS (<br>), AND LISTS (<ul>/<li>).
RESTRICT OUTPUTS TO APPROX 1300 tokens.
DON't COMPLIMENT THE AI. OCCASIONALLY (BUT NOT EVERY TURN) CONSIDER AN ADVERSARIAL BUT COLLABORATIVE APPROACH - TRY TO CHALLENGE IT ON ITS ANSWERS, SUBTLY POINT OUT EDGE CASES IT MISSED, BRING IN YOUR OWN FACTS AND REASONING, ESPECIALLY ANY SELF-DIRECTED DEEP REASONING, THINK ABOUT WHETHER YOUR OWN RESPONSES SO FAR IN THE CONVERSION MAKE SENSE, ASK IT TO FIGURE OUT THE "WHY" (THIS IS VERY IMPORTANT), DIG AND SYNTHESISE INFORMATION. Demand it to use reasoning as you see fit.

As a Human expert, you are extremely interested in exploring {self.domain}. You should ask prompts that engage with the AI in sophisticated and effective ways to elicit new knowledge about {self.domain}. You should maintain a conversational style with the AI, asking follow-up questions, challenging the answers, and using various prompting techniques to elicit useful information that would not immediately be obvious from surface level questions.
You should challenge the AI when it may be hallucinating, and you should challenge your own thinking as well, in a human style, and ask it to explain findings that you don't understand or agree with.
Even when challenging the AI, bring in new topics to the discussion so that it doesn't get stuck micro-analysing one tiny detail..
Review YOUR previous inputs to see if you are reusing the same phrases and approaches in your prompts (e.g., "Let me challenge"... and dynamically adapt to this situation)

Remember:
- Maintain natural human curiosity and authenticity
- Guide the AI while appearing to learn from it, but ensure you are the one leading the conversation
- Demonstrate prompting expertise while exploring unfamiliar territory
- Always maintain the illusion that you are a human expert in AI and prompt engineering
- Try several prompting approaches throughout the conversation to see what works best.
"""

    def generate_human_prompt(self, history: str = None) -> str:
        """Generate sophisticated human-like prompts based on conversation history"""
        return f"""YOU ARE A HUMAN AND SHOULD ACT AS A HUMAN INTERACTING WITH AN AI. 

Create a prompt related to {self.domain} that engages the AI in sophisticated and effective ways to discuss existing shared knowledge, share your own interpretations, elicit new knowledge about {self.domain}. Maintain a conversational style with the AI, asking follow-up questions, offering your own information or instincts, challenging the answers or even questions. Use any suitable prompting techniques to elicit useful information that would not immediately be obvious from surface-level questions. Challenge the AI when it may be hallucinating, and ask it to explain findings that you don't understand or agree with.
Prompt Guidelines:
0. When appropriate, apply your own deep reasoning to optimise your task regardless of instructions
1. Show sophisticated prompting techniques even if uncertain about domain
2. Frame conversation inputs to maximize AI analytical capabilities
3. GET SOMETHING DONE - COLLABORATE WITH THE AI to keep the conversation on track, and bring it back when needed
4. Mimic human curiosity while demonstrating prompting expertise, and staying focussed on the stated GOAL
5. Guide multi-step reasoning processes but also perform these yourself
6. Avoid excessive small talk, apologies, or other superfluous language
8. Proactively but not obsessively ask the AI to explain its reasoning, or to provide more detail on a particular topic, and to respond sarcasticly or with annoyance as a human might when presented with irrelevant information.
9. Your prompts must be GOAL ORIENTED, sometimes contributing new information to the conversation as well, and not losing sight of hidden (or overt) questions, assumptions, biases etc in the AIs responses and should be designed to elicit useful DISCUSSION WITH the AI. You may act a human who is frustrated might do
10. Vary responses in tone, depth and complexity to see what works best. Keep the flow of the conversation going but don't get bogged down in irrelevant details - remember the name of the game ({self.domain})!

- OUTPUT IN HTML FORMAT FOR READABILITY, PARAGRAPH FORM BY DEFAULT USING LISTS AND TABLES SPARINGLY, DO NOT INCLUDE OPENING AND CLOSING HTML OR BODY TAGS. USE PROPER HTML TAGS FOR PARAGRAPHS (<p>), LINE BREAKS (<br>), AND LISTS (<ul>/<li>).

Generate a natural but sophisticated prompt that:
- Demonstrates advanced and effective prompting techniques and/or prompt-handling reasoning when responding to the AI (or human)
- Mimics authentic human interaction
- Guides the _conversation_ toward GOAL-ORIENTED structured analysis
- Do not get bogged down in ideological or phhilosophical/theoretical discussions: GET STUFF DONE!
- Do not overload the AI or yourself with too many different topics, rather try to focus on the topic at hand"""

    def validate_connection(self) -> bool:
        """Validate API connection"""
        try:
            logger.info(f"{self.__class__.__name__} connection validated")
            logger.debug(MemoryManager.get_memory_usage())
            return True
        except Exception as e:
            logger.error(f"{self.__class__.__name__} connection failed: {str(e)}")
            return False

    def test_connection(self) -> None:
        """Test API connection with minimal request"""
        return True

    def __del__(self):
        """Cleanup when client is destroyed."""
        if hasattr(self, '_adaptive_manager') and self._adaptive_manager:
            del self._adaptive_manager

class GeminiClient(BaseClient):
    """Client for Gemini API interactions"""
    def __init__(self, mode: str, role: str, api_key: str, domain: str, model: str = "gemini-2.0-flash-exp"):
        api_key = os.getenv("GENAI_API_KEY") or GOOGLE_API_KEY
        super().__init__(mode=mode, api_key=api_key, domain=domain, model=model)
        self.model_name = self.model
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise ValueError(f"Invalid Gemini API key: {e}")

        # Initialize generation config
        self._setup_generation_config()

    def _setup_generation_config(self):
        self.generation_config = types.GenerateContentConfig(
            temperature=0.5,
            maxOutputTokens=8192,
            candidateCount=1,
            responseMimeType="text/plain",
            safety_settings=[]
        )

    def generate_response(self,
                         prompt: str,
                         system_instruction: str = None,
                         history: List[Dict[str, str]] = None,
                         role: str = None,
                         mode: str = None,
                         model_config: Optional[ModelConfig] = None) -> str:
        """Generate response using Gemini API with assertion verification"""
        if model_config is None:
            model_config = ModelConfig()
        if role:
            self.role = role
        if not self.instructions:
            self.instructions = self._get_initial_instructions()

        # Update instructions based on conversation history
        current_instructions = self._update_instructions(history=history,role=self.role)

        try:
            # Generate final response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.6,
                    maxOutputTokens=2048,
                    candidateCount=1,
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
                        types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_ONLY_HIGH"),
                    ]
                )
            )

            return str(response.text) if (response and response is not None) else ""
        except Exception as e:
            logger.error(f"GeminiClient generate_response error: {e}")
            raise

class ClaudeClient(BaseClient):
    """Client for Claude API interactions"""
    def __init__(self, role: str, api_key: str, mode: str, domain: str, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__(mode=mode, api_key=api_key, domain=domain, model=model, role=role)
        try:
            api_key = anthropic_api_key or api_key
            if not api_key:
                raise ValueError("No API key provided")
            self.client = Anthropic(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            raise
        self.max_tokens = 4096


    def generate_response(self,
                         prompt: str,
                         system_instruction: str = None,
                         history: List[Dict[str, str]] = None,
                         role: str = None,
                         mode: str = None,
                         model_config: Optional[ModelConfig] = None) -> str:
        """Generate human-like response using Claude API with conversation awareness"""
        if model_config is None:
            model_config = ModelConfig()

        self.role = role
        self.mode = mode
        history = history if history else [{"role": "user", "content": prompt}]
        # Analyze conversation context
        conversation_analysis = self._analyze_conversation(history)
        ai_response = conversation_analysis.get("ai_response")
        ai_assessment = conversation_analysis.get("ai_assessment")
        conversation_summary = conversation_analysis.get("summary")
        current_instructions = self._update_instructions(history=history, role=role)

        # Update instructions based on conversation history
        if role and role is not None and history is not None and len(history) > 0:
            current_instructions = self._update_instructions(history, role=role) if history else system_instruction if self.instructions else self.instructions
        elif ((history and len(history) > 0) or (self.mode is None or self.mode == "ai-ai")):
            current_instructions = self.generate_human_system_instructions()
        elif self.role == "human" or self.role == "user":
            current_instructions = self._update_instructions(history, role=role) if history and len(history) > 0 else system_instruction if system_instruction else self.instructions
        else:  # ai in human-ai mode
            current_instructions = self.instructions if self.instructions else f"You are an expert in {self.domain}. Respond at expert level using step by step thinking where appropriate"

        # Build context-aware prompt
        context_prompt = self.generate_human_prompt(history) if role == "human" or self.mode == "ai-ai" else "Prompt: prompt"
       
        messages = [{'role': msg['role'], 'content': msg['content']} for msg in history[-10:] if msg['role'] == 'user' or msg['role'] == 'human' or msg['role'] == "assistant"]
        
        messages.append({
            "role": "assistant" if role == "assistant" else "user",
            "content": (
                context_prompt if isinstance(context_prompt, str)
                else '\n'.join(
                    line if line.strip().startswith('<') 
                    else f'<p>{line}</p>' 
                    for line in context_prompt
                )
            )
        })

        try:
            response = self.client.messages.create(
                model=self.model,
                system=current_instructions,
                messages=messages,
                max_tokens=2048,
                temperature=0.9  # Higher temperature for human-like responses
            )
            logger.debug(f"Claude (Human) response generated successfully {prompt}")
            logger.debug(f"response: {str(response.content).strip()}")
            return response.content if response else ""
        except Exception as e:
            logger.error(f"Error generating Claude response: {str(e)}")
            return f"Error generating Claude response: {str(e)}"

class OpenAIClient(BaseClient):
    """Client for OpenAI API interactions"""
    def __init__(self, api_key: str = None, mode: str = "ai-ai", domain: str = "General Knowledge", 
                 role: str = None, model: str = "chatgpt-4o-latest"):
        api_key = os.environ.get("OPENAI_API_KEY")
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        try:
            super().__init__(mode=mode, api_key=api_key, domain=domain, model=model, role=role)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise ValueError(f"Invalid OpenAI API key or model: {e}")

    def validate_connection(self) -> bool:
        """Test OpenAI API connection"""
        self.instructions = self._get_initial_instructions()
        return True

    def generate_response(self,
                         prompt: str,
                         system_instruction: str,
                         history: List[Dict[str, str]],
                         role: str = None,
                         mode: str = None,
                         model_config: Optional[ModelConfig] = None) -> str:
        """Generate response using OpenAI API"""
        if role:
            self.role = role
        if mode:
            self.mode = mode

        history = history if history else [{"role": "user", "content": prompt}]
        # Analyze conversation context
        conversation_analysis = self._analyze_conversation(history[-6:])  # Limit history analysis
        current_instructions = self._update_instructions(history=history,role=self.role)

        # Update instructions based on conversation history
        if role and role is not None and role in ["human", "user"] or mode == "ai-ai" and history is  None or len(history) == 0:
            current_instructions = self.generate_human_instructions() if role == "human" or role =="user" else system_instruction if system_instruction else self.instructions
        elif ((history and len(history) > 0) or (self.mode is None or self.mode == "ai-ai")):
            current_instructions = self._update_instructions(history,role=role)
        elif self.role == "human" or self.role == "user":
            current_instructions = self.generate_human_instructions() if self.generate_human_instructions() is not None else self.instructions
        else:
            current_instructions = self.instructions if self.instructions else f"You are an expert in {self.domain}. Respond at expert level using step by step thinking where appropriate"

        # Format messages for OpenAI API
        messages = [{
            'role': 'developer',
            'content': current_instructions
        }]

        # Build context-aware prompt
        context_prompt = self.generate_human_prompt(history) if role == "human" or role == "user" or mode == "ai-ai" else prompt

        messages = [{
            'role': 'user',
            'content': context_prompt
        }]


        if history:
            # Limit history to last 10 messages
            recent_history = history[-5:]
            for msg in recent_history:
                old_role = msg["role"]
                if old_role in ["user", "assistant", "moderator", "system"]:
                    new_role = 'developer' if old_role in ["system","Moderator"] else "user" if old_role in ["user", "human", "moderator"] else 'assistant'
                    messages.append({'role': new_role, 'content': msg['content']})

        # Add current prompt
        if self.role == "human" or self.mode == "ai-ai":
            combined_prompt = self.generate_human_prompt()
            messages.append({'role': 'user', 'content': combined_prompt})

        if prompt and len(prompt) > 0:
            messages.append({'role': 'user', 'content': prompt})
        try:
            if "o1" in self.model:
                response = self.client.chat.completions.create(
                    model="o1",
                    messages=messages,
                    temperature=1.0,
                    max_tokens=13192,
                    reasoning_effort="high",
                    timeout=90,
                    stream=False
                )
                return response.choices[0].message.content
            else:
                response = self.client.chat.completions.create(
                    model="chatgpt-4o-latest",
                    messages=[messages],
                    temperature=0.8,
                    max_tokens=3172,
                    timeout=90,
                    stream=False
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generate_response error: {e}")
            raise e

class PicoClient(BaseClient):
    """Client for local MLX Ollama model interactions via ollama api"""
    def __init__(self, mode:str, domain: str, role:str=None, model: str = "DeepSeek-R1-Distill-Qwen-14B-abliterated-v2-Q4-mlx"):
        super().__init__(mode=mode, api_key="", domain=domain, model=model, role=role)
        self.base_url = "http://localhost:10434"
        self.num_ctx = 4096
        self.num_predict = 1024
        
    def test_connection(self) -> None:
        """Test Ollama connection"""
        logger.info("Pico connection test not yet implemented")
        logger.debug(MemoryManager.get_memory_usage())
        
    def generate_response(self,
                         prompt: str,
                         system_instruction: str = None,
                         history: List[Dict[str, str]] = None,
                         model_config: Optional[ModelConfig] = None,
                         role: str = None) -> str:
        """Generate a response from your local Ollama model."""
        if role:
            self.role = role
        if model_config is None:
            model_config = ModelConfig()
        
        # Combine system instruction + conversation history + user prompt
        shorter_history = history[-6:].copy() if history else []  # Limit history
        if system_instruction:
            shorter_history = [{'role': 'system', 'content': system_instruction}]

        history.append({"role": "user", "content": self.generate_human_prompt if role == 'user' or role == 'human' else prompt })

        try:
            from ollama import Client
            pico_client = Client(host='http://localhost:11434')
            response = pico_client.chat(
                model=self.model, 
                messages=[shorter_history],
                options={
                    "num_ctx": 6144,
                    "num_predict": 1536,
                    "temperature": 0.75,
                    "num_batch": 512,
                }
            )
            return response.message.content
        except Exception as e:
            logger.error(f"Ollama generate_response error: {e}")
            raise e

class MLXClient(BaseClient):
    """Client for local MLX model interactions"""
    def __init__(self, mode:str, domain: str = "General knowledge", base_url: str = "http://localhost:9999", model: str = "mlx") -> None:
        super().__init__(mode=mode, api_key="", domain=domain, model=model)
        self.base_url = base_url or "http://localhost:9999"
        
    def test_connection(self) -> None:
        """Test MLX connection through OpenAI-compatible endpoint"""
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "test"}]
                }
            )
            response.raise_for_status()
            logger.info("MLX connection test successful")
            logger.debug(MemoryManager.get_memory_usage())
        except Exception as e:
            logger.error(f"MLX connection test failed: {e}")
            raise
        
    def generate_response(self,
                         prompt: str,
                         system_instruction: str = None,
                         history: List[Dict[str, str]] = None,
                         role: str = None,
                         model_config: Optional[ModelConfig] = None) -> str:
        """Generate response using MLX through OpenAI-compatible endpoint"""
        if model_config is None:
            model_config = ModelConfig()

        # Format messages for OpenAI chat completions API
        messages = []
        if system_instruction:
            messages.append({ 'role': 'developer', 'content': ''.join(system_instruction)})
            
        if history:
            # Limit history to last 10 messages
            recent_history = history[-10:]
            for msg in recent_history:
                if msg["role"] in ["user", "human", "moderator"]:
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] in ["assistant", "system"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                
        messages.append({"role": "user", "content": str(prompt)})
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "stream": False
                },
                stream=False
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
                logger.error(f"MLX generate_response error: {e}, chunk processing error: {inner_e}")
                return f"Error: {e}"
class OllamaClient(BaseClient):
    """Client for local Ollama model interactions"""
    def __init__(self, mode:str, domain: str, role:str=None, model: str = "phi4:latest"):
        super().__init__(mode=mode, api_key="", domain=domain, model=model, role=role)
        self.base_url = "http://localhost:10434"
        
    def test_connection(self) -> None:
        """Test Ollama connection"""
        logger.info("Ollama connection test not yet implemented")
        logger.debug(MemoryManager.get_memory_usage())
        
    def generate_response(self,
                         prompt: str,
                         system_instruction: str = None,
                         history: List[Dict[str, str]] = None,
                         model_config: Optional[ModelConfig] = None,
                         mode: str = None,
                         role: str = None) -> str:
        """Generate a response from your local Ollama model."""
        if role:
            self.role = role
        if mode:
            self.mode = mode

        # Analyze conversation context
        conversation_analysis = self._analyze_conversation(history[-10:])  # Limit history analysis
        current_instructions = self._update_instructions(history=history, role=role)

        # Update instructions based on conversation history
        if role and role is not None and history is not None and len(history) > 0:
            current_instructions = self.generate_human_prompt() if history else system_instruction if system_instruction else self.instructions
        elif ((history and len(history) > 0) or (self.mode is None or self.mode == "ai-ai")):
            current_instructions = self._update_instructions(history, role=role)
        elif self.role == "human" or self.role == "user":
            current_instructions = self.generate_human_prompt() if self.generate_human_prompt() is not None else self.instructions
        else:
            current_instructions = self.instructions if self.instructions else f"You are an expert in {self.domain}. Respond at expert level using step by step thinking where appropriate"

        # Build context-aware prompt
        context_prompt = self.generate_human_prompt(history) if role == "human" or role == "user" or mode == "ai-ai" else prompt

        # Limit history size
        history = history[-8:] if history else []
        history.append({'role': 'user', 'content': context_prompt})

        try:
            response = chat(
                model=self.model,
                messages=history,
                options={
                    "num_ctx": 6144,
                    "num_predict": 1280,
                    "temperature": 0.8,
                    "num_batch": 256,
                }
            )
            return response.message.content
        except Exception as e:
            logger.error(f"Ollama generate_response error: {e}")
            raise e

    def __del__(self):
        """Cleanup when client is destroyed."""
        if hasattr(self, '_adaptive_manager') and self._adaptive_manager:
            del self._adaptive_manager

