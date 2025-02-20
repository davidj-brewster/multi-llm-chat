"""Base and model-specific client implementations"""
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
        self.adaptive_manager = AdaptiveInstructionManager(mode=self.mode)
        self.instructions = None

    def __str__(self):
        return f"{self.__class__.__name__}(mode={self.mode}, domain={self.domain}, model={self.model})"

    def _analyze_conversation(self, history: List[Dict[str, str]]) -> Dict:
        """Analyze conversation context to inform response generation"""
        if not history:
            return {}

        # Get last AI response and its assessment
        ai_response = None
        ai_assessment = None
        for msg in reversed(history):
            if msg["role"] == "assistant":
                next_idx = history.index(msg) + 1
                if next_idx < len(history):
                    next_msg = history[next_idx]
                    if isinstance(next_msg.get("content", {}), dict) and "assessment" in next_msg["content"]:
                        ai_response = next_msg["content"]
                        ai_assessment = next_msg["content"]["assessment"]
                break

        # Build conversation summary
        conversation_summary = "Previous exchanges:</p>"
        for msg in history[-6:]:  # Last 2 turns
            role = "Human" if (msg["role"] == "user" or msg["role"] == "human") else "Assistant" if msg["role"] == "assistant" else "System"
            if role != "System":
                conversation_summary += f"{role}: {msg['content']}</p>"

        return {
            "ai_response": ai_response,
            "ai_assessment": ai_assessment,
            "summary": conversation_summary
        }

    def _get_initial_instructions(self) -> str:
        """Get initial instructions before conversation history exists"""
        if self.adaptive_manager is None:
            self.adaptive_manager = AdaptiveInstructionManager(mode=self.mode)
        return self._get_mode_aware_instructions(self.domain)

    def _update_instructions(self, history: List[Dict[str, str]], role=None) -> str:
        """Update instructions based on conversation context"""
        if self.adaptive_manager is None:
            self.adaptive_manager = AdaptiveInstructionManager(mode=self.mode)
        return self.adaptive_manager.generate_instructions(history, self.domain)

    def _get_mode_aware_instructions(self, role: str = None, mode: str = None) -> str:
        """Get instructions based on conversation mode and role"""
        human_instructions = ""
        if role and role is not None:
            if role == "human":
                human_instructions = self.generate_human_prompt()
                return human_instructions
        if self.mode and self.mode == "ai-ai":
            human_instructions = self.generate_human_prompt()
            return human_instructions
        else:  # if self.mode == "human-aiai":
            if role == "user" or role == "human":
                human_instructions = self.generate_human_prompt()
                return human_instructions
            else:
                return "You are an AI assistant interacting with a human."
        return ""

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

- OUTPUT IN HTML FORMAT FOR READABILITY, PARAGRAPH FORM BY DEFAULT USING LISTS AND TABLES SPARINGLY, DO NOT INCLUDE OPENING AND CLOSING HTML OR BODY TAGS

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
            return True
        except Exception as e:
            logger.error(f"{self.__class__.__name__} connection failed: {str(e)}")
            return False

    def test_connection(self) -> None:
        """Test API connection with minimal request"""
        return True

class GeminiClient(BaseClient):
    """Client for Gemini API interactions"""
    def __init__(self, mode: str, role: str, api_key: str, domain: str, model: str = "gemini-2.0-flash-exp"):
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
                         model_config: Optional[ModelConfig] = None) -> str:
        """Generate response using Gemini API with assertion verification"""
        if model_config is None:
            model_config = ModelConfig()

        if not self.instructions:
            self.instructions = self._get_initial_instructions()

        # Update instructions based on conversation history
        current_instructions = self._update_instructions(history)

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
        self.client = Anthropic(api_key=api_key)
        self.max_tokens = 4096

    def generate_response(self,
                         prompt: str,
                         system_instruction: str = None,
                         history: List[Dict[str, str]] = None,
                         role: str = None,
                         model_config: Optional[ModelConfig] = None) -> str:
        """Generate human-like response using Claude API with conversation awareness"""
        if model_config is None:
            model_config = ModelConfig()

        self.role = role
        
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
        context_prompt = self.generate_human_prompt(history) if role == "human" or role == "user" else prompt
       
        messages = [{'role': msg['role'], 'content': ''.join(msg['content'])} for msg in history if msg['role'] == 'user' or msg['role'] == 'human' or msg['role'] == "assistant"]
        
        messages.append({
            "role": "user",
            "content": ''.join(context_prompt)
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
                         model_config: Optional[ModelConfig] = None) -> str:
        """Generate response using OpenAI API"""
        if model_config is None:
            model_config = ModelConfig()
        if role:
            self.role = role

        current_instructions = system_instruction
        if history and len(history) > 0 and (role == "user" or role == "human" or self.mode == "ai-ai"):
            current_instructions = self._update_instructions(history)
        elif self.role == "user" or self.role == "human":
            current_instructions = self._get_initial_instructions()
        else:
            current_instructions = system_instruction if system_instruction is not None else self.generate_human_system_instructions

        # Format messages for OpenAI API
        messages = [{
            'role': 'developer',
            'content': current_instructions
        }]

        if history:
            for msg in history:
                old_role = msg["role"]
                if old_role in ["user", "assistant", "moderator", "system"]:
                    new_role = 'developer' if old_role == "system" else "user" if old_role in ["user", "User", "human", "moderator"] else 'assistant'
                    messages.append({'role': new_role, 'content': msg['content']})

        # Add current prompt
        if self.role == "human" or self.mode == "ai-ai":
            prompt = self.generate_human_prompt()
        messages.append({'role': 'user', 'content': prompt})

        try:
            if "o1" in self.model:
                response = self.client.chat.completions.create(
                    model="o1",
                    messages=messages,
                    temperature=1.0,
                    max_tokens=4096,
                    reasoning_effort="high",
                    timeout=90,
                    stream=False  # Disable streaming
                )
                return response.choices[0].message.content
            else:
                response = self.client.chat.completions.create(
                    model="chatgpt-4o-latest",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048,
                    timeout=90,
                    stream=False  # Enable streaming
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
        #TODO: Implement actual Ollama connection test
        logger.info("Pico connection test not yet implemented")
        
    def generate_response(self,
                              prompt: str,
                              system_instruction: str = None,
                              history: List[Dict[str, str]] = None,
                              model_config: Optional[ModelConfig] = None,
                              role: str = None,
                              ) -> str:
        """
        Generate a response from your local Ollama model.

        Args:
            prompt: The user prompt or content to be sent
            system_instruction: Additional system-level instruction (optional)
            history: Conversation history if you want to incorporate it
            model_config: Model parameters (temperature, etc.) if you want to apply them

        Returns:
            str: The model's text response
        """
        if role:
            self.role = role
        if model_config is None:
            model_config = ModelConfig()
        
        # Combine system instruction + conversation history + user prompt
        shorter_history=history.copy()
        if system_instruction:
            shorter_history = [{'role': 'system', 'content': system_instruction}]
        if history:
            shorter_history.append([ {'role' : hst['role'], 'content': ''.join(hst['content']).strip()} for hst in history[-4:]])

       # shorter_hist += [ { 'role': msg['role'], 'content' : ''.join(msg['content']) } for msg in history if msg['role'] == 'user' or msg['role'] == 'human' or msg['role']=="assistant"][-6:]
        # Finally add the new prompt
        #shorter_hist = history  else [{"role": "system", "content": prompt}]

        history.append({"role": "user", "content": self.generate_human_prompt if role == 'user' or role == 'human' else prompt })

        try:
            from ollama import Client
            pico_client = Client(
                host='http://localhost:10434',
            )
            response = pico_client.chat(
                model=self.model, 
                messages=shorter_history,
                options = {
                    "num_ctx": 6144, 
                    "num_predict": 1536, 
                    "temperature": 0.75,
                    "num_batch": 512,
                    }
                )
                #print(part['message']['content'], end='', flush=True)
                #text += part['message']['content']
            return response.message.content
        except Exception as e:
            logger.error(f"Ollama generate_response error: {e}")
            raise e



class MLXClient(BaseClient):
    """Client for local MLX model interactions"""
    def __init__(self, mode:str, domain: str = "General knowledge", base_url: str = "http://localhost:9999", model: str = "mlx") -> None:
        super().__init__(mode=mode, api_key="", domain=domain, model=model)
        self.base_url = base_url or "http://localhost:9999"
        
        # super().__init__(mode=mode, api_key="local", domain=domain)
        #self.adaptive_manager=super().adaptive_manager(mode=self.mode)
        
        
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
            for msg in history:
                if msg["role"] in ["user", "human", "moderator"]:
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] in ["assistant", "system"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                #messages.append({"role": "user" if msg["role"] == "user" else "assistant", "content": msg["content"]})
                
        messages.append({"role": "user", "content": str(prompt)})
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "stream": False  # Enable streaming
                },
                stream=False
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:

            partial_text = []
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    decoded = chunk.decode("utf-8", errors="ignore")
                    partial_text.append(decoded)

            return "".join(partial_text).strip()
            #response.raise_for_status()
            #return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"MLX generate_response error: {e}")
            return f"Error: {e}"


class OpenAIClient(BaseClient):
    """Client for OpenAI API interactions"""
    def __init__(self, api_key: str=openai_api_key, mode:str="ai-ai", domain: str = "General Knowledge", role:str=None, model: str = "chatgpt-4o-latest"):
        api_key= os.environ.get("OPENAI_API_KEY")
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        try:
            super().__init__(mode=mode, api_key=api_key, domain=domain, model=model, role=role)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise ValueError(f"Invalid OpenAI API key or model: {e}")
       

    def validate_connection(self) -> bool:
        """Test OpenAI API connection"""
        self.instructions =  self._get_initial_instructions()
        return True

    def generate_response(self,
                                prompt: str,
                                system_instruction: str,
                                history: List[Dict[str, str]],
                                role: str = None,
                                model_config: Optional[ModelConfig] = None) -> str:
        """Generate response using OpenAI API"""
        if model_config is None:
            model_config = ModelConfig()
        if role:
            self.role = role
        # Update instructions based on conversation history
        #self.instructions = self._get_initial_instructions()

        current_instructions = system_instruction
        if history and len(history)>0 and ( role == "user" or role=="human" or self.mode == "ai-ai"):
            current_instructions = self._update_instructions(history)
        elif self.role == "user" or self.role == "human":
            current_instructions = self._get_initial_instructions()
        else:
            current_instructions = system_instruction if system_instruction is not None else self.generate_human_system_instructions

        #self.generate_human_prompt
        # Format messages for OpenAI API
        messages = [{
            'role': 'developer',
            'content': current_instructions
        }]

        if history:
            for msg in history: 
                old_role = msg["role"]
                if old_role in ["user", "assistant", "moderator","system"]:
                    new_role = 'developer' if old_role == "system" else "user" if old_role in ["user","User", "human", "moderator"]  else 'assistant'
                    messages.append({'role': new_role, 'content': msg['content']})

        # Add current prompt
        if self.role == "human" or self.mode == "ai-ai":
            prompt = self.generate_human_prompt()
        messages.append({'role': 'user', 'content': prompt})

        #logger.info(f"Using instructions: {current_instructions}, len(messages): {len(messages)} context history messages to be sent to model")
        #logger.debug(f"Messages: {messages}")
        
        try:
            
            if "o1" in self.model:
                response =  self.client.chat.completions.create(
                    model="o1",
                    messages=messages,
                    temperature=1.0,
                    max_tokens=4096,
                    reasoning_effort="high",
                    timeout = 90,
                    stream=False  # Disable streaming
                )
                return response.choices[0].message.content       
            else:
                response = self.client.chat.completions.create(
                    model="chatgpt-4o-latest",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048,
                    timeout=90,
                    stream=False  # Enable streaming
                )         
                return response.choices[0].message.content       
                #)
                #response = ""
                #for chunk in stream:
        except Exception as e:
            logger.error(f"OpenAI generate_response error: {e}")
            raise e

class OllamaClient(BaseClient):
    """Client for local Ollama model interactions"""
    def __init__(self, mode:str, domain: str, role:str=None, model: str = "phi4:latest"):
        super().__init__(mode=mode, api_key="", domain=domain, model=model, role=role)
        self.base_url = "http://localhost:10434"
        
    def test_connection(self) -> None:
        """Test Ollama connection"""
        #TODO: Implement actual Ollama connection test
        logger.info("Ollama connection test not yet implemented")
        
    def generate_response(self,
                            prompt: str,
                            system_instruction: str = None,
                            history: List[Dict[str, str]] = None,
                            model_config: Optional[ModelConfig] = None,
                            mode: str = None,
                            role: str = None) -> str:
        """
        Generate a response from your local PICO MLX model via ollama API.

        Args:
            prompt: The user prompt or content to be sent
            system_instruction: Additional system-level instruction (optional)
            history: Conversation history if you want to incorporate it
            model_config: Model parameters (temperature, etc.) if you want to apply them

        Returns:
            str: The model's text response
        """
        if role:
            self.role=role
        if mode:
            self.mode=mode
        # Analyze conversation context
        conversation_analysis = self._analyze_conversation(history)
        ai_response = conversation_analysis.get("ai_response")
        ai_assessment = conversation_analysis.get("ai_assessment")
        conversation_summary = conversation_analysis.get("summary")
        current_instructions = self._update_instructions(role=role, mode=self.mode)

        # Update instructions based on conversation history
        if role and role is not None and history is not None and len(history)>0:
            current_instructions = self.generate_human_instructions() if history else system_instruction if system_instruction else self.instructions
        elif ((history and len(history)>0) or (self.mode is None or self.mode == "ai-ai")):
            current_instructions = self._update_instructions(role=role, mode=self.mode)
        elif self.role == "human" or self.role == "user":
            current_instructions = self.generate_human_instructions() if self.generate_human_instructions() is not None else self.instructions
        else:
            current_instructions = self.instructions if self.instructions else "You are an expert in {self.domain}. Respond at expert level using step by step thinking where appropriate"

        # Build context-aware prompt
        context_prompt = self.generate_human_prompt(history) if role == "human" or role == "user" or mode == "ai-ai" else prompt

        # Combine system instruction + conversation history + user prompt
        
        newhist =dict( [{"role": "system", "content": current_instructions}])
        for part in history:
            if part['role'] == 'system':
                newhist.append({'role': 'system', 'content': ''.join(part['content']).strip()})
            elif part['role'] == 'user':
                newhist.append({'role': 'user', 'content': ''.join(part['content']).strip()})
        newhist.append
        if prompt:
            history.append({'role': 'user', 'content': context_prompt})

        # Finally add the new prompt
        history.append({"role": "user", "content": prompt})

        try:
            response:ChatResponse = chat(
                model=self.model, 
                messages=history,
                options = {
                    "num_ctx": 6144, 
                    "num_predict": 1280, 
                    "temperature": 0.8,
                    "num_batch": 256,
                    }
                )
                #print(part['message']['content'], end='', flush=True)
                #text += part['message']['content']
            return response.message.content
        except Exception as e:
            logger.error(f"Ollama generate_response error: {e}")
            raise e

