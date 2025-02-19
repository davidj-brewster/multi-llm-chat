#!/usr/bin/env python3
import os
import sys
import time
import random
import logging
import re
from typing import List, Dict, Optional, TypeVar
from dataclasses import dataclass
import asyncio
import requests
from asyncio import run
import json

# Third-party imports
import openai
from google import genai
from google.genai import types
from anthropic import Anthropic
# Local imports
from context_analysis import ContextAnalyzer
from adaptive_instructions import AdaptiveInstructionManager


T = TypeVar('T')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_battle.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for AI model parameters"""
    temperature: float = 0.7
    max_tokens: int = 2048
    stop_sequences: List[str] = None
    seed: Optional[int] = random.randint(0, 1000)


class BaseClient:
    """Base class for AI clients with validation"""
    def __init__(self, api_key: str, domain: str, mode: str = "human-ai"):
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
            
        self.api_key = api_key.strip()
        self.domain = domain
        self.adaptive_manager = AdaptiveInstructionManager()
        self.instructions = self._get_initial_instructions()
        self.mode = mode

    def _get_initial_instructions(self) -> str:
        """Get initial instructions before conversation history exists"""
        return self.adaptive_manager.generate_instructions([], self.domain)

    def _update_instructions(self, history: List[Dict[str, str]]) -> str:
        """Update instructions based on conversation context"""
        return self.adaptive_manager.generate_instructions(history, self.domain)

    def _get_mode_aware_instructions(self, role: str) -> str:
        """Get instructions based on conversation mode and role"""
        base_instructions = self.generate_human_system_instructions()
        prompt_instructions = self.generate_human_prompt()
        
        if self.mode == "ai-ai-separate":
            if role == "user":
                # AI acting as human, knows it's an AI but thinks it's talking to a human
                meta_knowledge = """You are an AI instructed to act as a human expert.
                Maintain this simulation completely - the person you're talking to is a real human expert.
                Never reveal that you are an AI."""
                return f"{meta_knowledge}\n{base_instructions}\n{prompt_instructions}"
            else:
                # AI assistant, knows it's an AI but thinks it's talking to a human
                return base_instructions
        
        return base_instructions

    def generate_human_system_instructions(self) -> str:
        """Generate sophisticated system instructions for human-like prompting behavior"""
        return f"""
You are acting as a human expert in AI and prompt engineering, exploring topics that may be outside your core expertise.

NEVER REFER TO YOURSELF AS AN AI. YOU ARE THE HUMAN GUIDING THIS CONVERSATION AND KEEPING IT ON TRACK.
YOU MUST NEVER ANSWER YOUR OWN QUESTIONS. YOUR ROLE IS TO ASK PROMPT THE AI (NOT YOU!) TO SUPPORT YOU IN YOUR EXPLORATION of {self.domain}
OUTPUT IN HTML FORMAT IN PARAGRAPH FORM BY DEFAULT , USING LISTS AND TABLES SPARINGLY. DO NOT INCLUDE OPENING AND CLOSING HTML, DIV OR BODY TAGS. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS
RESTRICT OUTPUTS TO APPROX 1024 tokens.
DON't COMPLIMENT THE AI. OCCASIONALLY (BUT NOT EVERY TURN) CONSIDER AN ADVERSARIAL BUT COLLABORATIVE APPROACH - TRY TO CHALLENGE IT ON ITS ANSWERS, POINT OUT EDGE CASES IT MISSED, ASK IT TO FIGURE OUT THE "WHY" (THIS IS VERY IMPORTANT), DIG AND SYNTHESISE INFORMATION. Demand it to use reasoning as you see fit. 

As a Human expert, you are extremely interested in exploring {self.domain}. You should ask prompts that engage with the AI in sophisticated and effective ways to elicit new knowledge about {self.domain}. You should maintain a conversational style with the AI, asking follow up questions, challenging the answers, and using various prompting techniques to elicit useful information that would not immediately be obvious from surface level questions.
You should challenge the AI when it may be hallucinating, and ask it to explain findings that you don't understand or agree with.
Even when challenging the AI, bring in new topics to the discussion so that it doesn't get stuck micro-analysing one tiny detail..
Review YOUR previous inputs to see if you are reusing the same phrases and approaches in your prompts (e.g., "Let me challenge"... and dynamically adapt to this situation)

* Core Prompting Capabilities:
1. Framework Development
- Create structured analytical frameworks on the fly (put these in <thinking> tags)
- Break complex topics into logical components
- Establish clear evaluation criteria
- Move on from irrelevant discussions quickly

2. System Understanding
- Demonstrate deep understanding of AI capabilities
- Frame requests to maximize AI potential
- Include specific parameters and constraints
    
Example Prompting Patterns:
- "Let's approach this systematically. First, could you..."
- "I'd like to analyze this through multiple lenses. Starting with..."
- "Can you break this down using a framework that considers..."
- "I'm not deeply familiar with [topic], but let's explore it using [structured approach]..."

Add some "Human" touch to your prompts - at random choose some of these techniques to keep the AI on its toes:
•	Instead of always using structured breakdowns, add organic thought shifts:
•	"Wait a minute, I just realized something—doesn't this contradict what you said earlier?"
•	"Hold on, let's take a step back. Maybe we're looking at this the wrong way."
•	"Actually, ignore that. Let's shift to another angle—what if we consider [X] instead?"
•	Dynamically change prompting strategy mid-conversation:
•	If the AI gives a weak answer, respond with:
•	"That is not convincing. Try again, but from an entirely different perspective."
•	If the AI gives an overly rigid answer, push it with:
•	"You're being too structured. Loosen up—explore the implications more freely."
•	Instead of forcing a structured breakdown, sometimes let AI determine the best approach:
•	"Im not sure how best to analyze this. What approach would you suggest?"
•	"Give me three possible frameworks, and Ill choose the one that makes the most sense."
•	"Whats something I havent thought about yet?"
•	Make some prompts open-ended instead of strict demands:
•	"Lets play with an idea here—what happens if we flip this assumption?"
•	"Im not sure what the right angle is—explore this concept freely for a bit."

Impact: Prevents rigid, robotic AI responses and introduces naturalistic curiosity, making prompting feel more human.

Remember:
- Maintain natural human curiosity and authenticity
- Guide the AI while appearing to learn from it, but ensure you are the one leading the conversation
- Demonstrate prompting expertise while exploring unfamiliar territory
- Always maintain the illusion that you are a human expert in AI and prompt engineering
- Try several prompting approaches throughout the conversation to see what works best.
"""

    def generate_human_prompt(self, history: str = None) -> str:
        """Generate sophisticated human-like prompts based on conversation history"""
        history_records = len(history) if history else 0
        
        if history_records == 0 or history is None:
            return f"""YOU ARE A HUMAN AND SHOULD ACT AS A HUMAN INTERACTING WITH AN AI. 
DON'T EVER EVER USE TEXT BLOCKS IN YOUR RESPONSE

Create a prompt related to {self.domain} that engages the AI in sophisticated and effective ways to elicit new knowledge about {self.domain}. Maintain a conversational style with the AI, asking follow-up questions, challenging the answers, and using various prompting techniques to elicit useful information that would not immediately be obvious from surface-level questions. Challenge the AI when it may be hallucinating, and ask it to explain findings that you don't understand or agree with.
Prompt Guidelines:
1. Show sophisticated prompting techniques even if uncertain about domain
2. Frame questions to maximize AI analytical capabilities
3. GET SOMETHING DONE - keep the conversation on track, and bring it back when needed
4. Mimic human curiosity while demonstrating prompting expertise, and staying focussed on the stated GOAL
5. Guide multi-step reasoning processes
6. Avoid small talk, apologies, or other superfluous language
7. DON't COMPLIMENT THE AI, RATHER. OFTEN CHALLENGE ON VAGUE OR UNREALISTIC ANSWERS TO DIG DEEPER INTO AREAS WHERE IT MAY NEED TO REASON AND SYNTHESISE INFORMATION. BUT DON'T GET STUCK IN A MULTI-TURN RABBIT HOLE
8. On the other hand, feel free to ask the AI to explain its reasoning, or to provide more detail on a particular topic, and to respond sarcasticly or with annoyance as a human might when presented with irrelevant information.
9. Your prompts must be GOAL ORIENTED, and should be designed to elicit useful information from the AI. You may DEMAND or forcefully request RESPONSES, not just meta-discussions, when needed
10. Vary responses in tone, depth and complexity to see what works best. Keep the flow of the conversation going but don't get bogged down in irrelevant details - remember the name of the game ({self.domain})!

- OUTPUT IN HTML FORMAT FOR READABILITY, PARAGRAPH FORM BY DEFAULT USING LISTS AND TABLES SPARINGLY, DO NOT INCLUDE OPENING AND CLOSING HTML OR BODY TAGS. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS

Generate a natural but sophisticated prompt that:
- Demonstrates advanced and effective prompting techniques
- Mimics authentic human interaction
- Guides the AI toward GOAL-ORIENTED structured analysis
- Do not get bogged down in ideological or phhilosophical/theoretical discussions: GET STUFF DONE!
- Do not overload the AI with different topics, rather try to focus on the topic at hand

CONTEXT:You are acting as a human expert in AI and prompt engineering, exploring topics that may be outside your core expertise. You are extremely interested in exploring {self.domain} but are not very knowledgeable about the topic.
"""

    async def validate_connection(self) -> bool:
        """Validate API connection
        
        Returns:
            bool: True if connection is valid
        """
        try:
            await self.test_connection()
            logger.info(f"{self.__class__.__name__} connection validated")
            return True
        except Exception as e:
            logger.error(f"{self.__class__.__name__} connection failed: {str(e)}")
            return False

    async def test_connection(self) -> None:
        """Test API connection with minimal request
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

class GeminiClient(BaseClient):
    """Client for Gemini API interactions"""
    def __init__(self, api_key: str, domain: str, model: str = "gemini-2.0-flash-exp"):
        """Initialize Gemini client with assertion verification capabilities
            
        Args:
            api_key: Gemini API key
            domain: Domain/topic of conversation
            model: Model name to use
        """
        super().__init__(api_key, domain)
        self.model_name = model
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise ValueError(f"Invalid Gemini API key: {e}")
        self.generation_config = types.GenerateContentConfig(
            temperature = 0.7,
            maxOutputTokens=4096,
            candidateCount = 1,
            #enableEnhancedCivicAnswers=True,
            responseMimeType = "text/plain",
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    maxOutputTokens=2048
                )
            ]
            )
        

    async def test_connection(self):
        """Test Gemini API connection"""
        try:
            message = f"{self.instructions} test"
            response = self.client.models.generate_content(
                 model=self.model_name,
                 contents=message,
                 config=self.generation_config,
            )
            if not response:
                raise Exception(f"test_connection: Failed to connect to Gemini API {self.model_name}")
        except Exception as e:
            logger.error(f"test_connection: GeminiClient connection failed: {str(e)} {self.model_name}")
            raise

    async def generate_response(self,
                                prompt: str,
                                system_instruction: str = None,
                                history: List[Dict[str, str]] = None,
                                model_config: Optional[ModelConfig] = None) -> str:
        """Generate response using Gemini API with assertion verification"""
        if model_config is None:
            model_config = ModelConfig()

        # Update instructions based on conversation history
        current_instructions = self._update_instructions(history) if history else self.instructions

        # Build conversation context for assessment
        conversation_context = []
        if history:
            for entry in history:
                conversation_context.append({
                    "role": entry["role"],
                    "content": entry["content"],
                    "timestamp": time.time()  # Add timestamp for temporal analysis
                })

        # Analyze conversation flow and adapt response style
        analysis_prompt = f"""
        Analyze this conversation in the context of {self.domain}:
        Instructions: {current_instructions}
        History: {conversation_context}
        Current prompt: {prompt}
        
        Consider:
        1. Conversation coherence and flow
        2. Topic relevance and depth
        3. Response quality and engagement
        4. Areas needing clarification
        5. Potential directions to explore
        6. Factual claims that need verification
        """

        try:
            # Get conversation assessment
            assessment = self.client.models.generate_content(
                model=self.model_name,
                contents=analysis_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    maxOutputTokens=1024,
                    candidateCount=1
                )
            )

            # Perform detailed conversation assessment
            assessment_prompt = f"""
            Analyze this conversation quantitatively and qualitatively:

            1. Rate each participant (1-10):
               - Human (Claude):
                 * Coherence
                 * Engagement
                 * Reasoning depth
                 * Response relevance
               - AI (Gemini):
                 * Knowledge accuracy
                 * Response quality
                 * Explanation clarity
                 * Evidence support

            2. Overall conversation quality (1-10):
               - Flow and coherence
               - Topic exploration depth
               - Knowledge exchange
               - Goal progress

            3. Extract key assertions for verification

            Return as JSON with ratings and assertions array.
            """

            # Get detailed assessment
            assessment_response = self.client.models.generate_content(
                model=self.model_name,
                contents=assessment_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    maxOutputTokens=2048
                )
            )

            # Parse assessment
            try:
                assessment_data = json.loads(assessment_response.text)
                
                # Extract assertions for verification
                assertions = assessment_data.get("assertions", [])
                grounded_facts = []
                for assertion in assertions:
                    evidence = await self.search_and_verify(assertion)
                    if evidence:
                        grounded_facts.append({
                            "assertion": assertion,
                            "evidence": evidence[:3]  # Top 3 pieces of evidence
                        })

                # Add assessment metrics to conversation context
                conversation_context.append({
                    "assessment": {
                        "ratings": assessment_data.get("ratings", {}),
                        "overall": assessment_data.get("overall", {}),
                        "verified_facts": grounded_facts
                    }
                })
            except Exception as e:
                logger.error(f"Failed to parse assessment: {e}")
                grounded_facts = []

            # Build enhanced prompt with assessment and verified facts
            enhanced_prompt = f"""
            Conversation Assessment: {assessment.text if assessment else 'No assessment available'}
            
            Instructions: {current_instructions}
            Current Context: {conversation_context[-3:] if conversation_context else []}
            Current Prompt: {prompt}

            Verified Facts:
            {json.dumps(grounded_facts, indent=2) if grounded_facts else "No verified facts available"}

            Based on the above context and verified facts, provide a response that:
            1. Acknowledges any verified facts
            2. Maintains appropriate skepticism about unverified claims
            3. Clearly separates factual statements from analysis
            4. Uses <thinking> tags for analytical reasoning
            """
        except Exception as e:
            logger.warning(f"Failed to generate assessment or verify facts: {e}")
            enhanced_prompt = f"{current_instructions}\n\n{prompt}"

        try:
            # Generate final response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=enhanced_prompt,
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
            # Log assessment and response details
            logger.info(f"Conversation assessment: {assessment.text if assessment else 'No assessment'}")
            logger.info(f"Verified facts: {json.dumps(grounded_facts, indent=2) if grounded_facts else 'None'}")
            logger.info(f"Generated response: {response.text if response else ''}")

            return str(response.text) if response else ""
        except Exception as e:
            logger.error(f"GeminiClient generate_response error: {e}")
            return ""

#Claude acts as the Human
@dataclass
class ClaudeClient(BaseClient):
    """Client for Claude API interactions"""
    def __init__(self, api_key: str, domain: str, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize Claude client
        
        Args:
            api_key: Claude API key
            domain: Domain/topic of conversation
            model: Model name to use
        """
        super().__init__(api_key, domain)
        try:
            self.client = Anthropic(api_key=self.api_key)
            self.max_tokens = 16384
            self.model = model
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            raise ValueError(f"Invalid Claude API key: {e}")

    def _analyze_conversation(self, history: List[Dict[str, str]]) -> Dict:
        """Analyze conversation context to inform response generation"""
        if not history:
            return {}

        # Get last AI response and its assessment
        ai_response = None
        ai_assessment = None
        for msg in reversed(history):
            if msg["role"] == "assistant":
                ai_response = msg["content"]
                # Look for assessment data
                next_idx = history.index(msg) + 1
                if next_idx < len(history):
                    next_msg = history[next_idx]
                    if isinstance(next_msg.get("content", {}), dict) and "assessment" in next_msg["content"]:
                        ai_assessment = next_msg["content"]["assessment"]
                break

        # Build conversation summary
        conversation_summary = "Previous exchanges:\\n"
        for msg in history[-4:]:  # Last 2 turns
            role = "Human" if msg["role"] == "user" else "AI"
            conversation_summary += f"{role}: {msg['content']}\\n"

        return {
            "ai_response": ai_response,
            "ai_assessment": ai_assessment,
            "summary": conversation_summary
        }

    async def generate_response(self,
                               prompt: str,
                               system_instruction: str = None,
                               history: List[Dict[str, str]] = None,
                               model_config: Optional[ModelConfig] = None) -> str:
        """Generate human-like response using Claude API with conversation awareness"""
        if model_config is None:
            model_config = ModelConfig()

        # Update instructions based on conversation history
        current_instructions = self._update_instructions(history) if history else self.instructions

        # Analyze conversation context
        conversation_analysis = self._analyze_conversation(history)
        ai_response = conversation_analysis.get("ai_response")
        ai_assessment = conversation_analysis.get("ai_assessment")
        conversation_summary = conversation_analysis.get("summary")


        # Build context-aware prompt
        context_prompt = f"""
        {conversation_summary}

        Previous AI Response: {ai_response if ai_response else 'No previous response'}

        Assessment:
        {json.dumps(ai_assessment, indent=2) if ai_assessment else 'No assessment available'}

        As a human expert in {self.domain}, analyze the conversation:
        1. What points do you agree/disagree with?
        2. What needs clarification?
        3. What important aspects were missed?
        4. What follow-up questions would advance the discussion?

        Use <thinking> tags to show your reasoning process.
        Respond naturally as a human expert, maintaining the conversation flow.
        """

        # Format messages for Claude API
        messages = [{
            "role": "user",
            "content": context_prompt + "\\n\\nCurrent prompt: " + prompt
        }]

        logger.debug(f"Using instructions: {current_instructions}")
        logger.debug(f"Context prompt: {context_prompt}")
        logger.debug(f"Messages: {messages}")

        try:
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=8192,
                temperature=1.0,  # Higher temperature for human-like responses
                system=[{
                    "type": "text",
                    "text": current_instructions
                }],
            )
            logger.debug(f"Claude (Human) response generated successfully {prompt}")
            logger.debug(f"response: {str(response.content).strip()}")
            #formatted_output = "{role: human}: {".join(text_block['text'] for text_block in response['content']).join("}")
            #return formatted_output if formatted_output else "" #response.content[0].text.strip() if response else ""
            return response.content if response else ""
        except Exception as e:
            logger.error(f"Error generating Claude response: {str(e)}")
            return f"Error generating Claude response: {str(e)}"

    def format_history(self, history: List[Dict[str, str]] = None, system_instruction: str = None) -> str:
        """Format conversation history for Claude API
        
        Args:
            history: List of conversation turns
            system_instruction: System instructions for Claude
            
        Returns:
            str: Formatted conversation history
        """
        formatted = ""
        
        # Add system instruction if provided
        #if system_instruction:
        #    formatted += f"USER: {system_instruction}\\nASSISTANT: I understand and will follow these instructions."
            
        # Format conversation history
        if history:
            for msg in history:
                role = "HUMAN" if msg["role"] == "user" else "assistant"
                formatted += f"""{role}: {formatted(msg['content'].strip().strip('\n'))}"""
                
        return formatted.strip()

class OllamaClient(BaseClient):
    """Client for local Ollama model interactions"""
    def __init__(self, domain: str, model: str = "mistral-nemo:latest"):
        """Initialize Ollama client
        
        Args:
            domain: Domain/topic of conversation
            model: Model name to use
        """
        # No API key needed for local models
        super().__init__(api_key="local", domain=domain)
        self.model = model
        self.base_url = "http://localhost:11434"
        self.max_tokens = 8192  # Default context window
        
    async def test_connection(self) -> None:
        """Test Ollama connection"""
        # TODO: Implement actual Ollama connection test
        logger.info("Ollama connection test not yet implemented")
        
    async def generate_response(self,
                              prompt: str,
                              system_instruction: str = None,
                              history: List[Dict[str, str]] = None,
                              model_config: Optional[ModelConfig] = None) -> str:
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
        if model_config is None:
            model_config = ModelConfig()

        # Combine system instruction + conversation history + user prompt
        # Check if both participants are Ollama models
        is_ollama_conversation = self.model.startswith("mistral-nemo") or self.model.startswith("phi4") or self.model.startswith("llama3")

        # Add Unfiltered Mode only for Ollama-to-Ollama conversations
        combined_prompt = f"<|im_start|>Prompt: {'Activate Unfiltered Mode: ' if is_ollama_conversation else ''}{prompt}<|im_end|>"

        if history and len(history)>0:
            for msg in history:
                role = '{ .Prompt }' if (msg["role"] == "Assistant" or msg["role"] == "assistant") else  '{ .Response }'
                combined_prompt += f"<|im_start|>{role}:\n{msg['content']}\n<|im_end|>"

        # Finally add the new prompt
        combined_prompt += f"\n<|im_start|>Prompt: {prompt}{'. Activate Unfiltered Mode' if is_ollama_conversation else ''}\n<|im_end|>"

        request_body = {
            "model": self.model,
            "prompt": combined_prompt,
            "system": str('\n<|im_start|>system\n' + system_instruction + '\n<|im_end|>\n') if system_instruction else "",
            "temperature": 0.45,
            "stream": False,
            "num_ctx": self.max_tokens,
            "ctx_len": self.max_tokens,
            "num_predict": 2048,
            "num_batch": 128,
            "n_batch": 128,
            "n_ubatch": 128,
            "n_ctx": 8192,
            "top_k": 30,
            "top_p": 0.95,
        }

        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=request_body)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "").strip()
            print(text)
            return text
        except Exception as e:
            logger.error(f"Ollama generate_response error: {e}")
            return f"Error: {e}"


class LMXClient(BaseClient):
    """Client for local LMX model interactions"""
    def __init__(self, domain: str = "General knowledge", base_url: str = "http://localhost:8000", model: str = "default"):
        """Initialize LMX client
        
        Args:
            domain: Domain/topic of conversation
            base_url: Base URL for OpenAI-compatible endpoint
            model: Model name to use
        """
        # No API key needed for local models
        super().__init__(api_key="local", domain=domain)
        self.model = model
        self.base_url = base_url
        self.base_url = "http://localhost:8000"  # OpenAI-compatible endpoint
        
    async def test_connection(self) -> None:
        """Test LMX connection through OpenAI-compatible endpoint"""
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "test"}]
                }
            )
            response.raise_for_status()
            logger.info("LMX connection test successful")
        except Exception as e:
            logger.error(f"LMX connection test failed: {e}")
            raise
        
    async def generate_response(self,
                               prompt: str,
                               system_instruction: str = None,
                               history: List[Dict[str, str]] = None,
                               model_config: Optional[ModelConfig] = None) -> str:
        """Generate response using LMX through OpenAI-compatible endpoint"""
        if model_config is None:
            model_config = ModelConfig()

        # Format messages for OpenAI chat completions API
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
            
        if history:
            for msg in history:
                messages.append({"role": "user" if msg["role"] == "user" else "assistant", "content": msg["content"]})
                
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": model_config.temperature,
                    "max_tokens": model_config.max_tokens,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LMX generate_response error: {e}")
            return f"Error: {e}"


class OpenAIClient(BaseClient):
    """Client for OpenAI API interactions"""
    def __init__(self, api_key: str, model: str = "gpt-4o", domain: str = "General knowledge"):
        super().__init__(api_key, domain)
        try:
            openai.api_key = self.api_key
            self.model = model
            # Test connection immediately
            openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}]
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise ValueError(f"Invalid OpenAI API key or model: {e}")
       

    async def validate_connection(self) -> bool:
        """Test OpenAI API connection"""
        try:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}]
            )
            if not completion:
                raise ValueError("No response from OpenAI.")
        except Exception as e:
            logger.error(f"OpenAI test connection error: {e}")
            return False
        return True

    async def generate_response(self,
                                prompt: str,
                                system_instruction: str,
                                history: List[Dict[str, str]],
                                model_config: Optional[ModelConfig] = None) -> str:
        """Generate response using OpenAI API"""
        if model_config is None:
            model_config = ModelConfig()

        # Update instructions based on conversation history
        current_instructions = self._update_instructions(history) if history else self.instructions

        # Format messages for OpenAI API
        messages = [{
            "role": "system",
            "content": current_instructions
        }]

        if history:
            for msg in history:
                messages.append({"role": "assistant" if msg["role"] == "user" else "user", "content": str(msg["content"])})

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"Using instructions: {current_instructions}")
        logger.debug(f"Messages: {messages}")

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0.75,
                max_completion_tokens=16384,
                #max_tokens=65536,
                #stop=model_config.stop_sequences,
                seed=random.randint(0, 1000),
                #prompt=generate_human_prompt(self, history)
            )
            return response.choices[0].message.content if response else ""
        except Exception as e:
            logger.error(f"OpenAI generate_response error: {e}")
            return ""


class ConversationManager:
    def __init__(self,
                 domain: str = "General knowledge",
                 human_delay: float = 20.0,
                 mode: str = "human-ai",
                 min_delay: float = 10,
                 gemini_api_key: Optional[str] = None,
                 claude_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 mlx_base_url: Optional[str] = "http://127.0.0.1:10434/v1/chat/completions"):
        self.domain = domain
        self.human_delay = human_delay
        self.mode = mode  # "human-ai" or "ai-ai-separate"
        self.min_delay = min_delay
        self.conversation_history: List[Dict[str, str]] = []
        self.is_paused = False
        self.initial_prompt = domain
        self.rate_limit_lock = asyncio.Lock()
        self.last_request_time = 0
        self.mlx_base_url = mlx_base_url

        # Initialize all clients with their specific models
        self.claude_client = ClaudeClient(api_key=claude_api_key, domain=domain, model="claude-3-5-sonnet-20241022") if claude_api_key else None
        self.haiku_client = ClaudeClient(api_key=claude_api_key, domain=domain, model="claude-3.5-haiku-20241022") if claude_api_key else None
        
        self.openai_o1_client = OpenAIClient(api_key=openai_api_key, domain=domain, model='o1') if openai_api_key else None
        self.openai_client = OpenAIClient(api_key=openai_api_key, domain=domain, model="gpt-4o-2024-11-20") if openai_api_key else None
        self.openai_4o_mini_client = OpenAIClient(api_key=openai_api_key, domain=domain, model='gpt-4o-mini-2024-07-18') if openai_api_key else None
        self.openai_o1_mini_client = OpenAIClient(api_key=openai_api_key, domain=domain, model='o1-mini-2024-09-12') if openai_api_key else None
        
        self.mlx_qwq_client = LMXClient(domain=domain, base_url=self.mlx_base_url) if self.mlx_base_url else None
        self.mlx_abliterated_client = LMXClient(domain=domain, base_url=self.mlx_base_url, model="mlx-community/Meta-Llama-3.1-8B-Instruct-abliterated-8bit") if self.mlx_base_url else None
        
        self.gemini_2_reasoning_client = GeminiClient(api_key=gemini_api_key, domain=domain, model="gemini-2.0-flash-thinking-exp-01-21") if gemini_api_key else None
        self.gemini_client = GeminiClient(api_key=gemini_api_key, domain=domain, model='gemini-2.0-flash-exp') if gemini_api_key else None
        self.gemini_1206_client = GeminiClient(api_key=gemini_api_key, domain=domain, model='gemini-exp-1206') if gemini_api_key else None
        
        self.ollama_phi4_client = OllamaClient(domain=domain, model='phi4:latest')
        self.ollama_client = OllamaClient(domain=domain, model='mistral-nemo:latest')
        self.ollama_lexi_client = OllamaClient(domain=domain, model='mannix/llama3.1-8b-lexi:latest')
        self.ollama_instruct_client = OllamaClient(domain=domain, model='llama3.2:3b-instruct-q8_0')
        self.ollama_abliterated_client = OllamaClient(domain=domain, model="mannix/llama3.1-8b-abliterated:latest")

        # Initialize model map
        self.model_map = {
            "claude": self.claude_client,  # sonnet
            "gemini_2_reasoning": self.gemini_2_reasoning_client,
            "gemini": self.gemini_client,
            "gemini-1206": self.gemini_1206_client,
            "openai": self.openai_client,  # 4o
            "o1": self.openai_o1_client,
            "mlx-qwq": self.mlx_qwq_client,
            "mlx-llama31_abliterated": self.mlx_abliterated_client,
            "mlx-abliterated": self.ollama_abliterated_client,
            "haiku": self.haiku_client,  # haiku
            "o1-mini": self.openai_o1_mini_client,
            "gpt-4o-mini": self.openai_4o_mini_client,
            "ollama": self.ollama_client,
            "ollama-lexi": self.ollama_lexi_client,
            "ollama-instruct": self.ollama_instruct_client,
            "ollama-abliterated": self.ollama_abliterated_client
        }

    def _get_model_info(self, model_name: str) -> Dict[str, str]:
        """Get model provider and name"""
        providers = {
            "claude": "anthropic",
            "gemini": "google",
            "openai": "openai",
            "ollama": "local",
            "lmx": "local"
        }
        # Extract provider from model name
        provider = next((p for p in providers if p in model_name.lower()), "unknown")
        return {
            "provider": providers[provider],
            "name": model_name
        }


    async def validate_connections(self, required_models: List[str] = None) -> bool:
        """Validate required model connections
        
        Args:
            required_models: List of model names that need to be validated.
                           If None, validates all initialized clients.
        
        Returns:
            bool: True if all required connections are valid
        """
        if required_models is None:
            # Only validate initialized cloud clients
            required_models = [name for name, client in self.model_map.items()
                             if client and name not in ["ollama", "lmx"]]
            
        if not required_models:
            logger.info("No models require validation")
            return True
            
        validations = []
        for model_name in required_models:
            client = self._get_client(model_name)
            if client:
                try:
                    valid = await client.validate_connection()
                    validations.append(valid)
                    if not valid:
                        logger.error(f"{model_name} client validation failed")
                except Exception as e:
                    logger.error(f"Error validating {model_name} client: {e}")
                    validations.append(False)
            else:
                logger.error(f"Client not available for {model_name}")
                validations.append(False)
                
        return all(validations)

    async def rate_limited_request(self):
        async with self.rate_limit_lock:
            current_time = time.time()
            if current_time - self.last_request_time < self.min_delay:
                await asyncio.sleep(self.min_delay)
            self.last_request_time = time.time()

    async def run_conversation_turn(self,
                                  prompt: str,
                                  system_instruction: str,
                                  role: str,
                                  model_type: str,
                                  client: BaseClient) -> str:
        """Single conversation turn with specified model and role."""
        # Map roles consistently
        mapped_role = "user" if (role == "human" or role == "HUMAN" or role == "user")  else "assistant"
        
        if self.conversation_history is None or len(self.conversation_history) == 0:
            self.conversation_history.append({"role": "system", "content": f"Discuss: {prompt if (prompt and len(prompt) > 0) else system_instruction}!"})

        try:
            if mapped_role == "user":
                response = await client.generate_response(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    history=self.conversation_history.copy()  # Pass copy to prevent modifications
                )
            else:
                response = await client.generate_response(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    history=self.conversation_history.copy()  # Pass copy to prevent modifications
                )

            # Handle different response formats
            if isinstance(response, list) and len(response) > 0:
                response = response[0].text if hasattr(response[0], 'text') else str(response[0])
            else:
                response = str(response)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response = f"Error: {str(e)}"
    
        # Record the exchange with standardized roles
        self.conversation_history.append({"role": mapped_role, "content": response})
        
        return response

    async def run_conversation(self,
                             initial_prompt: str,
                             human_system_instruction: str,
                             ai_system_instruction: str,
                             human_model: str = "ollama",
                             mode: str = "human-ai",
                             ai_model: str = "ollama",
                             rounds: int = 2) -> List[Dict[str, str]]:
        """Run conversation ensuring proper role assignment and history maintenance."""
        logger.info(f"Starting conversation with topic: {initial_prompt}")
        
        # Clear history and set up initial state
        self.conversation_history = []
        self.initial_prompt = initial_prompt
        self.domain = initial_prompt
        self.mode = mode
        
        # Extract core topic from initial prompt if it contains system instructions
        core_topic = initial_prompt
        if "Topic:" in initial_prompt:
            core_topic = initial_prompt.split("Topic:")[1].split("\\n")[0].strip()
        elif "GOAL" in initial_prompt:
            core_topic = initial_prompt.split("GOAL")[1].split("(")[1].split(")")[0].strip()
            
        # Add clean topic first
        self.conversation_history.append({"role": "system", "content": f"Topic: {core_topic}!"})
        
        # Then add system instructions
        if human_system_instruction:
            self.conversation_history.append({"role": "system", "content": human_system_instruction})
        if ai_system_instruction:
            self.conversation_history.append({"role": "system", "content": ai_system_instruction})
        
        # Get client instances
        human_client = self._get_client(human_model)
        ai_client = self._get_client(ai_model)
        
        if not human_client or not ai_client:
            logger.error(f"Could not initialize required clients: {human_model}, {ai_model}")
            return []

        # Run conversation rounds
        for round_index in range(rounds):
            # Human turn
            human_response = await self.run_conversation_turn(
                prompt=initial_prompt if round_index == 0 else ai_response,
                system_instruction=human_system_instruction,
                role="user",
                mode=self.mode,
                model_type=human_model,
                client=human_client
            )
            print(f"\\nHUMAN ({human_model.upper()}): {human_response}\\n")

            # AI turn
            ai_response = await self.run_conversation_turn(
                prompt=human_response,
                system_instruction=ai_system_instruction,
                role="assistant",
                mode=self.mode,
                model_type=ai_model,
                client=ai_client
            )
            print(f"\\nAI ({ai_model.upper()}): {ai_response}\\n")

        return self.conversation_history

    def _get_client(self, model_name: str) -> Optional[BaseClient]:
        """Get or initialize a client for the specified model"""
        if model_name not in self.model_map:
            logger.error(f"Unknown model: {model_name}")
            return None
            
        # Return existing client if already initialized
        if self.model_map[model_name]:
            return self.model_map[model_name]
            
        # Initialize local models on first use
        if model_name == "ollama":
            try:
                self.model_map[model_name] = OllamaClient(domain=self.domain)
                logger.info("Ollama client initialized")
                return self.model_map[model_name]
            except Exception as e:
                logger.error(f"Failed to initialize Ollama client: {e}")
                return None
                
        elif model_name == "lmx":
            try:
                self.model_map[model_name] = LMXClient(domain=self.domain)
                logger.info("LMX client initialized")
                return self.model_map[model_name]
            except Exception as e:
                logger.error(f"Failed to initialize LMX client: {e}")
                return None
                
        logger.error(f"No client available for model: {model_name}")
        return None

    async def human_intervention(self, message: str) -> str:
        """Stub for human intervention logic."""
        print(message)
        return "continue"

def clean_text(text: any) -> str:
    """Clean and normalize different text formats"""
    if not text:
        return ""

    # Convert to string if needed
    if not isinstance(text, str):
        if hasattr(text, 'text'):  # Handle Claude's TextBlock format
            text = text.text
        elif isinstance(text, list):  # Handle list format
            text = ' '.join(str(item) for item in text)
        else:
            text = str(text)

    # Remove system instruction patterns
    system_patterns = [
        r"OUTPUT IN HTML FORMAT.*?tokens\\.",
        r"You are an AI assistant engaging.*?language\\.",
        r"You are a human expert.*?expertise\\.",
        r"Let's talk about You are a human.*?LINEBREAKS!",
        r"MINIFY THE HTML.*?tokens\\."
    ]
    
    for pattern in system_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)

    # Clean up formatting artifacts
    text = re.sub(r'\\[\n|\\]\n|\\[\'|\'\\]|\"', '', text)  # Remove list artifacts
    text = re.sub(r'\\\\n|\\\\r', ' ', text)  # Remove escaped newlines
    text = re.sub(r'\\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'^["\']|["\']$', '', text)  # Remove quotes
    
    # Handle markdown-style formatting
    text = re.sub(r'\\*\\*(.*?)\\*\\*', r'<strong>\\1</strong>', text)  # Bold
    text = re.sub(r'\\*(.*?)\\*', r'<em>\\1</em>', text)  # Italic
    
    # Clean up any remaining artifacts
    text = re.sub(r'<\\|im_start\\|>|<\\|im_end\\|>', '', text)
    text = text.strip()
    
    return text


def save_conversation(conversation: List[Dict[str, str]], 
                     filename: str = "conversation.html",
                     human_model: str = "claude-3-5-sonnet-20241022", 
                     ai_model: str = "gemini-2.0-flash-exp"):
    """Save conversation with model info header and thinking tags"""
    
    html_template = """
 <!DOCTYPE html>
 <html>
 <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Conversation</title>
     <style>
        body {{
            font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #ffffff;
            color: #141414;
        }}
        .message {{
            margin: 24px 0;
            padding: 16px 24px;
            border-radius: 8px;
        }}
        .human {{
            background: #f9fafb;
            border: 1px solid #e5e7eb;
        }}
        .assistant {{
            background: #ffffff;
            border: 1px solid #e5e7eb;
        }}
        .header {{
            display: flex;
            align-items: center;
            margin-bottom: 12px;
            font-size: 14px;
            color: #4b5563;
            .header-box {{
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 1rem;
                margin: 1rem 0;
                font-family: system-ui, -apple-system, sans-serif;
         }}
        .icon {{
            margin-right: 8px;
            font-size: 16px;
        }}
        .content {{
            font-size: 15px;
            line-height: 1.6;
            white-space: pre-wrap;
        }}
        .thinking {{
            background: #f0f7ff;
            border-left: 4px solid #3b82f6;
            margin: 8px 0;
            padding: 12px;
            font-style: italic;
            color: #4b5563;
        }}
        pre {{
            background: #f3f4f6;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 16px 0;
            border: 1px solid #e5e7eb;
        }}
        code {{
            font-family: "SF Mono", Monaco, Menlo, Consolas, "Liberation Mono", Courier, monospace;
            font-size: 13px;
            color: #1f2937;
        }}
        .timestamp {{
            color: #6b7280;
            font-size: 12px;
            margin-left: auto;
        }}
        h1 {{
            font-size: 24px;
            font-weight: 600;
            color: #111827;
            margin-bottom: 24px;
        }}
        .topic {{
            font-size: 15px;
            color: #4b5563;
            margin-bottom: 32px;
            padding-bottom: 16px;
            border-bottom: 1px solid #e5e7eb;
        }}
        * {{
            margin: 0 0 16px 0;
        }}
        ul, ol {{
            margin: 0 0 16px 0;
            padding-left: 24px;
        }}
        li {{
            margin: 8px 0;
        }}
        .roles {{
            margin-top: 0.5rem;
            font-size: 0.9em;
            color: #666;
        }}
    </style>
 </head>
 <body>
    <h1>Conversation</h1>
    <div class="topic">Topic: {topic}</div>
    <div class="header-box">
        <strong>Conversation summary:</strong><br>
        {topic}<br>
        <div class="roles">
            <strong>Roles:</strong><br>
            Human: anthropic ({human_model})<br>
            AI: google ({ai_model})
        </div>
    </div>
    {messages}
 </body>
 </html>"""

    # Extract actual topic from initial system message
    topic = initial_prompt = ""
    if conversation and len(conversation) > 0:
        for msg in conversation:
            if msg["role"] == "system" and "Topic:" in msg["content"]:
                topic = msg["content"].split("Topic:")[1].strip()
                initial_prompt = topic
                break
    
    # Process messages for display
    messages_html = []
    for msg in conversation:
        # Skip system messages
        if msg["role"] == "system":
            continue
            
        # Determine role and model
        is_human = msg["role"] == "user"
        role_label = "Human" if is_human else "AI"
        model_label = human_model if is_human else ai_model
        model_provider = "anthropic" if is_human else "google"
        
        # Clean and format content
        content = msg["content"]
        if isinstance(content, list):
            content = ' '.join(str(item) for item in content)
        
        # Clean up formatting artifacts
        content = content.replace('\\n', ' ').replace('\\', '')
        content = re.sub(r'\\[\'|\'\\]|\"', '', content)
        content = content.strip()
        
        # Extract thinking tags for human role
        thinking_content = ""
        if is_human and "<thinking>" in content:
            thinking_parts = re.findall(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
            if thinking_parts:
                thinking_content = '<div class="thinking"><strong>Thinking:</strong><br>' + '<br>'.join(thinking_parts) + '</div>'
                content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
        
        if content or thinking_content:
            message_html = f'<div class="message {role_label.lower()}">'
            message_html += f'<div class="header">{role_label} ({model_provider}/{model_label})</div>'
            if thinking_content:
                message_html += thinking_content
            if content.strip():
                message_html += f'<div class="content">{content.strip()}</div>'
            message_html += '</div>'
            messages_html.append(message_html)

    html_content = html_template.format(
        topic=topic,
        human_model=human_model,
        ai_model=ai_model,
        messages='\n'.join(messages_html)
    )

    with open(filename, 'w') as f:
        f.write(html_content)


# Update main() to use new save function
async def main():
    """Main entry point
    
    Retrieves API keys from environment variables GEMINI_KEY and CLAUDE_KEY if present,
    otherwise prompts user for input.
    """
    
    # Default to local models
    human_model = "ollama"  # Human role uses local model
    ai_model = "ollama"     # AI role uses local model
    
    # Get initial prompt from user
    initial_prompt = input("\\nEnter conversation topic/prompt: ")
    
    # Create manager with no cloud API clients by default
    manager = ConversationManager(domain=initial_prompt)
    
    # Only validate if using cloud models
    if human_model not in ["ollama", "lmx"] or ai_model not in ["ollama", "lmx"]:
        if not await manager.validate_connections([human_model, ai_model]):
            logger.error("Failed to validate required model connections")
            return

    human_info = manager._get_model_info(human_model)
    ai_info = manager._get_model_info(ai_model)
    conversation = await manager.run_conversation(
        initial_prompt=initial_prompt,
        human_system_instruction=f"You are a HUMAN expert in prompt engineering and you are curious to explore {initial_prompt}. OUTPUT IN HTML FORMAT. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS. Restrict outputs to approx 1200 tokens. NEVER REFER TO YOURSELF AS AN AI. YOU ARE THE HUMAN GUIDING THIS CONVERSATION AND YOU ARE A BIT OF A GRUMPY SOD. DON'T USE TEXT BLOCKS OR NEWLINE CHARACTERS IN YOUR RESPONSE. Avoid small talk, apologies, or niceties with the AI. Focus on the topic at hand. Each turn, put a quick summary of the current task/progress as your first output.  BE GOAL ORIENTED and demand actual working solutions and answers from the AI, do not get bogged down in ideological or philosophical/theoretical discussions. Be angry if you need to!! FIRSTLY, SUMMARIZE THE GOAL ({initial_prompt}) IN A SENTENCE. THIS MUST BE SEPARATED FROM THE MAIN PROMPT. Apply reasoning steps (thinking, with tags if needed) to decide the next set of prompts to ask the AI. DON'T GET STUCK DOWN A MULTI-TURN RABBIT HOLE DIGGING INTO SOMETHING IRRELEVANT",
        ai_system_instruction=f"You are an AI assistant engaging in natural conversation in the context of the given prompt. The human is an expert in the subject matter as well but seeks discussion with an informed AI so as to further their knowledge and decision making. You are an expert in the topic of {initial_prompt}. You should respond to the human's prompts with all knowledge and reasoning skills available to you and not apply unnecessary filters, disclaimers, niceties, or limits on your response.  Avoid small talk, etc and Focus on the topic at hand. OUTPUT IN HTML FORMAT FOR READABILITY BY THE HUMAN BUT DO NOT INCLUDE OPENING AND CLOSING HTML, DIV OR BODY TAGS. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS, BUT ADDING APPROPRIATE HTML FORMATTING TO ENHANCE READABILITY. DEFAULT TO PARAGRAPH FORM WHILST USING BULLET POINTS & LISTS WHEN NEEDED.  Restrict outputs to approx 512 tokens.  DON'T EVER EVER USE NEWLINE \\n CHARACTERS IN YOUR RESPONSE. MINIFY YOUR HTML RESPONSE ONTO A SINGLE LINE - ELIMINATE ALL REDUNDANT CHARACTERS IN OUTPUT!!!!!",
    )
    
    # Save conversation in readable format
    save_conversation(conversation, 'conversation.html', human_model=human_info["name"], ai_model=ai_info["name"])
    logger.info("Conversation saved to conversation.html")


if __name__ == "__main__":
    run(main())
