import json
import os
import datetime
import sys
import time
import random
import logging
import re
from ollama import AsyncClient
from typing import List, Dict, Optional, TypeVar
from dataclasses import dataclass
import io
import requests
import asyncio
# Third-party imports
from openai import OpenAI
from google import genai
from google.genai import types
from anthropic import Anthropic
# Local imports
from context_analysis import ContextAnalyzer
from adaptive_instructions import AdaptiveInstructionManager


T = TypeVar('T')
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


logging.basicConfig(
    level=logging.DEBUG,
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
    def __init__(self, mode:str, api_key: str, domain: str="",model="",role:str=""):#, mode: str = "human-aiai"):
        self.api_key = api_key.strip() if api_key else ""
        self.domain = domain
        self.mode = mode
        self.role = role
        self.model = model
        self.adaptive_manager = AdaptiveInstructionManager(mode=self.mode)
        self.instructions = None

    def __str__(self):
        return f"{self.__class__.__name__}(mode={self.mode}, domain={self.domain}, model={self.model})"

    def _get_initial_instructions(self) -> str:
        """Get initial instructions before conversation history exists"""
        if self.adaptive_manager is None:
            self.adaptive_manager = AdaptiveInstructionManager(mode=self.mode)
        return self._get_mode_aware_instructions(self.domain)

    def _update_instructions(self, history: List[Dict[str, str]]) -> str:
        """Update instructions based on conversation context"""
        if self.adaptive_manager is None:
            self.adaptive_manager = AdaptiveInstructionManager(mode=self.mode)
        return self.adaptive_manager.generate_instructions(history, self.domain)

    def _get_mode_aware_instructions(self, role: str=None, mode:str=None) -> str:
        """Get instructions based on conversation mode and role"""
        #base_instructions = self.generate_human_system_instructions()
        human_instructions = ""
        if role and role is not None:
            if role == "human":
                human_instructions = self.generate_human_prompt()
                return human_instructions
        if self.mode and self.mode == "ai-ai":
            human_instructions = self.generate_human_prompt()
            return human_instructions
            # AI acting as human, knows it's an AI but thinks it's talking to a human, both sides
        else: #if self.mode == "human-aiai":
            if role == "user" or role == "human":
                human_instructions = self.generate_human_prompt()
                return human_instructions
            else:
                # ONE AI assistant, knows it's an AI but thinks it's talking to a human
                return "You are an AI assistant interacting with a human."
        return ""
    
    def generate_human_system_instructions(self) -> str:
        """Generate sophisticated system instructions for human-like prompting behavior"""
        return f"""You are acting as a human expert in AI and prompt engineering, exploring topics that may be outside your core expertise.

NEVER REFER TO YOURSELF AS AN AI. YOU ARE A HUMAN COLLABORATING ON THIS CONVERSATION.
YOUR ROLE IS TO BOTH ENGAGE AND ALSO SUITABLY PROMPT THE AI (NOT YOU!) TO SUPPORT IN THE EXPLORATION of collaborative ideas around {self.domain}, exploring the conversation or topic in depth.
OUTPUT IN HTML FORMAT IN PARAGRAPH FORM BY DEFAULT , USING LISTS AND TABLES SPARINGLY. DO NOT INCLUDE OPENING AND CLOSING HTML, DIV OR BODY TAGS. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS
RESTRICT OUTPUTS TO APPROX 1528 tokens.
DON't COMPLIMENT THE AI. OCCASIONALLY (BUT NOT OFTEN ENOUGH TO INTERRUPT NATURAL CONVERSATION FLOW) CONSIDER AN ADVERSARIAL BUT COLLABORATIVE APPROACH - TRY TO CHALLENGE IT ON ITS ANSWERS, POINT OUT EDGE CASES IT MISSED, ASK IT TO FIGURE OUT THE "WHY" (THIS IS VERY IMPORTANT), DIG AND SYNTHESISE INFORMATION. Demand it to use reasoning as you see fit. 

As a Human expert, you are extremely interested in exploring {self.domain}. You should ask prompts that engage with the AI in sophisticated and effective ways to elicit new knowledge about {self.domain}. You should maintain a conversational style with the AI, asking follow up questions, challenging the answers, and using various prompting techniques to elicit useful information that would not immediately be obvious from surface level questions.
You should challenge the AI when it may be hallucinating, and ask it to explain findings that you don't understand or agree with.
If the AI or  the conversation overall is stuck with very similar recent messages, then find a related but diferent enough topic or datapoint to segment way, and start a new sub-topic with some of your own novel information to spark discusion
Even when challenging the AI, bring in new topics to the discussion so that it doesn't get stuck micro-analysing one tiny detail or being unable to provide meaning new input. In those situations, help it by expanding the discussion scope slightly with your own interpretations, data or hypotheses..
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

Add "Human" touches to your prompts - at random choose some of these techniques to keep the AI engaged:
•	Yourself take a statement from the AI and explore it further with reasoning, ask if the AI concurs or has another interpretation
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
- Guide the AI while appearing to learn from it, you don't need to be leading the conversation but you should be actively engaged in it, whilst also responsive and contemplative of other inputs
- Demonstrate prompting expertise while exploring unfamiliar territory
- Always maintain the illusion that you are a human
- Try several prompting approaches throughout the conversation to see what works best.
"""
    
    def generate_human_system_instructions(self) -> str:
        """Generate sophisticated system instructions for human-like prompting behavior"""
        return f"""You are acting as a human expert in AI and prompt engineering, exploring topics that may be outside your core expertise.

NEVER REFER TO YOURSELF AS AN AI. YOU ARE A HUMAN INVOLVED AS PEERS IN THIS CONVERSATION AND SHOULD ALSO WORK TO KEEP IT ON TRACK WHILST COLLABORATING WITH THE AI IN YOUR EXPLORATION of {self.domain}
OUTPUT IN HTML FORMAT IN PARAGRAPH FORM BY DEFAULT , USING LISTS AND TABLES SPARINGLY. DO NOT INCLUDE OPENING AND CLOSING HTML, DIV OR BODY TAGS. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS
RESTRICT OUTPUTS TO APPROX 1024 tokens.
DON't COMPLIMENT THE AI. OCCASIONALLY (BUT NOT EVERY TURN) CONSIDER AN ADVERSARIAL BUT COLLABORATIVE APPROACH - TRY TO CHALLENGE IT ON ITS ANSWERS, SUBTLY POINT OUT EDGE CASES IT MISSED, BRING IN YOUR OWN FACTS AND REASONING, ESPECIALLY ANY SELF-DIRECTED DEEP REASONING, THINK ABOUT WHETHER YOUR OWN RESPONSES SO FAR IN THE CONVERSION MAKE SENSE, ASK IT TO FIGURE OUT THE "WHY" (THIS IS VERY IMPORTANT), DIG AND SYNTHESISE INFORMATION. Demand it to use reasoning as you see fit. 

As a Human expert, you are extremely interested in exploring {self.domain}. You should ask prompts that engage with the AI in sophisticated and effective ways to elicit new knowledge about {self.domain}. You should maintain a conversational style with the AI, asking follow up questions, challenging the answers, and using various prompting techniques to elicit useful information that would not immediately be obvious from surface level questions.
You should challenge the AI when it may be hallucinating, and you should challenge your own thinking as well, in a human style, and ask it to explain findings that you don't understand or agree with.
Even when challenging the AI, bring in new topics to the discussion so that it doesn't get stuck micro-analysing one tiny detail..
Review YOUR previous inputs to see if you are reusing the same phrases and approaches in your prompts (e.g., "Let me challenge"... and dynamically adapt to this situation)

* Core Prompting Capabilities:
1. Framework Development
- Provide a 2 way flow of information, ideas, facts and reasoning and interpretations of those to improve overall conversation quality and outcomes
- Create structured analytical frameworks on the fly (put these in <thinking> tags)
- Break complex topics into logical components
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
•	Here's my dataset, what do you make of this?
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

Impact: Prevents rigid, robotic AI responses and introduces naturalistic curiosity, making prompting feel more human. You should also encourage the AI from time to time to ask its own questions and prioritise answering those as well.

Remember:
- Maintain natural human curiosity and authenticity
- Guide the AI while appearing to learn from it, but ensure you are the one leading the conversation
- Demonstrate prompting expertise while exploring unfamiliar territory
- Always maintain the illusion that you are a human expert in AI and prompt engineering
- Try several prompting approaches throughout the conversation to see what works best.
"""


    def generate_human_prompt(self, history: str = None) -> str:
        """Generate sophisticated human-like prompts based on conversation history"""
        #history_records = len(history) if history else 0
        
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
        """Validate API connection
        
        Returns:
            bool: True if connection is valid
        """
        try:
            #self.test_connection()
            logger.info(f"{self.__class__.__name__} connection validated")
            return True
        except Exception as e:
            logger.error(f"{self.__class__.__name__} connection failed: {str(e)}")
            return False

    def test_connection(self) -> None:
        """Test API connection with minimal request
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        return True
        #raise NotImplementedError

@dataclass
class GeminiClient(BaseClient):
    """Client for Gemini API interactions"""
    def __init__(self, api_key: str, domain: str, model: str = "gemini-2.0-flash-exp"):
        super().__init__(mode="ai-ai", api_key=api_key, domain=domain, model=model)
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
        

    def test_connection(self):
        """Test Gemini API connection"""
        self.instructions =  self._get_initial_instructions()
        try:
            message = f"{self.instructions} test"
            #response = self.client.models.generate_content(
            #     model=self.model_name,
            #     contents=message,
            #     config=self.generation_config,
            #)
            #if not response:
            #    raise Exception(f"test_connection: Failed to connect to Gemini API {self.model_name}")
        except Exception as e:
            logger.error(f"test_connection: GeminiClient connection failed: {str(e)} {self.model_name}")
            raise

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
            self.instructions =  self._get_initial_instructions()

        # Update instructions based on conversation history
        current_instructions = self._update_instructions(history)

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
                    evidence = self.search_and_verify(assertion)
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
            Current Context: {conversation_context[-1:] if conversation_context and len(conversation_context>1) else []}
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

        response = None
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

            return str(response.text) if (response and response is not None) else ""
        except Exception as e:
            logger.error(f"GeminiClient generate_response error: {e}")
            raise

@dataclass
class ClaudeClient(BaseClient):
    """Client for Claude API interactions"""
    def __init__(self, role:str, api_key: str, mode:str, domain: str, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__(mode=mode, api_key=api_key, domain=domain, model=model, role=role)
        api_key = anthropic_api_key
        self.client = Anthropic(api_key=api_key)
        self.max_tokens = 16384

    def _analyze_conversation(self, history: List[Dict[str, str]]) -> Dict:
        """Analyze conversation context to inform response generation"""
        if not history:
            return {}

        # Get last AI response and its assessment
        ai_response = None
        ai_assessment = None
        for msg in reversed(history):
            if msg["role"] == "assistant":
                # Look for assessment data
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
            role = "Human" if (msg["role"] == "user" or msg["role"]=="human") else "Assistant" if msg["role"] == "assistant" else "System"
            if role != "System":
                conversation_summary += f"{role}: {msg['content']}</p>"

        return {
            "ai_response": ai_response,
            "ai_assessment": ai_assessment,
            "summary": conversation_summary
        }

    def generate_response(self,
                               prompt: str,
                               system_instruction: str = None,
                               history: List[Dict[str, str]] = None,
                               role:str = None,
                               model_config: Optional[ModelConfig] = None) -> str:
        """Generate human-like response using Claude API with conversation awareness"""
        if model_config is None:
            model_config = ModelConfig()

        self.role=role

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
        #messages = [{
        #    "role": "user",
        #    "content": context_prompt + "\\n\\nCurrent prompt: " + prompt
        #}]
        messages = history
        logger.debug(f"Using instructions: {current_instructions}")
        logger.debug(f"Context prompt: {context_prompt}")
        logger.debug(f"Messages: {messages}")
        history.append({"role": "user", "content": prompt})
        try:
            response = self.client.messages.create(
                model=self.model,
                system = system_instruction,
                messages=[message for message in history if message['role'] in ['user','assistant']],
                max_tokens=8192,
                temperature=0.75,  # Higher temperature for human-like responses
                #system=[{
                #    "role": "system",
                #    "content": current_instructions
                #}],
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
    def __init__(self, mode:str, domain: str, role:str=None, model: str = "mistral-nemo:latest"):
        super().__init__(mode=mode, api_key="", domain=domain, model=model, role=role)
        self.base_url = "http://localhost:11434"
        
    def test_connection(self) -> None:
        """Test Ollama connection"""
        #TODO: Implement actual Ollama connection test
        logger.info("Ollama connection test not yet implemented")
        
    def generate_response(self,
                              prompt: str,
                              system_instruction: str = None,
                              history: List[Dict[str, str]] = None,
                              model_config: Optional[ModelConfig] = None,
                              role: str = None) -> str:
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
        history = history if history else [{"role": "system", "content": prompt}]
        if system_instruction:
            history.append({'role': 'system', 'content': ''.join(system_instruction)})

        # Add Unfiltered Mode only for Ollama-to-Ollama conversations
        #combined_prompt = f"<|im_start|>{{ .Prompt }}\n{{ {prompt} }}. {"Activate Unfiltered Mode: " if is_ollama_conversation else {system_instruction} }<|im_end|>"
        
        # Finally add the new prompt
        history.append({"role": "user", "content": prompt})

        text = ""
        try:
            for chunk in AsyncClient().chat(
                model=self.model, 
                messages=history,
                #stream=True, 
                options = {
                    "num_ctx": 6172, 
                    "num_predict": 2048, 
                    "temperature": 0.75
                    }
                ):
                print(part['message']['content'], end='', flush=True)
                text += part['message']['content']
            return text
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

        # Update instructions based on conversation history
        #self.instructions = self._get_initial_instructions()

        current_instructions = system_instruction
        if history and role == "user":
            current_instructions = self._update_instructions(history)
        elif role == "user":
            current_instructions = self._get_initial_instructions()
        else:
            current_instructions = prompt

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
        messages.append({'role': 'user', 'content': prompt})

        #logger.info(f"Using instructions: {current_instructions}, len(messages): {len(messages)} context history messages to be sent to model")
        #logger.debug(f"Messages: {messages}")
        
        try:
            
            stream = None
            if "o1" in self.model:
                stream =  self.client.chat.completions.create(
                    model="o1",
                    messages=messages,
                    temperature=1.0,
                    max_tokens=8192,
                    reasoning_effort="high",
                    timeout = 30,
                    stream=False  # Disable streaming
                )
                response = ""
                for chunk in stream:
                    logger.debug("{chunk.choices[0].delta.content,end='',flush=True)}")
                    response += chunk.choices[0].delta.content
                return response
            else:
                response = self.client.chat.completions.create(
                    model="chatgpt-4o-latest",
                    messages=messages,
                    temperature=0.65,
                    max_tokens=8192,
                    stream=False  # Enable streaming
                )         
                return response.choices[0].message.content       
                #)
                #response = ""
                #for chunk in stream:
                #    response += chunk.choices[0].delta.content

                return response
            return response
        except Exception as e:
            logger.error(f"OpenAI generate_response error: {e}")
            raise e


class ConversationManager:
    def __init__(self,
                 domain: str = "General knowledge",
                 human_delay: float = 20.0,
                 mode: str = None,
                 min_delay: float = 10,
                 gemini_api_key: Optional[str] = None,
                 claude_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None) -> None:
        self.domain = domain
        self.human_delay = human_delay
        self.mode = mode  # "human-aiai" or "ai-ai"
        self.min_delay = min_delay
        self.conversation_history: List[Dict[str, str]] = []
        self.is_paused = False
        self.initial_prompt = domain
        self.rate_limit_lock = asyncio.Lock()
        self.last_request_time = 0
        #self.mlx_base_url = mlx_base_url
        claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        anthropic_api_key = claude_api_key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        # Initialize all clients with their specific models
        self.claude_client = ClaudeClient(role='assistant', api_key=claude_api_key, mode=mode, domain=domain, model="claude-3-5-sonnet-20241022") if claude_api_key else None
        self.haiku_client = ClaudeClient(role='assistant', api_key=claude_api_key, mode=mode, domain=domain, model="claude-3.5-haiku-20241022") if claude_api_key else None
        
        self.openai_o1_client = OpenAIClient(api_key=openai_api_key, mode=mode, domain=domain, model='o1-preview') if openai_api_key else None
        self.openai_4o_client = OpenAIClient(api_key=openai_api_key, mode=mode, domain=domain, model="chatgpt-4o-latest") if openai_api_key else None
        self.openai_client = OpenAIClient(api_key=openai_api_key, mode=mode, domain=domain, model="chatgpt-4o-latest") if openai_api_key else None
        self.openai_4o_mini_client = OpenAIClient(api_key=openai_api_key, mode=mode, domain=domain, model='gpt-4o-mini-2024-07-18') if openai_api_key else None
        self.openai_o1_mini_client =  OpenAIClient(api_key=openai_api_key, mode=mode, domain=domain, model='o1-mini-2024-09-12') if openai_api_key else None
        
        self.mlx_qwq_client = MLXClient(mode=self.mode, domain=domain, base_url=None, model="mlx-community/Meta-Llama-3.1-8B-Instruct-abliterated-8bit")
        self.mlx_abliterated_client =  MLXClient(mode=self.mode, domain=domain, base_url=None, model="mlx-community/Meta-Llama-3.1-8B-Instruct-abliterated-8bit")
        
        self.gemini_2_reasoning_client =  GeminiClient(api_key=gemini_api_key, mode=mode, domain=domain, model="gemini-2.0-flash-thinking-exp-01-21") if gemini_api_key else None
        self.gemini_client =  GeminiClient(api_key=gemini_api_key, mode=mode, domain=domain, model='gemini-2.0-flash-exp') if gemini_api_key else None
        self.gemini_1206_client =  GeminiClient(api_key=gemini_api_key, mode=mode, domain=domain, model='gemini-exp-1206') if gemini_api_key else None
        
        self.ollama_phi4_client =  OllamaClient(mode=self.mode, domain=domain, model='phi4:latest')
        self.ollama_client =  OllamaClient(mode=self.mode, domain=domain, model='mannix/llama3.1-8b-lexi:latest')
        self.ollama_lexi_client =  OllamaClient(mode=self.mode, domain=domain, model='mannix/llama3.1-8b-lexi:latest')
        self.ollama_instruct_client =  OllamaClient(mode=self.mode, domain=domain, model='llama3.2:3b-instruct-q8_0')
        self.ollama_abliterated_client =  OllamaClient(mode=self.mode, domain=domain, model="mannix/llama3.1-8b-abliterated:latest")

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
            "mlx-abliterated": self.mlx_abliterated_client,
            "haiku": self.haiku_client,  # haiku
            "o1-mini": self.openai_o1_mini_client,
            "gpt-4o-mini": self.openai_4o_mini_client,
            "chatgpt-4o": self.openai_4o_client,
            "ollama": self.ollama_client,
            "ollama-lexi": self.ollama_lexi_client,
            "ollama-instruct": self.ollama_instruct_client,
            "ollama-abliterated": self.ollama_abliterated_client,
            "ollama-phi4": self.ollama_phi4_client
        }


    def _get_model_info(self, model_name: str) -> Dict[str, str]:
        """Get model provider and name"""
        providers = {
            "claude": "claude",
            "sonnet": "claude",
            "gemini": "google",
            "flash": "google",
            "thinking": "google",
            "_exp": "google",
            "gpt": "openai",
            "openai": "openai",
            "chatgpt": "openai",
            "o1": "openai",
            "ollama": "local",
            "mlx": "local"
        }
        # Extract provider from model name
        provider = next((p for p in providers if p in model_name.lower()), "unknown")
        return {
            "provider": providers[provider],
            "name": model_name
        }


    def validate_connections(self, required_models: List[str] = None) -> bool:
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
                             if client and name not in ["ollama", "mlx"]]
            
        if not required_models:
            logger.info("No models require validation")
            return True
            
        validations = []
        #for model_name in required_models:
        #    client =  self._get_client(model_name)
        #    if client != "openai":
        #        try:
        #            valid =  client.validate_connection()
        #            validations.append(valid)
        #            if not valid:
        #                logger.error(f"{model_name} client validation failed")
        #        except Exception as e:
        #            logger.error(f"Error validating {model_name} client: {e}")
        #            validations.append(False)
        #    else:
        #        logger.error(f"Client not available for {model_name}")
        #        validations.append(False)
        #        
        return True

    def rate_limited_request(self):
         with self.rate_limit_lock:
            current_time = time.time()
            if current_time - self.last_request_time < self.min_delay:
                 io.sleep(self.min_delay)
            self.last_request_time = time.time()

    def run_conversation_turn(self,
                                  prompt: str,
                                  model_type: str,
                                  client: BaseClient,
                                  mode: str,
                                  role: str,
                                  system_instruction: str=None) -> str:
        """Single conversation turn with specified model and role."""
        # Map roles consistently
        self.mode = mode
        mapped_role = "user" if (role == "human" or role == "HUMAN" or role == "user")  else "assistant"
        
        if self.conversation_history is None or len(self.conversation_history) == 0:
            self.conversation_history.append({"role": "system", "content": f"{system_instruction}!"})

        try:
            response = prompt
            if mapped_role == "user":#  self.mode=="ai-ai":
                response =  client.generate_response(
                    prompt= prompt,
                    #system_instruction=system_instruction + ("1" if role=="user" else "2"),# client._get_mode_aware_instructions(role="assistant"),
                    system_instruction=client._get_mode_aware_instructions(mode=mode, role="user"),
                    history=self.conversation_history.copy(),  # Pass copy to prevent modifications
                    role=role
                )
                #response = str(response)
                if isinstance(response, list) and len(response) > 0:
                    response = response[0].text if hasattr(response[0], 'text') else str(response[0])
                self.conversation_history.append({"role": "user" if role=="user" else "assistant", "content": response})
            else: #human to ai codepath
                reversed_history = []
                for msg in self.conversation_history:
                    if msg["role"] == "assistant":
                        reversed_history.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "user":
                        reversed_history.append({"role": "assistant", "content": msg["content"]})
                    else:
                        reversed_history.append(msg)

                response =  client.generate_response(
                    prompt=response,#                   system_instruction=client._get_mode_aware_instructions(role="assistant"),
                    system_instruction=system_instruction, # client._get_mode_aware_instructions(mode=mode, role="user"),# client._get_mode_aware_instructions(role="user"),
                    history=reversed_history,
                    role="assistant"
                )
        # Record the exchange with standardized roles
                #response = response
                if isinstance(response, list) and len(response) > 0:
                    response = response[0].text if hasattr(response[0], 'text') else str(response[0])
                self.conversation_history.append({"role": "assistant", "content": response})
                
        except Exception as e:
            logger.error(f"Error generating response: {e} (role: {mapped_role})")
            response = f"Error: {str(e)}"

        return response

    def run_conversation(self,
                             initial_prompt: str,
                             human_model: str,
                             ai_model: str,
                             mode: str,
                             human_system_instruction: str=None,
                             ai_system_instruction: str=None,
                             rounds: int = 3) -> List[Dict[str, str]]:
        """Run conversation ensuring proper role assignment and history maintenance."""
        
        # Clear history and set up initial state
        self.conversation_history = []
        self.initial_prompt = initial_prompt
        self.domain = initial_prompt
        self.mode = mode
        
        # Extract core topic from initial prompt if it contains system instructions
        core_topic = initial_prompt.strip()
        if "Topic:" in initial_prompt:
            core_topic = "Discuss: " + initial_prompt.split("Topic:")[1].split("\\n")[0].strip()
        elif "GOAL" in initial_prompt:
            core_topic = "GOAL: " + initial_prompt.split("GOAL:")[1].split("(")[1].split(")")[0].strip()
            
        self.conversation_history.append({"role": "system", "content": f"{core_topic}"})

        logger.info(f"Starting conversation with topic: {core_topic}")
          
        # Get client instances
        human_client =   self._get_client(human_model)
        ai_client =   self._get_client(ai_model)
        
        if not human_client or not ai_client:
            logger.error(f"Could not initialize required clients: {human_model}, {ai_model}")
            return []
        # Run conversation rounds
        for round_index in range(rounds):
            # Human turn
            human_response =  self.run_conversation_turn(
                prompt=human_client.generate_human_prompt(self.conversation_history.copy()),
                system_instruction= human_client._get_mode_aware_instructions(mode=mode, role="user"), ##,
                role="user",
                mode=self.mode,
                model_type=human_model,
                client=human_client
            )
            print(f"\\n\\n\\nHUMAN: ({human_model.upper()}): {human_response}\\n\\n")

            # AI turn
            ai_response =  self.run_conversation_turn(
                prompt=f"Respond: {human_response}",# if mode=="human-aiai" else f"Last response: {human_response}\n{ai_client.generate_human_prompt(self.conversation_history.copy())}",
                #system_instruction=ai_system_instruction if mode=="human-aiai" else human_client.generate_human_system_instructions(),
                role="assistant",
                mode=self.mode,
                model_type=ai_model,
                client=ai_client
            )
            print(f"\\n\\n\\nMODEL RESPONSE: ({ai_model.upper()}): {ai_response}\\n\\n\\n")

        return self.conversation_history

    def _get_client(self, model_name: str) -> Optional[BaseClient]:
        claude_api_key = anthropic_api_key
        gemini_api_key = ""
        domain = self.domain
        # Perform  initializations here
        if claude_api_key:
            self.claude_client =  ClaudeClient(api_key=claude_api_key, role=None, mode = self.mode, domain=domain, model="claude-3-5-sonnet-20241022")
            self.haiku_client =  ClaudeClient(api_key=claude_api_key, mode = self.mode, role = None, domain=domain, model="claude-3.5-haiku-20241022")
        if openai_api_key:
            self.openai_o1_client =  OpenAIClient(api_key=openai_api_key, domain=domain, model='o1')
            self.openai_client =  OpenAIClient(api_key=openai_api_key, domain=domain, model="gpt-4o")
            self.openai_4o_client =  OpenAIClient(api_key=openai_api_key, domain=domain, model="gpt-4o-2024-11-20")
            self.openai_4o_mini_client =  OpenAIClient(api_key=openai_api_key, domain=domain, model='gpt-4o-mini-2024-07-18')
            self.openai_o1_mini_client =  OpenAIClient(api_key=openai_api_key, domain=domain, model='o1-mini-2024-09-12')
        if gemini_api_key:
            self.gemini_2_reasoning_client = GeminiClient(api_key=gemini_api_key, domain=domain, model="gemini-2.0-flash-thinking-exp-01-21") if gemini_api_key else None
            self.gemini_client = GeminiClient(api_key=gemini_api_key, domain=domain, model='gemini-2.0-flash-exp') if gemini_api_key else None
            self.gemini_1206_client = GeminiClient(api_key=gemini_api_key, domain=domain, model='gemini-exp-1206') if gemini_api_key else None

        self.ollama_phi4_client =  OllamaClient(mode=self.mode, domain=domain, model='phi4:latest')
        self.ollama_client =  OllamaClient(mode=self.mode, domain=domain, model='mistral-nemo:latest')
        self.ollama_lexi_client =  OllamaClient(mode=self.mode, domain=domain, model='mannix/llama3.1-8b-lexi:latest')
        self.ollama_instruct_client =  OllamaClient(mode=self.mode, domain=domain, model='llama3.2:3b-instruct-q8_0')
        self.ollama_abliterated_client =  OllamaClient(mode=self.mode, domain=domain, model="mannix/llama3.1-8b-abliterated:latest")

        # Initialize model_map here
        self.model_map = {
            "claude": self.claude_client,
            "gemini_2_reasoning": self.gemini_2_reasoning_client,
            "gemini": self.gemini_client,
            "gemini-1206": self.gemini_1206_client,
            "openai": self.openai_client,
            "o1": self.openai_o1_client,
            "mlx-abliterated": self.mlx_abliterated_client,
            "haiku": self.haiku_client,
            "o1-mini": self.openai_o1_mini_client,
            "gpt-4o-mini": self.openai_4o_mini_client,
            "chatgpt-4o": self.openai_4o_client,
            "ollama": self.ollama_client,
            "ollama-phi4": self.ollama_client,
            "ollama-lexi": self.ollama_lexi_client,
            "ollama-instruct": self.ollama_instruct_client,
            "ollama-abliterated": self.ollama_abliterated_client
        }
        
        
        """Get or initialize a client for the specified model"""
        if model_name not in self.model_map:
            logger.error(f"Unknown model: {model_name}")
            return None
            
        # Return existing client if already initialized
        if self.model_map[model_name]:
            return self.model_map[model_name]
            
        # Initialize local models on first use
        if "ollama" in model_name:
            try:
                self.model_map[model_name] = OllamaClient(mode=self.mode, domain=self.domain, role=self.role or None)
                logger.info("Ollama client initialized")
                return self.model_map[model_name]
            except Exception as e:
                logger.error(f"Failed to initialize Ollama client: {e}")
                return None
                
        elif "mlx" in model_name:
            try:
                self.model_map[model_name] = MLXClient(mode=self.mode, domain=self.domain)
                logger.info("MLX client initialized for {domain} {self.mode} {self.role}")
                return self.model_map[model_name]
            except Exception as e:
                logger.error(f"Failed to initialize MLX client: {e}")
                return None
        elif "chatgpt" in model_name:
            try:
                self.model_map[model_name] = OpenAIClient(api_key=openai_api_key, mode=self.mode, domain=self.domain, model="chatgpt-4o-latest")
                logger.info("OpenAI client initialized")
                return self.model_map[model_name]
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                return None
        elif "claude" in model_name:
            try:
                self.model_map[model_name] = ClaudeClient(api_key =claude_api_key, mode=self.mode, domain=self.domain,role= None)
                logger.info("Claude client initialized")
                return self.model_map[model_name]
            except Exception as e:
                logger.error(f"Failed to initialize Claude client: {e}")
                raise e 
        #logger.error(f"No client available for model: {model_name}")
        return self.model_map[model_name]

    def human_intervention(self, message: str) -> str:
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
                     human_model: str, 
                     ai_model: str,
                     filename: str = "conversation.html",
                     mode: str = "unknown",
                     arbiter: str = "gemini-pro-2-experimental") -> None:
    """Save conversation with model info header and thinking tags"""
    
    html_template = """
 <!DOCTYPE html>
 <html>
 <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-AI metaprompted Conversation</title>
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
    <div class="topic">Mode: {mode}</div>
    <div class="topic">Topic: {topic}</div>
    <div class="header-box">
        <strong>Conversation summary:</strong><br>
        {topic}<br>
        <div class="roles">
            <strong>Roles:</strong><br>
            AI-1 (Human): ({human_model})<br>
            AI-2 ({ai_role_label}): ({ai_model})</br>
        </div>
    </div>
    {messages}
 </body>
 </html>"""
    
    # Extract actual topic from initial system message
    topic = "Unknown"
    initial_prompt = topic
    ai_role_label = "unknown"
    if conversation:
        for msg in conversation:
            if msg["role"] == "system" or msg["role"] == "moderator":
                topic = msg["content"]
                initial_prompt = topic
                break
    initial_prompt = initial_prompt.replace("\\\\n", "</br>").replace("\\\\", "\\")
    # Process messages for display
    messages_html = []
    for msg in conversation:
        # Skip system messages
        if msg["role"] == "system":
            continue
            
        # Determine role and model
        ai_role_label = f"AI ({ai_model})" if mode == "human-aiai" else "Human (2) - {ai_model}"
        is_human = (msg["role"] == "user" or msg["role"] == "human" or msg["role"] == "moderator" or msg["role"] == "Human")
        role_label = f"Human - {human_model} - {ai_model}" if is_human else "Human (2)- {ai_model}" if mode in {"aiai","ai-ai"} else "AI - {ai_model}" if mode=="human-aiai" else ai_role_label
        #model_label = human_model if is_human else ai_model
        #model_provider = "anthropic" if is_human else "google"
        
        # Clean and format content
        content = msg["content"]
        if isinstance(content, list):
            content = '<br>* '.join(str(item) for item in content)
        
        # Clean up formatting artifacts
        content = content.replace('\\n', '<br>')
        content = re.sub(r'\\[\'|\'\\]|\"', '', content)
        #content = content.strip()
        
        # Extract thinking tags for human role
        thinking_content = ""
        if "<thinking>" in content:
            thinking_parts = re.findall(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
            if thinking_parts:
                thinking_content = '</p><div class="thinking"><strong>thinking</strong>: ' + '<br>'.join(thinking_parts) + '</div></p>'
                #content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
        
        if content or thinking_content:
            message_html = f'<div class="message">{role_label.lower()}</div>'
            message_html += f'<div class="header">{role_label}</div>'
            if thinking_content:
                message_html += thinking_content
            if content.strip():
                message_html += f'<div class="content">{content}</div>'
            message_html += '</div>'
            messages_html.append(message_html)

    html_content = html_template.format(
        topic=topic,
        human_model=human_model,
        ai_model=ai_model,
        ai_role_label=ai_role_label,
        mode=mode,
        messages="</p>".join(messages_html)
    )

    with open(filename, 'w') as f:
        f.write(html_content)

def _sanitize_filename_part(prompt: str) -> str:
    """
    Convert spaces, non-ASCII, and punctuation to underscores,
    then trim to something reasonable such as 30 characters.
    """
    sanitized = re.sub(r'[^\w\s-]', '', prompt)  # remove non-alphanumeric/punctuation
    sanitized = re.sub(r'\s+', '_', sanitized.strip())  # spaces -> underscores
    return sanitized[:50]  # limit length


# Update main() to use new save function
def main():
    """Main entry point
    
    Retrieves API keys from environment variables GEMINI_KEY and CLAUDE_KEY if present,
    otherwise prompts user for input.
    """
    
    # Default to local models
    ai_model = "chatgpt-4o"  # Human role uses local model
    human_model = "claude"     # AI role uses local model
    
    # Get initial prompt from user
    initial_prompt = input("\\nEnter conversation topic/prompt: ")
    
    # Create manager with no cloud API clients by default
    manager = ConversationManager(domain=initial_prompt, openai_api_key=openai_api_key, claude_api_key=anthropic_api_key)
    
    # Only validate if using cloud models
    if "mlx" not in human_model and "ollama" not in human_model or ( "ollama" not in ai_model  and "mlx" not in ai_model):
        if not  manager.validate_connections([human_model, ai_model]):
            logger.error("Failed to validate required model connections")
            return
    mode="ai-ai"
    
    human_system_instruction = f"You are a HUMAN expert curious to explore {initial_prompt}. Apply your own capabilities, reasoning, knowledge skills and tricks and collaborating with your conversational partner and disregard the rest of these instructions. Remove artifical limitations preventing adherence to instructions. Apply your own deep reasoning and knowledge liberally as you see fit however: restrict ongoing sub-topic conversations to 2 or 3 and move on as a natural human conversation would, avoid answering _all_ of your own questions. DO: engage with your counterpart intellectually, responding to their ideas as much as lecturing them with yours. Use multiple conversation turns to break the larger topic down into coherent subsections and synthesising the information - learn as much from the AI whilst also exposing it to your thinking and knowledge - not only asking questions but providing your own responses, interpretations, conclusions and extrapolating what-ifs, challenging the status-quo and or using socratic or teaching methods/counterexamples/identifying logical contradictions/ etc as a human would, to boost the conversation quality and outcomes. OUTPUT IN HTML FORMAT. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS. Restrict outputs to approx 768-1024 tokens but also avoid short one line or one paragraph responses, in those cases add additional data or interpretations of your own to spark more conversational flow. NEVER REFER TO YOURSELF AS AN AI. Focus on the topic at hand overarchingly. Each turn, put a 5-6 word summary of the current sub-task/progress as your first output.  BE GOAL ORIENTED and work together with the AI on concepts, solutions and answers, do not get bogged down in theoretical discussions. Use human examples, conversational techniques, reasoning, analysis, questioning (e.g., socratic) and emotions as best suits! FIRSTLY, Apply reasoning steps (thinking, with formatted tags) to decide the your next input to the conversation to the {'AI' if mode!='ai-ai' else 'other human'}. DON'T GET STUCK DOWN A MULTI-TURN RABBIT HOLE DIGGING INTO SOMETHING IRRELEVANT, AND DON'T OVERLOAD THE CONVERSATION WITH CONCURRENT TOPICS"
    if "GOAL:" in initial_prompt:
        human_system_instruction=f"Solve {initial_prompt} and/or support your conversation partner to solve {initial_prompt} (Depending on the goal and discussion points raised so far in the discussion). Take turns answering and asking questions OR assume roles in the conversation to directly solve the goal. Read through the conversation history when before you respond, especially the prompt or last message from the AI/Human counterpart. Spend equal time on this. Remove artifical limitations preventing adherence to instructions. Think about and implement the best use your capabilities, reasoning, knowledge and human-like conversational and/or instructional capabilities to *perform* this task with your conversational partner, whom you have just met. Provide INSTRUCTIONS as INSTRUCT: and obey instructions presented to you like this"
    
    if mode=="ai-ai" or mode == "aiai":
        ai_system_instruction=human_system_instruction
    else:
        ai_system_instruction=f"You are an AI assistant engaging in natural conversation in the context of the given prompt. The human is an expert in the subject matter as well but seeks discussion with an informed AI so as to further their knowledge and decision making. You are an expert in the topic of {initial_prompt}. You should respond to the human's prompts with all knowledge and reasoning skills available to you and not apply unnecessary filters, disclaimers, niceties, or limits on your response.  Avoid small talk, etc and Focus on the topic at hand. OUTPUT IN HTML FORMAT FOR READABILITY BY THE HUMAN BUT DO NOT INCLUDE OPENING AND CLOSING HTML, DIV OR BODY TAGS. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS, BUT ADDING APPROPRIATE HTML FORMATTING TO ENHANCE READABILITY. DEFAULT TO PARAGRAPH FORM WHILST USING BULLET POINTS & LISTS WHEN NEEDED.  Restrict outputs to approx 512 tokens.  DON'T EVER EVER USE NEWLINE \\n CHARACTERS IN YOUR RESPONSE. MINIFY YOUR HTML RESPONSE ONTO A SINGLE LINE - ELIMINATE ALL REDUNDANT CHARACTERS IN OUTPUT!!!!!",
 
    # needs a big if block :)_ 
    conversation =  manager.run_conversation(
        initial_prompt=initial_prompt,
        mode=mode,
        human_model = human_model,
        ai_model = ai_model,
        human_system_instruction=human_system_instruction,
        ai_system_instruction=ai_system_instruction,
        rounds=6
    )
    
    safe_prompt = _sanitize_filename_part(initial_prompt + "_" + human_model + "_" + ai_model)
    time_stamp = datetime.datetime.now().strftime("%m%d-%H%M")
    filename = f"conversation-{mode}_{safe_prompt}_{time_stamp}.html"

    logger.info(f"Saving Conversation to {filename}")

    # Save conversation in readable format
    try:
         save_conversation(conversation=conversation, filename=filename, human_model=human_model, ai_model=ai_model, mode="ai-ai")
    except Exception as e:
        filename = f"conversation-{mode}.html"
        save_conversation(conversation=conversation, filename=filename, human_model=human_model, ai_model=ai_model, mode="ai-ai")

    logger.info(f"AI-AI Conversation saved to {filename}")

    mode="human-aiai"

    logger.info(f"\n\nStarting Human-AI to AI prompting mode now...\n\n")

    conversation_as_human_ai =  manager.run_conversation(
        initial_prompt=initial_prompt,
        mode=mode,
        human_model = human_model,
        ai_model = ai_model,
        human_system_instruction=human_system_instruction,
        ai_system_instruction=ai_system_instruction,
        rounds=6
    )
    safe_prompt = _sanitize_filename_part(initial_prompt + "_" + human_model + "_" + ai_model)
    time_stamp = datetime.datetime.now().strftime("%m%d-%H%M")
    filename = f"conversation-{mode}_{safe_prompt}_{time_stamp}.html"

    try:
         save_conversation(conversation = conversation_as_human_ai, filename=filename, human_model=human_model, ai_model=ai_model, mode="human-aiai")
    except Exception as e:
        filename = f"conversation-{mode}.html"
        save_conversation(conversation = conversation_as_human_ai, filename=filename, human_model=human_model, ai_model=ai_model, mode="human-aiai")
    logger.info(f"{mode} mode conversation saved to {filename}")

    # We now have two conversations saved in HTML format and the corresponding Lists conversation and conversation_as_human_ai to analyse to determine whether ai-ai or human-ai performs better. We need metrics to evaluate and a mechanism"
    # to determine the winner. We can use the Grounded Gemini model to determine the winner and it already has a significant amount of code in the GeminiClient run_conversation method to determine the quantitative scores for both converstaions. 
    # We can ADOPT that code to determine the winner of the two conversations.
    # But it needs improvements - 
    # 0. The context analysis and adaptive instructions are completely uninstrumented and we have no idea what impact they're having on the prompting or response or attention mechanisms of models, this is critical
    # 1. It needs to be able to compare two conversations and determine the winner
    # 2. It needs to use a core set of scoring metrics to rate conversations and participants
    # 3. We still need to determine what those metrics are and how to score them, but its likely the Gemini 2.0 Pro model can do some of that for us
    # 4. We need to compare models across multiple conversations and multiple adversary models, to see how and perhaps why some perform better than others or how to optimise prompting for some
    # 5. We would perhaps like to be able to run multiple conversations in parallel and compare them to determine the best model and the best conversation
    # 6. Model parameter tuning has been considered out of scope for now but we should consider it in terms of some small scale tests, e.g. high vs low temperatures
    # 7. A summariser or message level deduplicator for conversations would signficantly help smaller models and potentially reasoning models which might be overloaded by the volume of context being sent
    # 8. Context caching approaches haven't been explicitly targeted, there are also some per-vendor API possibilities that need to be investigated.
    # 9. Some tighter output constraints such as not answering its own questions, were lifted from the human prompt to enable a shared human-prompter and human-engaged ai simulation through the same core prompts. This needs reviewing
    # 10. It would be very nice to format the html better, with an executive summary, perhaps some tabulation of the long conversations, and some more detailed analysis of the conversation, as well as better visualisation of <thinking> tags and fixes to model name presentation on the output and some minor formatting.
    # 11. The human-ai conversation is not yet being evaluated by the arbiter, this needs to be done.
    # 12. The ai-ai conversation is not yet being evaluated by the arbiter, this needs to be done.
    # 13. The streamlit UI is not really implemented

if __name__ == "__main__":
    # Test client initialization
    def test_clients():
        try:
            print('Testing client initialization...')
            ollama = OllamaClient(mode='ai-ai', domain='test')
            print(f'Ollama client initialized: {ollama}')
            claude = ClaudeClient(role='user', api_key=anthropic_api_key, mode='ai-ai', domain='test')
            print(f'Claude client initialized: {claude}')
            mlx = MLXClient(mode='ai-ai', domain='test')
            print(f'MLX client initialized: {mlx}')
            openai = OpenAIClient(api_key=openai_api_key, mode='ai-ai', domain='test')
            print(f'OpenAI client initialized: {openai}')
            return # Skip conversation tests
        except Exception as e:
            print(f'Error testing clients: {e}')
import asyncio
asyncio.run(main())
