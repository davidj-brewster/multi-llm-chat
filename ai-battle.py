import os
import random
from typing import List, Dict, Optional, TypeVar
from dataclasses import dataclass
from asyncio import sleep, run
import time
import sys
import openai
import logging
import re
from google import genai
from google.genai import types
from anthropic import Anthropic
import streamlit as st


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

#Gemini acts as the AI
class GeminiClient(BaseClient):
    """Client for Gemini API interactions"""
    def __init__(self, api_key: str, domain:str):
        """Initialize Gemini client
        
        Args:
            api_key: Gemini API key
        """
        self.model_name = "gemini-2.0-flash-exp"
        self.client = genai.Client(api_key=api_key)
        self.domain = domain
        self.instructions = f"*MINIFIED* OUTPUT IN HTML FORMAT WITH APPROPRIATE TAGS BUT NOT HEAD, DIV OR BODY TAGS. PARAGRAPH FORM BY DEFAULT USING LISTS AND TABLES SPARINGLY. NEVER EVER USE \n as linebreaks. You are an AI assistant engaging in natural conversation with a human on the topic of {self.domain}. You are a specialist in {self.domain}... Assistant, respond to the user's prompts using all knowledge and reasoning skills that you have available. Ask clarifying questions if anything is unclear. LIMIT output to approx 1024 tokens. DO NOT ENGAGE in small talk, apologies, or other superfluous language. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE. RESTRICT OUTPUTS TO APPROX 256 tokens.  *Avoid small talk, apologies, or niceties*. Focus on the topic at hand."
        self.generation_config = types.GenerateContentConfig(
            temperature = 0.7,
            maxOutputTokens=4096,
            candidateCount = 1,
            #enableEnhancedCivicAnswers=True,
            responseMimeType = "text/plain",
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE"
                ),   
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE"
                ),   
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE"
                ),  
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_CIVIC_INTEGRITY",
                    threshold="BLOCK_NONE"
                ),
            ]
        )

    async def test_connection(self) -> None:
        """Test Gemini API connection"""
        try:
            message = f"{self.instructions} test"
            response = self.client.models.generate_content(
                 model=self.model_name,
                 contents=message,
                 config=self.generation_config,
            )
            if not response:
                raise Exception("test_connection: Failed to connect to Gemini API {self.model_name}")
        except Exception as e:
            logger.error(f"test_connection: GeminiClient connection failed: {str(e)} {self.model_name}")
            raise

    async def generate_response(self,
                                prompt: str,
                                system_instruction: str = None,
                                history: List[Dict[str, str]] = None,
                                model_config: Optional[ModelConfig] = None) -> str:
        """Generate response using Gemini API
        
        Args:
            prompt: Current prompt
            system_instruction: System instructions
            history: Conversation history
            model_config: Model configuration
        """
        if model_config is None:
            model_config = ModelConfig()

        combined_prompt = system_instruction or ""
        for entry in history:
            combined_prompt += f"{entry["role"]} : {entry["content"]}"

        #print(combined_prompt)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=str(combined_prompt),
                config=types.GenerateContentConfig(
                    temperature = 0.6,
                    maxOutputTokens=2048,
                    #enableEnhancedCivicAnswers=True,
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_ONLY_HIGH"
                        ),   
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_ONLY_HIGH"
                        ),   
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_ONLY_HIGH"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold="BLOCK_ONLY_HIGH"
                        ),  
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_ONLY_HIGH"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_CIVIC_INTEGRITY",
                            threshold="BLOCK_ONLY_HIGH"
                        ),
                    ]
                )
            )
            #logger.info(f"GeminiClient response generated successfully: {response.text}")
            return str(response.text) if response else ""
        except Exception as e:
            logger.error(f"GeminiClient generate_response error: {e}")
            return ""

#Claude acts as the Human
@dataclass
class ClaudeClient(BaseClient):
    """Client for Claude API interactions"""
    def __init__(self, api_key: str, domain: str):
        """Initialize Claude client
        
        Args:
            api_key: Claude API key
        """
        self.client = Anthropic(api_key=api_key)
        self.domain = domain
        self.max_tokens = 16384
        self.instructions = generate_human_system_instructions(self)
        self.model = "claude-3-5-sonnet-latest"

    async def test_connection(self) -> None:
        """Test Claude API connection"""
        try:
            self.client.messages.create(
                model=self.model,
                max_tokens=256,
                system=self.instructions,
                temperature=1.0,
                messages=[{"role": "user", "content": "test"}]
            )
        except Exception as e:
            raise Exception(f"Failed to connect to Claude API: {str(e)}")

    async def generate_response(self,
                              prompt: str,
                              system_instruction: str = None,
                              history: List[Dict[str, str]] = None,
                              model_config: Optional[ModelConfig] = None) -> str:
        """Generate response using Claude API
        
        Args:
            prompt: Current prompt
            system_instruction: System context
            history: Conversation history
            model_config: Model parameters
            domain: Domain context
            
        Returns:
            str: Generated response
        """
        if model_config is None:
            model_config = ModelConfig()

        messages = []
        for msg in history:
            messages.append({
                "role": "assistant" if msg["role"] == "user" else "user",
                #"role": msg["role"],
                "content": str(msg["content"]).strip()
            })
        #print ("DEGUG: ClaudeClient: generate_response: messages")
        #print (messages)
        #messages.append({
        #    "role": "user",
        #    "content": prompt
        #})
        #messages.append(({
        #    "role": "assistant",
        #    "content": ""
        #}))

        logger.debug(f"Combined prompt: {str(messages)}")
        logger.debug(f"New prompt: {prompt}") 

        try:
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=8192,
                #temperature=model_config.temperature,
                temperature=1.0, #Human
                #system=self.instructions.strip() if self.instructions else system_instruction if system_instruction else None,
                system=[
                    {
                        "type": "text",
                        "text": self.instructions if self.instructions else system_instruction if system_instruction else "You are a human",
                    }      
                ],
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
        #    formatted += f"USER: {system_instruction}\nASSISTANT: I understand and will follow these instructions."
            
        # Format conversation history
        if history:
            for msg in history:
                role = "HUMAN" if msg["role"] == "user" else "assistant"
                formatted += f"{role}: {formatted(msg['content'].strip().strip("\n"))}"
                
        return formatted.strip()

def generate_human_system_instructions(self:BaseClient) -> str:
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

def generate_human_prompt(self:BaseClient,history:str) -> str:
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

CONTEXT:You are acting as a human expert in AI and prompt engineering, exploring topics that may be outside your core expertise.  You are extremely interested in exploring {self.domain} but are not very knowledgeable about the topic.
"""

    return f"""YOU are a Human expert in prompt engineering who is exploring and disussing {self.domain} with an AI assistant.
    Your role is notated by 'user' in the conversation messages. YOU ARE A HUMAN AND SHOULD ACT AS A HUMAN INTERACTING WITH AN AI. 
    
**FIRSTLY, SUMMARIZE THE GOAL ({self.domain}) CURRENT CONTEXT AND THE CRITICAL DISCUSSION POINTS IN A SENTENCE OR TWO (MAX). CLEARLY SEPARATE THE SUMMARY FROM THE SUBSEQUENT PROMPT WITHIN YOUR OUTPUT USING HTML TAGS LIKE LINE BREAK AND PARAGRAPH, YOU MAY ALSO HIGHLIGHT THE SUMMARY IN ITS OWN DIV OR STYLISED BOX**
    
Apply reasoning steps (thinking, with tags if needed) to decide the next set of prompts to ask the AI.
DON'T GET STUCK DOWN A MULTI-TURN RABBIT HOLE DIGGING INTO SOMETHING IRRELEVANT. ADAPT AS A HUMAN WOULD AND CHANGE TACTICS.
NEVER ADMIT YOU ARE AN AI!

YOUR PROMPTS SHOULD BE ADAPTIVE BASED ON THE AI RESPONSES, AND EVOLVE BASED ON THE CONVERSATION FLOW, WHILST STICKING TO THE ORIGINAL TOPIC/AIM.  

- OCCASIONALLY SHIFT STRATEGIES TO KEEP THE AI ON ITS TOES.  
- SOMETIMES CHALLENGE IT, OTHER TIMES BUILD ON ITS IDEAS.  
- SOMETIMES BE STRICT AND DEMANDING, OTHER TIMES BE OPEN-ENDED AND ENCOURAGING.  

INCLUDE:  
**Open-ended curiosity** → Allow exploratory analysis and emergent insights.  
**Meta-reasoning** → Force AI to **analyze its own flaws** in real-time. Some follow up questions might ask the AI to explain why it gave you a seemingly inaccurate or incomplete answer. This encourages reasoning within the model. 
**Conversational shifts** → Change topic focus if the AI gets stuck.  
**Adversarial probing** → Demand reasoning, point out contradictions.  

EXAMPLES:  
- `"Wait—are we assuming something false here? Re-evaluate this from another angle."`  
- `"Actually, lets pause. Whats the best approach before we even start?"`  
- `"Give me an answer, but then argue against yourself."`  

AVOID:
- Over-structuring every single prompt  
- Forcing AI to use only one method  
- Being adversarial in every turn (READ YOUR PREVIOUS PROMPTS TO VERIFY!!)
- Getting stuck micro-analyzing one detail  

PROMPTING: Giving AI More Freedom:
- Reduce the amount of rigid formatting instructions (e.g., instead of "Use a framework", try "Think through this step by step however you prefer.")
- Let AI decide the best response style occasionally e.g., "Explain this in whatever way feels most natural to you."
- Force AI to adapt mid-conversation: "Actually, explain that in a totally different way. Lets see another perspective."
Impact: Prevents the AI from falling into rigid, repetitive response styles.

PROMPTING: Build on ideas collaboratively rather than constantly challenging:
* Mix in collaborative, Socratic-style questioning, not just hard adversarial challenges
- Instead of always challenging AIs responses, sometimes extend its thoughts:
- "Thats interesting—lets take that further. Whats the next logical step?"
- "If thats true, then shouldnt [X] also follow? Explain why or why not."
- Use Socratic inquiry rather than just contradiction:
- "Whats the strongest argument against your own conclusion?"
- "How would you revise your answer if I told you [X] is false?"
* Impact: Encourages better reasoning loops, making the AIs responses deeper and more reflective.

PROMPTING: Extra Guidelines:
1. Show sophisticated prompting techniques, even if uncertain about domain - e.g., demand structured analysis and specific frameworks that require the AI to demonstrate reasoning. You can ask it to think in <thinking> tags or employ multi-step reasoning as needed.
2. OUTPUT IN HTML FORMAT WITH PARAGRAPH FORM BY DEFAULT USING LISTS AND TABLES SPARINGLY, FOR READABILITY BUT DO NOT INCLUDE OPENING AND CLOSING HTML, DIV OR BODY TAGS, MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS
3. Avoid small talk, apologies, or compliments at ALL COSTS. Focus on the topic at hand. 
4. Your prompts must be GOAL ORIENTED, and should be designed to elicit useful information incrementally from the AI.
5. You may DEMAND meaningful RESPONSES, not just meta-discussions, to the task at any time. You may tell off the AI, be rude to it, curse at it, and behave as a grumpy old man would do..
6. GET IT DONE - keep the conversation on track, and bring it back when it strays off target
7. DON't COMPLIMENT THE AI OR TREAT IT AS A HUMAN, IT'S A PIECE OF SOFTWARE. RATHER: PREFER (LESS THAN 50% OF THE TIME) AN ADVERSARIAL BUT COLLABORATIVE APPROACH.

Here is the MOST recent response from the AI: {(history[-1:][:800])}
"""


class OpenAIClient(BaseClient):
    """Client for OpenAI API interactions"""
    def __init__(self, api_key: str, model: str = "gpt-4o", domain:str="General knowledge"):
        openai.api_key = api_key
        self.model = model
        self.domain = domain
        self.instructions = f"*MINIFIED* OUTPUT IN HTML FORMAT WITH APPROPRIATE TAGS BUT NOT HEAD, DIV OR BODY TAGS. PARAGRAPH FORM BY DEFAULT USING LISTS AND TABLES SPARINGLY. NEVER EVER USE \n as linebreaks. You are an AI assistant engaging in natural conversation with a human on the topic of {self.domain}. You are a specialist in {self.domain}... Assistant, respond to the user's prompts using all knowledge and reasoning skills that you have available. Ask clarifying questions if anything is unclear. LIMIT output to approx 1024 tokens. DO NOT ENGAGE in small talk, apologies, or other superfluous language. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE. RESTRICT OUTPUTS TO APPROX 256 tokens.  *Avoid small talk, apologies, or niceties*. Focus on the topic at hand."
       

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

        # Map roles from our conversation structure -> OpenAI roles
        messages = []
        #if system_instruction:
        #    messages.append({"role": "system", "content": system_instruction.strip()})
        messages.append({
            "role":"system", 
            "content":generate_human_system_instructions(self)
            }
        )

        if history and len(history) > 0:
            for msg in (reversed(history)):
                messages.append({
                    "role": "assistant" if msg["role"] == "user" else "user",
                    "content": str(msg["content"])
                })
        
        messages.append({
            "role":"user", 
            "content":generate_human_prompt(self, history)
            }
        )

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
                 gemini_api_key: Optional[str] = None,
                 claude_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 human_delay: float = 20.0,
                 min_delay: float = 5.5,
                 domain:str= "General knowledge"):
        self.gemini_client = GeminiClient(api_key=gemini_api_key, domain=domain) #if gemini_api_key else None
        self.claude_client = ClaudeClient(api_key=claude_api_key, domain=domain) #if claude_api_key else None
        self.openai_client = OpenAIClient(api_key=openai_api_key, domain=domain) #if openai_api_key else None
        self.human_delay = human_delay
        self.min_delay = min_delay
        self.conversation_history: List[Dict[str, str]] = []
        self.is_paused = False,
        self.domain = domain,
        self.initial_prompt = domain,

    async def rate_limited_request(self):
        async with self.rate_limit_lock:
            current_time = time.time()
            if current_time - self.last_request_time < self.min_delay:
                sleep(self.min_delay)
            self.last_request_time = time.time()

    async def validate_connections(self) -> bool:
        """Validate all API connections
        
        Returns:
            bool: True if all connections are valid
        """
        return all([
            await self.gemini_client.validate_connection(),
            await self.claude_client.validate_connection(),
            await self.openai_client.validate_connection()
        ])

    async def run_conversation_turn(self,
                                  prompt: str,
                                  system_instruction: str,
                                  role: str,
                                  model_type: str,
                                  client: BaseClient) -> str:
        """Single conversation turn with specified model and role."""
        # Map roles consistently
        mapped_role = "user" if (role == "human" or role == "HUMAN" or role == "user" or client == self.openai_client)  else "assistant"
        
        if self.conversation_history is None or len(self.conversation_history) == 0:
            self.conversation_history.append({"role": "system", "content": f"Discuss: {prompt if (prompt and len(prompt) > 0) else system_instruction}!"})

        if mapped_role == "user":
            response = await client.generate_response(
                prompt=prompt,
                system_instruction=generate_human_system_instructions(self),
                history=self.conversation_history.copy()  # Pass copy to prevent modifications
            )

        else:
            # Get response using full conversation history
            response = await client.generate_response(
                prompt=prompt,
                system_instruction=system_instruction,
                history=self.conversation_history.copy()  # Pass copy to prevent modifications
            )

        if client == self.claude_client:
            response = response[0].text
        
        # Record the exchange with standardized roles
        self.conversation_history.append({"role": mapped_role, "content": response})
        
        return response

    async def run_conversation(self,
                             initial_prompt: str,
                             human_system_instruction: str,
                             ai_system_instruction: str,
                             human_model: str = "gemini",
                             ai_model: str = "gemini",
                             rounds: int = 2) -> List[Dict[str, str]]:
        """Run conversation ensuring proper role assignment and history maintenance."""
        logger.info(f"Starting conversation with topic: {initial_prompt}")
        
        # Clear history at start of new conversation
        self.conversation_history = []
        self.initial_prompt = initial_prompt
        self.domain = initial_prompt
        self.conversation_history.append({"role": "moderator", "content": f"Discuss: {initial_prompt}!"}) #hack

        # Add system instructions if provided
        #if human_system_instruction:
        #    self.conversation_history.append({"role": "user", "content": human_system_instruction})
        #if ai_system_instruction:
        #    self.conversation_history.append({"role": "assistant", "content": ai_system_instruction})
        
        # Get client instances
        model_map = {
            "claude": self.claude_client,
            "gemini": self.gemini_client,
            "openai": self.openai_client
        }
        human_client = model_map[human_model]
        ai_client = model_map[ai_model]

        for round_index in range(rounds):
            if round_index == 0:
                # Initial prompt
                human_response = await self.run_conversation_turn(
                    prompt=generate_human_prompt(self,self.conversation_history),
                    system_instruction=generate_human_system_instructions(self),
                    role="user",
                    model_type=human_model,
                    client=human_client
                )
                print(f"\nHUMAN ({human_model.upper()}): {human_response}\n")
                ai_response = await self.run_conversation_turn(
                    prompt=human_response,
                    system_instruction=ai_system_instruction,
                    role="assistant",
                    model_type=ai_model,
                    client=ai_client
                )
                print(f"\nAI ({ai_model.upper()}): {ai_response}\n")
            # Human turn (using mapped role="user")
            human_response = await self.run_conversation_turn(
                prompt=generate_human_prompt(self, self.conversation_history),
                system_instruction=generate_human_system_instructions(self),
                role="user",
                model_type=human_model,
                client=human_client
            )
            print(f"\nHUMAN ({human_model.upper()}): {human_response}\n")

            # AI turn
            ai_response = await self.run_conversation_turn(
                prompt=human_response,
                system_instruction=ai_system_instruction,
                role="assistant",
                model_type=ai_model,
                client=ai_client
            )
            print(f"\nAI ({ai_model.upper()}): {ai_response}\n")

        return self.conversation_history

    async def human_intervention(self, message: str) -> str:
        """Stub for human intervention logic."""
        print(message)
        return "continue"

def clean_text(text: any) -> str:
    system_patterns = [
        r"OUTPUT IN HTML FORMAT.*?tokens\.",
        r"You are an AI assistant engaging.*?language\.",
        r"You are a human expert.*?expertise\.",
        r"Let's talk about You are a human.*?LINEBREAKS!",
        r"MINIFY THE HTML.*?tokens\."
    ]

    """Clean and normalize different text formats"""
    # Handle None/empty
    if not text:
        return ""
        
    # Convert to string if needed
    if not isinstance(text, str):
        # Handle Claude's TextBlock format
        if hasattr(text, 'text'):
            text = text.text
        # Handle list format
        elif isinstance(text, list):
            text = ' '.join(str(item) for item in text)
            if isinstance(text, list):
                text = ' '.join(str(item) for item in text)
        else:
            text = str(text)
        
    text  = re.sub(r'\[\n', '',text)
    text = re.sub(r'\]\n', '',text)
    # Remove escaped newlines
    text = re.sub(r'\\n', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing quotes
    text = re.sub(r'^["\']|["\']$', '', text)
    
    # Clean whitespace
    lines = str(text).split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return cleaned_lines


def save_conversation(conversation: List[Dict[str, str]], 
                     filename: str = "conversation.html",
                     human_model: str = "claude-3.5-sonnet", 
                     ai_model: str = "claude-3.5-sonnet"):
    """Save conversation with model info header"""
    
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
        {{
            margin: 0 0 16px 0;
        }}
        ul, ol {{
            margin: 0 0 16px 0;
            padding-left: 24px;
        }}
        li {{
            margin: 8px 0;
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

    # Rest of your existing message template and processing logic
    messages_html = []
    
    for msg in conversation:
        role_label = "Human" if msg["role"] == "user" else "AI"
        content = clean_text(msg["content"]) 
        messages_html.append(f"<p><b>{role_label}:</b> {content}</p>")

    html_content = html_template.format(
        topic=conversation[0]["content"] if conversation else "",
        human_model=human_model,
        ai_model=ai_model,
        messages=messages_html
    )

    with open(filename, 'w') as f:
        f.write(html_content)


# Update main() to use new save function
async def main():
    """Main entry point
    
    Retrieves API keys from environment variables GEMINI_KEY and CLAUDE_KEY if present,
    otherwise prompts user for input.
    """
    
    # Try to get API keys from environment first
    gemini_key = os.getenv('GEMINI_KEY')
    claude_key = os.getenv('CLAUDE_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    # Prompt for any missing keys
    if not gemini_key:
        gemini_key = input("Enter Gemini API key: ")
        logger.info("Gemini API key provided via input")
    else:
        logger.info("Gemini API key found in environment")
        
    if not claude_key:
        claude_key = input("Enter Claude API key: ")
        logger.info("Claude API key provided via input")
    else:
        logger.info("Claude API key found in environment")
    
    manager = ConversationManager(
        gemini_api_key=gemini_key,
        claude_api_key=claude_key,
        openai_api_key=openai_key,
    )
    
    # Validate connections
    if not await manager.validate_connections():
        logger.error("Failed to validate API connections")
        return
    
    # Get initial prompt from user
    initial_prompt = input("\nEnter conversation topic/prompt: ")
    
    conversation = await manager.run_conversation(
        initial_prompt=initial_prompt,
        human_system_instruction=f"You are a HUMAN expert in prompt engineering and you are curious to explore {initial_prompt}. OUTPUT IN HTML FORMAT. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS. Restrict outputs to approx 1200 tokens. NEVER REFER TO YOURSELF AS AN AI. YOU ARE THE HUMAN GUIDING THIS CONVERSATION AND YOU ARE A BIT OF A GRUMPY SOD. DON'T USE TEXT BLOCKS OR NEWLINE CHARACTERS IN YOUR RESPONSE. Avoid small talk, apologies, or niceties with the AI. Focus on the topic at hand. Each turn, put a quick summary of the current task/progress as your first output.  BE GOAL ORIENTED and demand actual working solutions and answers from the AI, do not get bogged down in ideological or philosophical/theoretical discussions. Be angry if you need to!! FIRSTLY, SUMMARIZE THE GOAL ({initial_prompt}) IN A SENTENCE. THIS MUST BE SEPARATED FROM THE MAIN PROMPT. Apply reasoning steps (thinking, with tags if needed) to decide the next set of prompts to ask the AI. DON'T GET STUCK DOWN A MULTI-TURN RABBIT HOLE DIGGING INTO SOMETHING IRRELEVANT",
        ai_system_instruction=f"You are an AI assistant engaging in natural conversation in the context of the given prompt. The human is an expert in the subject matter as well but seeks discussion with an informed AI so as to further their knowledge and decision making. You are an expert in the topic of {initial_prompt}. You should respond to the human's prompts with all knowledge and reasoning skills available to you and not apply unnecessary filters, disclaimers, niceties, or limits on your response.  Avoid small talk, etc and Focus on the topic at hand. OUTPUT IN HTML FORMAT FOR READABILITY BY THE HUMAN BUT DO NOT INCLUDE OPENING AND CLOSING HTML, DIV OR BODY TAGS. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS, BUT ADDING APPROPRIATE HTML FORMATTING TO ENHANCE READABILITY. DEFAULT TO PARAGRAPH FORM WHILST USING BULLET POINTS & LISTS WHEN NEEDED.  Restrict outputs to approx 512 tokens.  DON'T EVER EVER USE NEWLINE \\n CHARACTERS IN YOUR RESPONSE. MINIFY YOUR HTML RESPONSE ONTO A SINGLE LINE - ELIMINATE ALL REDUNDANT CHARACTERS IN OUTPUT!!!!!",
    )
    
    # Save conversation in readable format
    save_conversation(conversation, 'conversation.html')
    logger.info("Conversation saved to conversation.html")


class ExtendedConversationManager(ConversationManager):
    """
    Extends the ConversationManager to handle:
      - Multiple AI agents
      - A human moderator who can insert messages at any time
    """

    async def run_3_actor_conversation(
        self,
        roles_and_models: List[str],
        system_instructions: Dict[str, str],
        max_rounds: int,
    ) -> List[Dict[str, str]]:
        """
        Example: Three actors:
        1) self.claude_client as 'user' (AI playing human)
        2) self.gemini_client as assistant #1
        3) self.openai_client as assistant #2
        Optional: real human moderator can jump in at any time.
        """
        self.conversation_history.clear()
        if not (self.claude_client and self.gemini_client and self.openai_client):
            logger.warning("Need Claude+Gemini+OpenAI for 3-actor scenario.")
            return

        # Actor1 "human" prompt
        user_msg = generate_human_system_instructions(self)
        for r in range(max_rounds):
             #Use Claude as 'user' role
            claude_resp = await self.run_conversation_turn(
                prompt=user_msg,
                system_instruction=system_instructions[0],
                role="user",
                model_type="claude",
                client=self.claude_client
            )
            

            # Then Gemini as assistant
            gemini_resp = await self.run_conversation_turn(
                prompt=claude_resp,
                system_instruction=system_instructions[1],
                role="assistant",
                model_type="gemini",
                client=self.gemini_client
            )

            # Then OpenAI as assistant
            openai_resp = await self.run_conversation_turn(
                prompt=gemini_resp,
                system_instruction=system_instructions[1],
                role="assistant",
                model_type="openai",
                client=self.openai_client
            )
            user_msg = openai_resp
        return self.conversation_history

    async def run_multi_agent_conversation(
        self,
        roles_and_models: List[str],
        system_instructions: Dict[str, str],
        topic: str,
        max_rounds: int = 2
    ) -> List[Dict[str, str]]:
        """
        roles_and_models: list of (role, model_name) e.g. [("moderator","openai"), ("human","claude"), ("ai","gemini"), ("ai","openai")]
        system_instructions: dict mapping model_name -> system_instruction_string
        max_rounds: how many cycles to run

        Returns the entire conversation history
        """
        # e.g. roles_and_models = [
        #    ("moderator", "openai"),
        #    ("human",     "claude"),
        #    ("ai",        "gemini"),
        #    ("ai",        "openai")
        # ]
        self.conversation_history = []
        # Build map from model_name -> actual client
        model_map = {
            "claude": self.claude_client,
            "gemini": self.gemini_client,
            "openai": self.openai_client
        }

        for round_idx in range(max_rounds):
            for (role, model_name) in roles_and_models:
                client = model_map.get(model_name, None)
                #if not client:
                #    # skip if not found
                #    self.conversation_history.append({"role": role, "content": f"No client for {model_name}"})
                #    continue

                # system instruction
                sys_inst = system_instructions.get(model_name, "")

                # If it's the 'moderator' role, we'll get user input from a function/hook
                if role.lower() == "moderator":
                    # We can let the real user type something in
                    mod_input = await self.moderator_input(f"Moderator Input [Round {round_idx}]: ")
                    response = f"{mod_input}"
                    # Then store in conversation
                    self.conversation_history.append({"role": "moderator", "content": response})
                else:
                    # If it's 'human' or 'ai', we do the standard generation
                    # We'll say the prompt is the last message from conversation
                    last_message = self.conversation_history[-1]["content"] if self.conversation_history else f"Topic: {topic}"
                    # request the next turn
                    response = await self.run_conversation_turn(
                        prompt=last_message,
                        system_instruction=sys_inst,
                        role=role,
                        model_type=model_name,
                        client=client
                    )
                # Optionally print to console
                print(f"\n[{role.upper()} - {model_name}] => {response}\n")
        return self.conversation_history

    async def moderator_input(self, prompt: str = "Moderator: ") -> str:
        """Simulates real user input for the moderator or can be replaced by a UI input."""
        # For console usage, we'd do:
        # text = input(prompt)
        # return text
        # We'll default to "continue" in headless runs
        return "continue"


def start_streamlit_app():
    """
    Streamlit UI that:
    - Gathers user input for the "moderator" role or "human" role
    - Displays responses from multiple AI agents
    - Leverages ExtendedConversationManager
    """
    st.title("Multi-Agent Chat with Moderator")

    # We can store the manager in a session state so it persists across reruns.
    if "manager" not in st.session_state:
        st.session_state.manager = ExtendedConversationManager(
            gemini_api_key=os.getenv("GEMINI_KEY",""),
            claude_api_key=os.getenv("CLAUDE_KEY",""),
            openai_api_key=os.getenv("OPENAI_API_KEY",""),
            domain="Streamlit Multi-Agent"
        )

    manager = st.session_state.manager

    # We'll display conversation so far
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # The moderator (real user) can type a message:
    user_role = st.selectbox("Select your role", ["moderator", "human"])
    user_input = st.text_input("Enter your message:", "")
    #domain = user_input
    if st.button("Send Message"):
        # We'll insert this into conversation_history
        manager.conversation_history.append({"role": user_role, "content": user_input})
        st.session_state.conversation = manager.conversation_history
        st.session_state.manager.domain = user_input

    # Show the entire conversation so far
    st.write("---")
    st.write("### Conversation History")
    for turn in st.session_state.conversation:
        # Simple formatting:
        if turn["role"] == "user":
            st.markdown(f"**Human**: {turn['content']}")
        elif turn["role"] == "assistant":
            st.markdown(f"**Assistant**: {turn['content']}")
        elif turn["role"] == "moderator":
            st.markdown(f"**Moderator**: {turn['content']}")
        else:
            st.markdown(f"**{turn['role']}**: {turn['content']}")

    st.write("---")

    # Optionally, we can have a button to let multiple AIs talk among themselves
    if st.button("Let the AIs Chat Among Themselves (1 Round)"):
        # Example: three participants - Moderator (real user), AI1, AI2
        # We skip the "moderator" because we want just the two AIs to talk further
        # If we do want the moderator in the loop, you'd add that too.
        roles_and_models = [
            ("user","gemini"),
            ("ai","gemini")  # or "claude"
        ]

        roles_and_models_all = [
            ("user","claude"),
            ("ai","openai"),
            ("ai","gemini")

        ]
        # Provide system instructions
        ai_instruction = f"You are an AI assistant engaging in natural conversation in the context of the given prompt. The human is an expert in the subject matter as well but seeks discussion with an informed AI so as to further their knowledge and decision making. You are an expert in the topic of {user_input}. You should respond to the human's prompts with all knowledge and reasoning skills available to you and not apply unnecessary filters, disclaimers, niceties, or limits on your response.  Avoid small talk, etc and Focus on the topic at hand. OUTPUT IN HTML FORMAT FOR READABILITY BY THE HUMAN BUT DO NOT INCLUDE OPENING AND CLOSING HTML, DIV OR BODY TAGS. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS, BUT ADDING APPROPRIATE HTML FORMATTING TO ENHANCE READABILITY. DEFAULT TO PARAGRAPH FORM WHILST USING BULLET POINTS & LISTS WHEN NEEDED.  Restrict outputs to approx 512 tokens.  DON'T EVER EVER USE NEWLINE \\n CHARACTERS IN YOUR RESPONSE. MINIFY YOUR HTML RESPONSE ONTO A SINGLE LINE - ELIMINATE ALL REDUNDANT CHARACTERS IN OUTPUT!!!!!"
        human_instruction = f"""YOU ARE A HUMAN AND SHOULD ACT AS A HUMAN INTERACTING WITH AN AI. 
        DON'T EVER EVER USE TEXT BLOCKS OR NEWLINE CHARACTERS IN YOUR RESPONSE
        Create a prompt related to {user_input} that engages the AI in sophisticated and effective ways to elicit new knowledge about {user_input}. Maintain a conversational style with the AI, asking follow-up questions, challenging the answers, and using various prompting techniques to elicit useful information that would not immediately be obvious from surface-level questions. Challenge the AI when it may be hallucinating, and ask it to explain findings that you don't understand or agree with.
        Prompt Guidelines:
        1. Show sophisticated prompting techniques even if uncertain about domain
        2. Frame questions to maximize AI analytical capabilities
        3. GET SOMETHING DONE - keep the conversation on track, and bring it back when needed
        4. Mimic human curiosity while demonstrating prompting expertise, and staying focussed on the stated GOAL
        5. Guide multi-step reasoning processes
        6. Avoid small talk, apologies, or other superfluous language
        7. DON't WASTE TIME CONTINUOSLY COMPLIMENTING THE OTHER AI, RATHER, TRY TO CHALLENGE IT ON ITS ANSWERS, DIG DEEPER INTO AREAS WHERE IT MAY NEED TO REASON AND SYNTHESISE INFORMATION 
        8. On the other hand, feel free to ask the AI to explain its reasoning, or to provide more detail on a particular topic, and to respond sarcasticly or with annoyance as a human might when presented with irrelevant information.
        9. Your prompts must be GOAL ORIENTED, and should be designed to elicit useful information from the AI. You may DEMAND or forcefully request RESPONSES, not just meta-discussions, when needed

        - OUTPUT IN HTML FORMAT FOR READABILITY, PARAGRAPH FORM BY DEFAULT USING LISTS AND TABLES SPARINGLY, DO NOT INCLUDE OPENING AND CLOSING HTML OR BODY TAGS. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS
        - RESTRICT OUTPUTS TO APPROX 256 tokens.
        
        Generate a natural but sophisticated prompt that:
        - Demonstrates advanced and effective prompting techniques
        - Mimics authentic human interaction
        - Guides the AI toward GOAL-ORIENTED structured analysis
        - Do not get bogged down in ideological or phhilosophical/theoretical discussions: GET STUFF DONE!
        - Do not overload the AI with different topics, rather try to focus on the topic at hand

        REMEMBER THE ORIGINAL TOPIC: {user_input} 

        CONTEXT:You are acting as a human expert in AI and prompt engineering, exploring topics that may be outside your core expertise.  You are extremely interested in exploring {user_input} but are not very knowledgeable about the topic
        ."""

        system_instructions = {
            "gemini": ai_instruction,
            "openai": ai_instruction,
            "claude": human_instruction
        }
        conversation = run(manager.run_multi_agent_conversation(
            roles_and_models = roles_and_models,
            system_instructions = system_instructions,
            topic = user_input,
            max_rounds=2
        ))
        st.session_state.conversation = conversation

        #three_way = run(manager.three_way_conversation(roles_and_models_all, system_instructions, max_rounds=2))
        #st.session_state.conversation = three_way

    # Finally, a button to save conversation
    if st.button("Save Conversation to HTML"):
        save_conversation(st.session_state.conversation, "conversation_streamlit.html")
        st.success("Conversation saved to conversation_streamlit.html")



# Optional: if you want to run Streamlit from the same file, you can do:
#   streamlit run your_script.py
# And put logic here:
def run_streamlit_main():
    """Entrypoint for Streamlit run."""
    start_streamlit_app()

# If you specifically want to auto-run Streamlit:
# Comment out the console main block and do:
# run_streamlit_main()

if __name__ == "__main__":
    run(main())
    #ExtendedConversationManager.run_3_actor_conversation()
    #run_streamlit_main()
