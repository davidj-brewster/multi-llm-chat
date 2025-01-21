import logging
from google import genai
from google.genai import types
from anthropic import Anthropic
from typing import List, Dict, Optional, Union, TypeVar, Any
from dataclasses import dataclass
from asyncio import Lock, sleep, run
import json
import time
from pathlib import Path
import sys

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
    temperature: float = 0.3
    max_tokens: int = 1000
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: List[str] = None
    seed: Optional[int] = None

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

class GeminiClient(BaseClient):
    """Client for Gemini API interactions"""
    def __init__(self, api_key: str):
        """Initialize Gemini client
        
        Args:
            api_key: Gemini API key
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 20,
        }

    async def test_connection(self) -> None:
        """Test Gemini API connection"""
        response = await self.model.generate_content("test")
        if not response:
            raise Exception("Failed to connect to Gemini API")

    async def generate_response(self,
                              prompt: str,
                              system_instruction: str,
                              history: List[Dict[str, str]],
                              model_config: Optional[ModelConfig] = None) -> str:
        """Generate response using Gemini API
        
        Args:
            prompt: Current prompt
            system_instruction: System context
            history: Conversation history
            model_config: Model parameters
            
        Returns:
            str: Generated response
        """
        if model_config:
            self.generation_config.update({
                "temperature": model_config.temperature,
                "top_p": model_config.top_p,
                "top_k": model_config.top_k,
            })

        try:
            chat = self.model.start_chat(history=history)
            response = await chat.send_message(
                prompt,
                generation_config=self.generation_config
            )
            logger.info("Gemini response generated successfully")
            return response.text
        except Exception as e:
            logger.error(f"Error generating Gemini response: {str(e)}")
            return f"Error generating Gemini response: {str(e)}"

class ClaudeClient(BaseClient):
    """Client for Claude API interactions"""
    def __init__(self, api_key: str):
        """Initialize Claude client
        
        Args:
            api_key: Claude API key
        """
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-sonnet-20240229"

    async def test_connection(self) -> None:
        """Test Claude API connection"""
        try:
            await self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10
            )
        except Exception as e:
            raise Exception(f"Failed to connect to Claude API: {str(e)}")

    async def generate_response(self,
                              prompt: str,
                              system_instruction: str,
                              history: List[Dict[str, str]],
                              model_config: Optional[ModelConfig] = None) -> str:
        """Generate response using Claude API
        
        Args:
            prompt: Current prompt
            system_instruction: System context
            history: Conversation history
            model_config: Model parameters
            
        Returns:
            str: Generated response
        """
        if model_config is None:
            model_config = ModelConfig()

        messages = []
        if system_instruction:
            messages.append({
                "role": "system",
                "content": system_instruction
            })
        
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })

        try:
            response = await self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
                system=system_instruction if system_instruction else None,
            )
            logger.info("Claude response generated successfully")
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating Claude response: {str(e)}")
            return f"Error generating Claude response: {str(e)}"

class ConversationManager:
    """Manages conversation between AI models"""
    def __init__(self, 
                 gemini_api_key: str,
                 claude_api_key: str):
        """Initialize conversation manager
        
        Args:
            gemini_api_key: Gemini API key
            claude_api_key: Claude API key
        """
        self.gemini = GeminiClient(gemini_api_key)
        self.claude = ClaudeClient(claude_api_key)
        self.human_config = ModelConfig(temperature=0.9, max_tokens=1200)
        self.ai_config = ModelConfig(temperature=0.4, max_tokens=1200)
        self.conversation_history = []
        self.is_paused = False
        self.last_human_response_time = 0
        self.human_delay = 10.0  # 5 second delay for human responses
        self.rate_limit_lock = Lock()
        self.last_request_time = 0
        self.min_delay = 2.5  # seconds


    async def rate_limited_request(self):
        async with self.rate_limit_lock:
            current_time = time.time()
            if current_time - self.last_request_time < self.min_delay:
                await sleep(self.min_delay)
            self.last_request_time = time.time()


    async def validate_connections(self) -> bool:
        """Validate all API connections
        
        Returns:
            bool: True if all connections are valid
        """
        return all([
            await self.gemini.validate_connection(),
            await self.claude.validate_connection()
        ])

    async def run_conversation_turn(self,
                                  prompt: str,
                                  system_instruction: str,
                                  role: str,
                                  model_type: str) -> str:
        """Run single conversation turn
        
        Args:
            prompt: Current prompt
            system_instruction: System context
            role: Role (human/assistant)
            model_type: Model to use
            
        Returns:
            str: Generated response
        """
        if role == "human":
            # Enforce delay for human responses
            current_time = time.time()
            if current_time - self.last_human_response_time < self.human_delay:
                await asyncio.sleep(self.human_delay)
            self.last_human_response_time = time.time()

        config = self.human_config if role == "human" else self.ai_config
        
        if model_type == "claude":
            return await self.claude.generate_response(
                prompt=prompt,
                system_instruction=system_instruction,
                history=self.conversation_history,
                model_config=config
            )
        else:
            return await self.gemini.generate_response(
                prompt=prompt,
                system_instruction=system_instruction,
                history=self.conversation_history,
                model_config=config
            )

    async def run_conversation(self,
                             initial_prompt: str,
                             human_system_instruction: str,
                             ai_system_instruction: str,
                             rounds: int = 3) -> List[Dict[str, str]]:
        """Run full conversation between models
        
        Args:
            initial_prompt: Starting topic/prompt
            human_system_instruction: Context for human role
            ai_system_instruction: Context for AI role
            rounds: Number of conversation rounds
            
        Returns:
            List[Dict[str, str]]: Conversation history
        """
        logger.info(f"Starting conversation with prompt: {initial_prompt}")
        print("\n=== Starting AI Conversation ===")
        print(f"Topic: {initial_prompt}\n")

        for round in range(rounds):
            if self.is_paused:
                choice = await self.human_intervention("Resume conversation?")
                if choice == "stop":
                    break
                elif choice == "continue":
                    self.is_paused = False
                else:
                    continue

            # Human turn
            human_response = await self.run_conversation_turn(
                prompt=initial_prompt if round == 0 else "Continue the conversation naturally",
                system_instruction=human_system_instruction,
                role="human",
                model_type="claude"
            )
            
            self.conversation_history.append({
                "role": "human",
                "content": human_response
            })
            
            print(f"\nHUMAN: {human_response}\n")
            choice = await self.human_intervention(f"Generated human response shown above")
            if choice != "continue":
                continue

            # AI turn
            ai_response = await self.run_conversation_turn(
                prompt=human_response,
                system_instruction=ai_system_instruction,
                role="assistant",
                model_type="gemini"
            )
            
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_response
            })
            
            print(f"\nAI: {ai_response}\n")
            choice = await self.human_intervention(f"Generated AI response shown above")
            if choice == "stop":
                break

        return self.conversation_history

async def main():
    """Main entry point
    
    Retrieves API keys from environment variables GEMINI_KEY and CLAUDE_KEY if present,
    otherwise prompts user for input.
    """
    import os
    
    # Try to get API keys from environment first
    gemini_key = os.getenv('GEMINI_KEY')
    claude_key = os.getenv('CLAUDE_KEY')
    
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
        claude_api_key=claude_key
    )
    
    # Validate connections
    if not await manager.validate_connections():
        logger.error("Failed to validate API connections")
        return
    
    # Get initial prompt from user
    initial_prompt = input("\nEnter conversation topic/prompt: ")
    
    conversation = await manager.run_conversation(
        initial_prompt=initial_prompt,
        human_system_instruction="You are a human expert in AI and medical diagnosis...",
        ai_system_instruction="You are an AI assistant engaging in natural conversation..."
    )
    
    # Save final conversation
    with open('conversation_history.json', 'w') as f:
        json.dump(conversation, f, indent=2)
    logger.info("Conversation saved to conversation_history.json")

if __name__ == "__main__":
    run(main())
