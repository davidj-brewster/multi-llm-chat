#!/usr/bin/env python3
"""
Example script to test reasoning capabilities in Claude 3.7 and OpenAI O1/O3 models.

This script demonstrates how different reasoning levels affect model responses,
allowing users to compare the level of detail and explanation provided at each level.

Usage:
    python reasoning_test.py [model_type] [reasoning_level] [prompt]

Arguments:
    model_type: Type of model to use (claude or openai)
    reasoning_level: Reasoning level to use (high, medium, low, none, auto)
    prompt: The prompt to send to the model (default: explain quantum entanglement)
"""

import os
import sys
import asyncio
import logging
import importlib.util
from pathlib import Path

# Add parent directory to path to import from ai-battle.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import ai-battle.py using importlib since it has a hyphen in the name
ai_battle_path = os.path.join(parent_dir, "ai-battle.py")
spec = importlib.util.spec_from_file_location("ai_battle", ai_battle_path)
ai_battle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ai_battle)

# Import model clients
from model_clients import ClaudeClient, OpenAIClient, ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_api_key(provider):
    """Get API key from environment or key file."""
    if provider == "claude":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            try:
                with open(os.path.expanduser("~/.ANTHROPIC_API_KEY"), "r") as f:
                    key = f.read().strip()
            except:
                logger.error("Could not find ANTHROPIC_API_KEY")
                return None
        return key
    elif provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            try:
                with open(os.path.expanduser("~/.OPENAI_API_KEY"), "r") as f:
                    key = f.read().strip()
            except:
                logger.error("Could not find OPENAI_API_KEY")
                return None
        return key
    return None

def test_claude_reasoning(reasoning_level, prompt):
    """Test Claude model with specified reasoning level."""
    api_key = get_api_key("claude")
    if not api_key:
        print("Error: Could not find Claude API key")
        return
    
    # Map reasoning levels to model variants
    model = "claude-3-7-sonnet-20250219"  # Use the exact model ID from the API
    
    # Create client
    logger.info(f"Testing Claude with reasoning_level={reasoning_level}")
    client = ClaudeClient(
        role="user",
        api_key=api_key,
        mode="ai-ai",
        domain="testing",
        model=model
    )
    
    # Set reasoning level explicitly
    client.reasoning_level = reasoning_level
    logger.info(f"Using model {model} with reasoning_level={reasoning_level}")
    
    # Generate response
    try:
        # Use a model_config without a seed to avoid parameter issues
        model_config = ModelConfig(temperature=0.7, max_tokens=1024)
        model_config.seed = None  # Explicitly set seed to None
        
        response = client.generate_response(
            prompt=prompt,
            system_instruction=f"You are an expert providing information on the topic. Show your reasoning at {reasoning_level} level to explain the concept thoroughly.",
            model_config=model_config
        )
        
        print("\n" + "="*80)
        print(f"CLAUDE RESPONSE WITH REASONING LEVEL: {reasoning_level.upper()}")
        print("="*80)
        print(response)
        print("="*80 + "\n")
        
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        print(f"Error: {e}")
        return None

def test_openai_reasoning(reasoning_level, prompt):
    """Test OpenAI model with specified reasoning level."""
    api_key = get_api_key("openai")
    if not api_key:
        print("Error: Could not find OpenAI API key")
        return
    
    # Map reasoning levels to model variants
    model = "o1"  # Base model
    
    # Create client
    logger.info(f"Testing OpenAI with reasoning_level={reasoning_level}")
    client = OpenAIClient(
        role="user",
        api_key=api_key,
        mode="ai-ai",
        domain="testing",
        model=model
    )
    
    # Set reasoning level explicitly
    client.reasoning_level = reasoning_level
    logger.info(f"Using model {model} with reasoning_level={reasoning_level}")
    
    # Generate response
    try:
        # Use a model_config without a seed to avoid parameter issues
        model_config = ModelConfig(temperature=0.7, max_tokens=1024)
        model_config.seed = None  # Explicitly set seed to None
        
        response = client.generate_response(
            prompt=prompt,
            system_instruction=f"You are an expert providing information on the topic. Show your reasoning at {reasoning_level} level to explain the concept thoroughly.",
            history=None,
            file_data=None,
            model_config=model_config
        )
        
        print("\n" + "="*80)
        print(f"OPENAI RESPONSE WITH REASONING LEVEL: {reasoning_level.upper()}")
        print("="*80)
        print(response)
        print("="*80 + "\n")
        
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        print(f"Error: {e}")
        return None

async def main():
    """Run reasoning test based on command-line arguments."""
    # Get arguments from command line
    model_type = sys.argv[1] if len(sys.argv) > 1 else "claude"
    reasoning_level = sys.argv[2] if len(sys.argv) > 2 else "medium"
    prompt = sys.argv[3] if len(sys.argv) > 3 else "Explain quantum entanglement, including its implications for physics and computing."
    
    # Validate model type
    if model_type.lower() not in ["claude", "openai"]:
        logger.error(f"Invalid model type: {model_type}. Must be 'claude' or 'openai'")
        print("Error: Invalid model type. Must be 'claude' or 'openai'")
        return
    
    # Validate reasoning level
    valid_levels = ["high", "medium", "low", "none", "auto"]
    if reasoning_level.lower() not in valid_levels:
        logger.error(f"Invalid reasoning level: {reasoning_level}. Must be one of {valid_levels}")
        print(f"Error: Invalid reasoning level. Must be one of {valid_levels}")
        return
    
    print(f"\nTesting {model_type.upper()} model with reasoning level: {reasoning_level}")
    print(f"Prompt: {prompt}\n")
    
    # Run appropriate test
    if model_type.lower() == "claude":
        test_claude_reasoning(reasoning_level, prompt)
    else:
        test_openai_reasoning(reasoning_level, prompt)
    
    print("\nTest completed.")

if __name__ == "__main__":
    asyncio.run(main())