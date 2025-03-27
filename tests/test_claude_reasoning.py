#!/usr/bin/env python3
"""
Test script for Claude 3.7 reasoning configuration.

This script demonstrates how to use the extended reasoning configuration
options for Claude 3.7 models, including token limits, formatting, and extended thinking.

Usage:
    python test_claude_reasoning.py [template_name] [budget_tokens]

Arguments:
    template_name: Reasoning template to use (step_by_step, concise, mathematical, direct, 
                  extended_thinking, deep_analysis)
    budget_tokens: Maximum tokens for extended thinking (default: varies by template)
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import necessary modules
from model_clients import ClaudeClient, ModelConfig
from claude_reasoning_config import get_reasoning_config, REASONING_TEMPLATES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_api_key():
    """Get Claude API key from environment or file."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            with open(os.path.expanduser("~/.ANTHROPIC_API_KEY"), "r") as f:
                api_key = f.read().strip()
        except:
            logger.error("Could not find ANTHROPIC_API_KEY")
            return None
    return api_key

def main():
    """Test Claude 3.7 reasoning configuration."""
    # Get API key
    api_key = get_api_key()
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment or ~/.ANTHROPIC_API_KEY file")
        return
    
    # Get template name from command line or use default
    template_name = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in REASONING_TEMPLATES else "step_by_step"
    
    # Get budget tokens from command line or use default
    budget_tokens = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
    
    # Create reasoning configuration
    reasoning_config = get_reasoning_config(
        template_name=template_name,
        budget_tokens=budget_tokens if budget_tokens is not None else None
    )
    
    # Display configuration
    print(f"\n🔧 Using reasoning configuration:")
    print(f"  Template: {template_name}")
    print(f"  Level: {reasoning_config.level}")
    print(f"  Max reasoning tokens: {reasoning_config.max_reasoning_tokens or 'Unlimited'}")
    print(f"  Show working: {reasoning_config.show_working}")
    print(f"  Extended thinking: {reasoning_config.extended_thinking}")
    if reasoning_config.extended_thinking:
        print(f"  Budget tokens: {reasoning_config.budget_tokens or 'Default'}")
    
    # Create model configuration
    model_config = ModelConfig(
        temperature=0.7,
        max_tokens=1024,  # Max response tokens
        seed=None  # Avoid seed parameter issue
    )
    
    # Create client
    model = "claude-3-7-sonnet-20250219"  # Use the exact model ID
    logger.info(f"Creating Claude client with model {model}")
    
    client = ClaudeClient(
        role="user",
        api_key=api_key,
        mode="ai-ai",
        domain="testing",
        model=model
    )
    
    # Set reasoning parameters
    client.reasoning_level = reasoning_config.level
    logger.info(f"Set reasoning level to {client.reasoning_level}")
    
    # Handle extended thinking (would be implemented in the actual client)
    if reasoning_config.extended_thinking:
        logger.info(f"Enabling extended thinking with budget_tokens={reasoning_config.budget_tokens}")
        # In a full implementation, we would update the client to handle these parameters
    
    # Create test prompt
    test_prompt = "Solve this probability problem: If you roll two fair six-sided dice, what is the probability of getting a sum greater than 9?"
    
    # Add reasoning instructions to system prompt
    system_instruction = f"""You are a mathematics expert who explains concepts clearly and accurately.

{reasoning_config.to_system_instruction()}"""
    
    print(f"\n📝 Prompt: {test_prompt}")
    print(f"\n📋 System instructions:\n{system_instruction}")
    print("\n⏳ Generating response...")
    
    # Generate response
    try:
        response = client.generate_response(
            prompt=test_prompt,
            system_instruction=system_instruction,
            model_config=model_config
        )
        
        # Display response
        print("\n✅ Claude response:")
        print("="*80)
        print(response)
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()