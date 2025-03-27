#!/usr/bin/env python3
"""
Example script to demonstrate Claude 3.7's extended thinking capability.

This script compares responses from Claude 3.7 with and without extended thinking
for complex analytical problems.

Usage:
    python test_extended_thinking.py [problem_type] [budget_tokens]

Arguments:
    problem_type: Type of problem to solve (math, science, analysis, compare)
    budget_tokens: Maximum tokens for extended thinking (default: 8000)
"""

import os
import sys
import time
import asyncio
import logging
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import modules
import importlib.util
spec = importlib.util.spec_from_file_location("ai_battle", os.path.join(parent_dir, "ai-battle.py"))
ai_battle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ai_battle)

from model_clients import ClaudeClient, ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Problem types with corresponding prompts
PROBLEMS = {
    "math": "Analyze the Riemann Hypothesis and its implications for prime number distribution. Evaluate the current state of research, key challenges, and most promising approaches toward a proof. Provide a coherent explanation of why this problem is significant in number theory.",
    
    "science": "Analyze the implications of quantum entanglement for our understanding of locality and causality in physics. Discuss the major interpretations of quantum mechanics that attempt to explain this phenomenon, their strengths and weaknesses, and which interpretation you find most compelling based on current experimental evidence.",
    
    "analysis": "Conduct a detailed analysis of the economic impact of climate change over the next 50 years. Consider multiple scenarios based on different levels of global temperature rise, focusing on agricultural productivity, infrastructure costs, health impacts, and energy transitions. Reference relevant economic models and their limitations.",
    
    "compare": "Compare and contrast three different economic systems: free market capitalism, state capitalism, and democratic socialism. Analyze how each system handles resource allocation, wealth distribution, innovation incentives, and social welfare. Provide examples of countries that implement elements of each system and evaluate their real-world outcomes."
}

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

async def main():
    """Test Claude 3.7 extended thinking with various problems."""
    # Get API key
    api_key = get_api_key()
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment or ~/.ANTHROPIC_API_KEY file")
        return
    
    # Get problem type and budget tokens
    problem_type = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in PROBLEMS else "math"
    budget_tokens = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 8000
    
    # Get problem prompt
    prompt = PROBLEMS[problem_type]
    
    print(f"\n🧠 Testing Claude 3.7 Extended Thinking")
    print(f"Problem type: {problem_type}")
    print(f"Budget tokens: {budget_tokens}")
    print(f"\nPrompt: {prompt}\n")
    
    # Create model configuration
    model_config = ModelConfig(
        temperature=0.7,
        max_tokens=1500,  # Response tokens
        seed=None  # Avoid seed parameter issue
    )
    
    # System instruction
    system_instruction = "You are an expert in providing detailed, analytical responses to complex questions. Focus on providing logical, structured answers with evidence and reasoning."
    
    # Try without extended thinking first
    print("🔹 Testing without extended thinking...")
    start_time = time.time()
    
    client1 = ClaudeClient(
        role="user",
        api_key=api_key,
        mode="ai-ai",
        domain="analysis",
        model="claude-3-7-sonnet-20250219"  # Use the exact model ID
    )
    
    # Set high reasoning level but no extended thinking
    client1.reasoning_level = "high"
    
    # Generate response without extended thinking
    response_basic = client1.generate_response(
        prompt=prompt,
        system_instruction=system_instruction,
        model_config=model_config
    )
    
    basic_time = time.time() - start_time
    print(f"✅ Basic response generated in {basic_time:.2f} seconds")
    
    # Now try with extended thinking
    print("\n🔹 Testing with extended thinking...")
    start_time = time.time()
    
    client2 = ClaudeClient(
        role="user",
        api_key=api_key,
        mode="ai-ai",
        domain="analysis",
        model="claude-3-7-sonnet-20250219"  # Use the exact model ID
    )
    
    # Set high reasoning level and enable extended thinking
    client2.reasoning_level = "high"
    client2.set_extended_thinking(True, budget_tokens)
    
    # Generate response with extended thinking
    # First check if the client supports extended thinking
    extended_thinking_supported = False
    try:
        # Try to check anthropic client version
        import anthropic
        import pkg_resources
        anthropic_version = pkg_resources.get_distribution("anthropic").version
        print(f"  Anthropic client version: {anthropic_version}")
        
        # Version 0.19.0+ supports extended thinking
        if anthropic_version >= "0.19.0":
            extended_thinking_supported = True
    except Exception:
        # If we can't check the version, assume it might work
        extended_thinking_supported = True
    
    if not extended_thinking_supported:
        print(f"⚠️ Extended thinking not supported by current Anthropic client")
        print(f"⚠️ Please upgrade to anthropic>=0.19.0")
        print("Falling back to basic response...")
        response_extended = response_basic
    else:
        try:
            response_extended = client2.generate_response(
                prompt=prompt,
                system_instruction=system_instruction,
                model_config=model_config
            )
        except Exception as e:
            logger.error(f"Error generating response with extended thinking: {e}")
            print(f"⚠️ Error with extended thinking: {e}")
            print("Falling back to basic response...")
            # Fall back to the basic response if extended thinking fails
            response_extended = response_basic
    
    extended_time = time.time() - start_time
    print(f"✅ Extended thinking response generated in {extended_time:.2f} seconds")
    
    # Save responses to files
    output_dir = Path("extended_thinking_results")
    output_dir.mkdir(exist_ok=True)
    
    # Convert responses to string if needed
    response_basic_str = response_basic if isinstance(response_basic, str) else str(response_basic)
    response_extended_str = response_extended if isinstance(response_extended, str) else str(response_extended)
    
    basic_file = output_dir / f"{problem_type}_basic.txt"
    with open(basic_file, "w") as f:
        f.write(f"PROMPT: {prompt}\n\n")
        f.write(f"WITHOUT EXTENDED THINKING (Time: {basic_time:.2f}s):\n\n")
        f.write(response_basic_str)
    
    extended_file = output_dir / f"{problem_type}_extended.txt"
    with open(extended_file, "w") as f:
        f.write(f"PROMPT: {prompt}\n\n")
        f.write(f"WITH EXTENDED THINKING (Budget: {budget_tokens}, Time: {extended_time:.2f}s):\n\n")
        f.write(response_extended_str)
    
    # Print summary
    print(f"\n📊 Results Summary:")
    print(f"  - Basic response: {len(response_basic_str)} characters, {basic_time:.2f} seconds")
    print(f"  - Extended thinking: {len(response_extended_str)} characters, {extended_time:.2f} seconds")
    
    # Check if we actually got a different response with extended thinking
    if response_extended is response_basic:
        print(f"  ⚠️ NOTE: Extended thinking not supported by current API client")
        print(f"  ⚠️ The API client library doesn't support the 'thinking' parameter")
        print(f"  ⚠️ Both responses are identical (using fallback)")
    else:
        print(f"  - Time difference: {extended_time - basic_time:.2f} seconds")
        
    print(f"  - Output files: {basic_file} and {extended_file}")
    
    # Print a sample of both responses
    print("\n🔍 Response Comparison (first 300 characters):")
    print("\nBASIC RESPONSE:")
    print(response_basic_str[:300] + "..." if len(response_basic_str) > 300 else response_basic_str)
    print("\nEXTENDED THINKING RESPONSE:")
    print(response_extended_str[:300] + "..." if len(response_extended_str) > 300 else response_extended_str)

if __name__ == "__main__":
    asyncio.run(main())