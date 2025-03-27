#!/usr/bin/env python3
"""
Example script to use local LMStudio models with the Claude-Gemini-Chat framework.

This script demonstrates how to use the LMStudioClient to interact with locally
running LLMs through LMStudio's OpenAI-compatible API endpoint.

Usage:
    python run_lmstudio.py [prompt] [base_url]

Arguments:
    prompt: The prompt to send to the model (default: a general question)
    base_url: The base URL of the LMStudio server (default: http://localhost:1234/v1)
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import the LMStudio client
from lmstudio_client import LMStudioClient
from model_clients import ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run a simple conversation with a local LMStudio model."""
    # Get prompt from command line or use default
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain how large language models work in one paragraph."
    
    # Get base URL from command line or use default
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:1234/v1"
    
    try:
        # Initialize the LMStudio client
        logger.info(f"Connecting to LMStudio at {base_url}")
        client = LMStudioClient(
            role="assistant",
            mode="human-ai", 
            domain="General Knowledge",
            base_url=base_url
        )
        
        # Test the connection
        if not client.validate_connection():
            logger.error("Failed to connect to LMStudio server")
            print(f"\n❌ Error: Could not connect to LMStudio at {base_url}")
            print("Make sure LMStudio is running and a model is loaded.")
            return
        
        # Log available models
        logger.info(f"Available models: {client.available_models}")
        print(f"\n🔍 Found {len(client.available_models)} models on LMStudio server")
        for model in client.available_models:
            print(f"  - {model}")
        
        # If models are available, use the first one
        if client.available_models:
            client.model = client.available_models[0]
            logger.info(f"Using model: {client.model}")
            print(f"\n🤖 Using model: {client.model}")
        
        # Generate response
        print(f"\n📝 Prompt: {prompt}")
        print("\n⏳ Generating response...")
        
        model_config = ModelConfig(temperature=0.7, max_tokens=1024)
        model_config.seed = None  # Explicitly set seed to None
        
        response = client.generate_response(
            prompt=prompt,
            system_instruction="You are a helpful AI assistant with expertise in various fields.",
            model_config=model_config
        )
        
        # Display response
        print("\n✅ Response from local model:")
        print("="*80)
        print(response)
        print("="*80)
        
    except Exception as e:
        logger.exception(f"Error running LMStudio client: {e}")
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())