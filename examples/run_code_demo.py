#!/usr/bin/env python3
"""
Example script for running a code generation discussion with automated demo creation.

This script demonstrates how to use the AI Battle framework to:
1. Orchestrate a conversation between a 'product manager' and an 'engineer' AI.
2. Have the 'engineer' AI write a Python script that generates a visual plot.
3. Automatically trigger the post-conversation 'DemoGenerator' to create a
   video summary of the development process, including executing the code in a
   secure sandbox.

Usage:
    python examples/run_code_demo.py
"""

import os
import sys
import asyncio
import logging
import importlib.util

# Add the project root to the Python path to allow imports from 'src'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0, project_root)
src_path = os.path.join(project_root, "src")
#sys.path.insert(0, src_path)

# Import ai-battle.py using importlib since it has a hyphen in the name
ai_battle_path = os.path.join(src_path, "ai_battle.py")
src_path = ai_battle_path
spec = importlib.util.spec_from_file_location("ai_battle", ai_battle_path)
ai_battle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ai_battle)



# Dynamically import the ConversationManager and other necessary components
# This pattern is used because 'ai-battle.py' has a hyphen.
try:
    ai_battle_spec = importlib.util.spec_from_file_location("ai_battle", os.path.join(src_path, "ai_battle.py"))
    ai_battle = importlib.util.module_from_spec(ai_battle_spec)
    ai_battle_spec.loader.exec_module(ai_battle)
    ConversationManager = ai_battle.ConversationManager

    demo_generator_spec = importlib.util.spec_from_file_location("demo_generator", os.path.join(src_path, "demo_generator.py"))
    demo_generator_module = importlib.util.module_from_spec(demo_generator_spec)
    demo_generator_spec.loader.exec_module(demo_generator_module)
    DemoGenerator = demo_generator_module.DemoGenerator

    sandbox_manager_spec = importlib.util.spec_from_file_location("sandbox_manager", os.path.join(src_path, "sandbox_manager.py"))
    sandbox_manager_module = importlib.util.module_from_spec(sandbox_manager_spec)
    sandbox_manager_spec.loader.exec_module(sandbox_manager_module)
    SandboxManager = sandbox_manager_module.SandboxManager

    model_clients_spec = importlib.util.spec_from_file_location("model_clients", os.path.join(src_path, "model_clients.py"))
    model_clients_module = importlib.util.module_from_spec(model_clients_spec)
    model_clients_spec.loader.exec_module(model_clients_module)
    GeminiClient = model_clients_module.GeminiClient

except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """
    Run a code generation discussion that results in an automated demo video.
    """
    config_path = "examples/configs/code_demo_discussion.yaml"
    logger.info(f"Starting code demo discussion with config: {config_path}")

    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return

    try:
        # Initialize conversation manager from the configuration file
        manager = ConversationManager.from_config(config_path)

        # Extract model info for logging
        models = list(manager.config.models.items())
        human_model_config = models[0][1]
        ai_model_config = models[1][1]
        
        # Run the conversation
        conversation_history = await manager.run_conversation(
            initial_prompt=manager.config.goal,
            human_model=human_model_config.type,
            ai_model=ai_model_config.type,
            mode="ai-ai",
            rounds=manager.config.turns,
        )

        logger.info("Conversation finished. Checking for demo generation config...")

        # After conversation, check if demo generation is enabled
        if manager.config and manager.config.demo_generation and manager.config.demo_generation.enabled:
            logger.info("Demo generation is enabled. Starting process...")
            try:
                # The DemoGenerator needs a GeminiClient for analysis and TTS.
                gemini_client = manager._get_client(manager.config.demo_generation.analysis_model)
                if not gemini_client or not isinstance(gemini_client, GeminiClient):
                    raise ValueError(f"Could not get a Gemini client for demo generation using model {manager.config.demo_generation.analysis_model}")

                # The DemoGenerator now creates its own SandboxManager
                demo_generator = DemoGenerator(
                    gemini_client=gemini_client, 
                    config=manager.config.demo_generation
                )
                
                video_path = demo_generator.generate(
                    conversation_history=conversation_history
                )
                
                if video_path:
                    logger.info(f"Demo video successfully generated at: {video_path}")
                else:
                    logger.error("Demo video generation failed.")
            except Exception as e:
                logger.error(f"An error occurred during demo generation: {e}", exc_info=True)

        logger.info("Script finished.")

    except Exception as e:
        logger.exception(f"An error occurred during the discussion: {e}")


if __name__ == "__main__":
    logger.info("Running code demo script.")
    logger.warning("This script requires a local Docker daemon and the 'ai-battle-sandbox' image if not using local execution.")
    logger.warning("The Docker build may be rate-limited in this environment, which will cause an error during code execution.")
    asyncio.run(main())
