#!/usr/bin/env python3
"""
Test script to run a vision discussion with Claude 3.7 + reasoning.

This script runs a vision discussion using an MRI video file to test
the Claude 3.7 reasoning functionality in a real-world scenario.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vision_test")

# Add the parent directory to the module search path
sys.path.append(str(Path(__file__).parent))

# Make sure the ANTHROPIC_API_KEY is set
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    try:
        with open(os.path.expanduser("~/.ANTHROPIC_API_KEY"), "r") as f:
            api_key = f.read().strip()
            os.environ["ANTHROPIC_API_KEY"] = api_key
            logger.info("Loaded API key from ~/.ANTHROPIC_API_KEY")
    except:
        logger.error(
            "Could not find ANTHROPIC_API_KEY in environment or ~/.ANTHROPIC_API_KEY file"
        )
        sys.exit(1)


async def main():
    """Run the vision discussion test."""
    logger.info("Starting vision discussion test with Claude 3.7 reasoning")

    # Import the run_vision_discussion module
    vision_script_path = Path(__file__).parent / "examples/run_vision_discussion.py"
    spec = spec_from_file_location("run_vision_discussion", vision_script_path)
    run_vision_discussion = module_from_spec(spec)
    spec.loader.exec_module(run_vision_discussion)

    # Path to the config file
    config_path = (
        Path(__file__).parent / "examples/configs/claude_reasoning_vision.yaml"
    )

    # Path to the MRI video file
    video_path = Path(__file__).parent / "T2-SAG-FLAIR.mov"

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    logger.info(f"Using config file: {config_path}")
    logger.info(f"Using video file: {video_path}")

    # Set the sys.argv for the run_vision_discussion script
    saved_argv = sys.argv
    sys.argv = [str(vision_script_path), str(config_path), str(video_path)]

    try:
        # Run the vision discussion
        logger.info("Running vision discussion...")
        await run_vision_discussion.main()
        logger.info("Vision discussion completed successfully")
    except Exception as e:
        logger.error(f"Error running vision discussion: {e}")
        raise
    finally:
        # Restore sys.argv
        sys.argv = saved_argv

    logger.info("Vision discussion test completed")


if __name__ == "__main__":
    asyncio.run(main())
