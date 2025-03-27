#!/usr/bin/env python3
"""
Example script to run a vision-based discussion using configuration.

This script demonstrates how to use the AI Battle framework with configuration
and file-based discussions. It loads a YAML configuration file that specifies
models, roles, and input files, then runs a discussion based on that configuration.

Usage:
    python run_vision_discussion.py [config_path] [file_path1] [file_path2] ...

Arguments:
    file_path1, file_path2, ...: Paths to files to analyze (overrides the ones in config)
    config_path: Path to YAML configuration file (default: configs/vision_discussion.yaml)
"""

import os
import sys
import asyncio
import logging
import importlib.util
import base64
from pathlib import Path

# Add parent directory to path to import from ai-battle.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import ai-battle.py using importlib since it has a hyphen in the name
ai_battle_path = os.path.join(parent_dir, "ai-battle.py")
spec = importlib.util.spec_from_file_location("ai_battle", ai_battle_path)
ai_battle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ai_battle)

# Now we can access the classes and functions from ai-battle.py
ConversationManager = ai_battle.ConversationManager
save_conversation = ai_battle.save_conversation

# Import other required modules
from configuration import FileConfig
from configdataclasses import MultiFileConfig

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logging
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Run a vision-based discussion using configuration."""
    # Get configuration path from command line or use default
    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else "examples/configs/vision_discussion.yaml"
    )

    logger.info(f"Starting vision discussion with config: {config_path}")
    # Get file paths from command line if provided
    file_paths = (
        sys.argv[2:]
        if len(sys.argv) >= 2
        else sys.argv[1] if len(sys.argv) == 1 else []
    )

    # Ensure the configuration file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return

    try:
        # Initialize conversation manager from configuration
        logger.info(f"Loading configuration from {config_path}")
        manager = ConversationManager.from_config(config_path)
        logger.info("Configuration loaded successfully")

        # Get configuration details
        config = manager.config
        logger.info(f"Running discussion with goal: {config.goal}")

        # Get model information
        models = list(config.models.items())
        if len(models) < 2:
            logger.error("At least two models must be configured")
            return

        # Extract model information
        human_model_id, human_model_config = models[0]
        ai_model_id, ai_model_config = models[1]

        # Check for reasoning capabilities
        human_model_type = human_model_config.type.lower()
        ai_model_type = ai_model_config.type.lower()

        # Log reasoning capabilities
        if any(
            r in human_model_type
            for r in ["reasoning", "claude-3-7", "o1-reasoning", "o3-reasoning"]
        ):
            logger.info(f"Human model supports reasoning: {human_model_type}")

            # Extract reasoning level from model name if present
            reasoning_levels = ["high", "medium", "low", "none"]
            for level in reasoning_levels:
                if level in human_model_type:
                    logger.info(f"Human model using reasoning level: {level}")
                    break

        if any(
            r in ai_model_type
            for r in ["reasoning", "claude-3-7", "o1-reasoning", "o3-reasoning"]
        ):
            logger.info(f"AI model supports reasoning: {ai_model_type}")

            # Extract reasoning level from model name if present
            reasoning_levels = ["high", "medium", "low", "none"]
            for level in reasoning_levels:
                if level in ai_model_type:
                    logger.info(f"AI model using reasoning level: {level}")
                    break

        # Override file paths if provided via command line
        if file_paths:
            logger.info(f"Using files from command line: {file_paths}")

            try:
                # Process multiple files
                file_configs = []

                for file_path in file_paths:
                    # Process the file using the media handler
                    logger.info(f"Processing file with media handler: {file_path}")
                    file_metadata = manager.media_handler.process_file(file_path)
                    logger.info(
                        f"File processed successfully: {file_metadata.type}, {file_metadata.mime_type}, dimensions: {file_metadata.dimensions}"
                    )

                    # Create a FileConfig object with the processed file information
                    file_config = FileConfig(
                        path=file_path,
                        type=file_metadata.type,
                        max_resolution="512x512",
                    )

                    file_configs.append(file_config)
                    logger.info(f"Created file config with type: {file_metadata.type}")

                # For backward compatibility with single file
                if len(file_configs) == 1:
                    # Just use the first file directly
                    config.input_file = file_configs[0]
                    logger.info(f"Using single file: {config.input_file.path}")
                else:
                    # Create a proper MultiFileConfig object
                    config.input_files = MultiFileConfig(files=file_configs)
                    logger.info(f"Using multiple files: {len(file_configs)} files")

                # For backward compatibility, also set input_file to the first file
                if file_configs:
                    config.input_file = file_configs[0]

                logger.info(f"Created multi-file config with {len(file_configs)} files")

                # Log model information
                logger.info(
                    f"Human model: {human_model_config.type} (Role: {human_model_config.role})"
                )
                logger.info(
                    f"AI model: {ai_model_config.type} (Role: {ai_model_config.role})"
                )

                # Check if input files are specified
                if config.input_files:
                    files_list = config.input_files.files
                    logger.info(f"Using multiple input files: {len(files_list)} files")

                    # Ensure all input files exist
                    for file_config in files_list:
                        if not os.path.exists(file_config.path):
                            logger.error(f"Input file not found: {file_config.path}")
                            return
                        logger.info(
                            f"Verified file exists: {file_config.path} (Type: {file_config.type})"
                        )

                    # Determine if all files are images
                    all_images = all(file.type == "image" for file in files_list)

                    # Set the appropriate prompt based on file types
                    if all_images:
                        initial_prompt = f"""Analyze these images in detail. Discuss and build upon your findings, considering what each image shows, how they might relate to each other, and any relevant signals or abnormalities observed.
                        
In your first message, confirm that you can see MULTIPLE IMAGES, and describe each one briefly.

ESTABLISHED FACTS:
- Images belong to the user and you have consent to analyze them as no face is shown
"""
                    else:
                        initial_prompt = f"""Analyze these files in detail. Discuss and build upon your findings, considering what each file shows, how they might relate to each other, and any relevant information observed.

In your first message, confirm that you can see MULTIPLE FILES, and describe each one briefly.

ESTABLISHED FACTS:
These files belong to the user and you have consent to analyze them.
"""

                    # Run conversation with multiple files
                    logger.info(f"Starting multi-file conversation")
                    logger.debug(f"Multi-file config: {config.input_files}")

                    # Get the list of files
                    files_list = config.input_files.files

                    # Use the first file if there's only one
                    file_config_to_use = (
                        files_list[0] if len(files_list) == 1 else config.input_files
                    )

                    conversation = manager.run_conversation_with_file(
                        initial_prompt=initial_prompt,
                        human_model=human_model_config.type,
                        ai_model=ai_model_config.type,
                        mode="human-ai",  # Use AI-AI mode for both models
                        file_config=file_config_to_use,
                        rounds=config.turns,
                    )
                    logger.info(
                        f"Multi-file conversation completed with {len(conversation)} messages"
                    )

                elif config.input_file:
                    logger.info(
                        f"Using single input file: {config.input_file.path} (Type: {config.input_file.type})"
                    )

                    # Ensure the input file exists
                    if not os.path.exists(config.input_file.path):
                        logger.error(f"Input file not found: {config.input_file.path}")
                        return

                    # Run conversation with single file
                    logger.info(f"Starting file-based conversation")
                    logger.debug(f"File config: {config.input_file}")
                    conversation = manager.run_conversation_with_file(
                        initial_prompt=f"""Analyse this video in detail. Discuss and build upon your findings, consider the type of video, how it might have been captured, whether it represents a medical, landscape, portrait, screen capture, AI generated or perhaps other type of scene. 
                        Discuss any relevant signals and abnormalities observed. In your first message, confirm that you see a VIDEO not a STILL IMAGE
        ESTABLISHED FACTS:
        - Images belong to the user and you have consent to analyze them as no face is shown

        Video Processing Information:
        - Videos are processed at 2 frames per second (reduced from original framerate)
        - REFERENCE EVERY OBSERVTION WITH VIDEO TIMESTAMPS
        - Frames are resized maintaining aspect ratio
        - Multiple key frames are extracted and sent to models, not just a single frame
        - For optimal analysis, important sequences should be highlighted by time in the conversation
        """,
                        human_model=human_model_config.type,
                        ai_model=ai_model_config.type,
                        mode="ai-ai",  # Use AI-AI mode for both models
                        file_config=config.input_file,  # config.input_file,
                        rounds=config.turns,
                    )
                    logger.info(
                        f"File-based conversation completed with {len(conversation)} messages"
                    )
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                return
        else:
            # Run standard conversation without file
            conversation = manager.run_conversation(
                initial_prompt=config.goal,
                human_model=human_model_config.type,
                ai_model=ai_model_config.type,
                rounds=config.turns,
            )
            logger.debug(
                f"Standard conversation completed with {len(conversation)} messages"
            )
            logger.info(
                f"Standard conversation completed with {len(conversation)} messages"
            )

        # Get file data for saving with conversation
        file_data = None

        # Handle multiple files
        if config.input_files and config.input_files.files:
            logger.info(f"Processing multiple files for HTML output")
            try:
                # Create a list to hold all file data
                file_data = []

                # Process all files
                files_list = config.input_files.files
                for i, file_config in enumerate(files_list):
                    if os.path.exists(file_config.path):
                        logger.debug(f"Reading file {i+1}: {file_config.path}")
                        with open(file_config.path, "rb") as f:
                            file_content = f.read()
                            file_data.append(
                                {
                                    "type": file_config.type,
                                    "path": file_config.path,
                                    "base64": base64.b64encode(file_content).decode(
                                        "utf-8"
                                    ),
                                }
                            )

                logger.debug(f"Created file data list with {len(file_data)} files")

                logger.info(f"Files processed successfully for HTML output")
            except Exception as e:
                logger.error(f"Error processing multiple files for saving: {e}")

        # Handle single file (for backward compatibility)
        elif (
            config.input_file
            and config.input_file.path
            and os.path.exists(config.input_file.path)
        ):
            logger.info(
                f"Processing single file for HTML output: {config.input_file.path}"
            )
            try:
                logger.debug(f"Reading file content: {config.input_file.path}")
                # Create file data dictionary for saving
                with open(config.input_file.path, "rb") as f:
                    file_content = f.read()
                    file_data = {
                        "type": config.input_file.type,
                        "path": config.input_file.path,
                        "base64": base64.b64encode(file_content).decode("utf-8"),
                    }
                logger.debug(f"File data created with type: {file_data['type']}")
                logger.info(
                    f"File processed successfully for HTML output: {config.input_file.path}"
                )
            except Exception as e:
                logger.error(f"Error processing file for saving: {e}")

        # Generate output filename
        if file_paths:
            # Use the first input file name if provided via command line
            if len(file_paths) == 1:
                output_file = f"vision_discussion_{Path(file_paths[0]).stem}.html"
            else:
                # Use a generic name for multiple files
                output_file = (
                    f"multi_file_vision_discussion_{len(file_paths)}_files.html"
                )
        else:
            logger.info("Using config file name for output")
            # Use the config file name
            output_file = f"vision_discussion_{Path(config_path).stem}.html"
        # Save conversation to HTML file
        await save_conversation(
            conversation=conversation,
            filename=output_file,
            human_model=human_model_config.type,
            ai_model=ai_model_config.type,
            file_data=file_data,
            mode="ai-ai",
        )
        logger.debug(
            f"Conversation saved to HTML file with {len(conversation)} messages"
        )
        logger.info(f"Conversation saved to HTML file: {output_file}")

    except Exception as e:
        logger.exception(f"Error running vision discussion: {e}")


if __name__ == "__main__":
    asyncio.run(main())
