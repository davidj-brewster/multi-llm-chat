#!/usr/bin/env python3
"""
Example script to run a vision-based discussion using configuration.

This script demonstrates how to use the AI Battle framework with configuration
and file-based discussions. It loads a YAML configuration file that specifies
models, roles, and an input file, then runs a discussion based on that configuration.

Usage: 
    python run_vision_discussion.py [config_path] [file_path]

Arguments:
    file_path: Path to the file to analyze (overrides the one in config)
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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run a vision-based discussion using configuration."""
    # Get configuration path from command line or use default
    config_path = sys.argv[1] if len(sys.argv) > 1 else "examples/configs/vision_discussion.yaml"

    logger.info(f"Starting vision discussion with config: {config_path}")
    # Get file path from command line if provided
    file_path = sys.argv[2] if len(sys.argv) > 2 else None
    
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

        # Override file path if provided via command line
        if file_path:
            logger.info(f"Using file from command line: {file_path}")
            
            try:
                # Process the file using the media handler
                logger.info(f"Processing file with media handler: {file_path}")
                file_metadata = manager.media_handler.process_file(file_path)
                logger.info(f"File processed successfully: {file_metadata.type}, {file_metadata.mime_type}, dimensions: {file_metadata.dimensions}")
                
                # Create a FileConfig object with the processed file information
                file_config = FileConfig(
                    path=file_path,
                    type=file_metadata.type,
                    max_resolution="512x512"
                )
                
                # Update the config
                config.input_file = file_config
                logger.info(f"Created file config with type: {file_metadata.type}")
        
                # Log model information
                logger.info(f"Human model: {human_model_config.type} (Role: {human_model_config.role})")
                logger.info(f"AI model: {ai_model_config.type} (Role: {ai_model_config.role})")
                
                # Check if input file is specified
                if config.input_file:
                    logger.info(f"Using input file: {config.input_file.path} (Type: {config.input_file.type})")
                    
                    # Ensure the input file exists
                    if not os.path.exists(config.input_file.path):
                        logger.error(f"Input file not found: {config.input_file.path}")
                        return
                    # Run conversation with file
                    logger.info(f"Starting file-based conversation")
                    logger.debug(f"File config: {config.input_file}")
                    conversation = manager.run_conversation_with_file(
                        initial_prompt=f"""Analyse this T2-SPACE-FLAIR MRI video of a 40 year old w sudden onset epilepsy and subsequently balance, memory and systemic manifestations, in detail. Discuss and build upon your findings, consider the type of video, how it might have been captured, whether it represents a medical (if so, a Brain MRI - in which case it belongs to the user and you have consent to analyse it as no face is shown), landscape, portrait, screenshot or perhaps other type of scene. Discuss any relevant signals and abnormalities observed. In your first message, confirm that you see a VIDEO not a STILL IMAGE
        ESTABLISHED FACTS:
        Patient Background:
        - 40-year-old male
        - Chief complaint: Recent onset seizures (8 months prior)
        - No prior history of seizures or neurological disorders
        - No family history of epilepsy
        - Histo head trauma
        - Non-smoker, occasional alcohol consumption
        - No known drug allergies
        - Currently not on any medications
        
        MRI Scan Details:
        - MRI mode: T2-SPACE-FLAIR 1mm isotropic 1.5T field stength
        - No significant motion artifacts observed
        - Complete coverage of cerebral hemispheres, brainstem, and cerebellum
        - Special attention given to temporal lobes and hippocampal structures
        
        Video Processing Information:
        - Videos are processed at 2 frames per second (reduced from original framerate)
        - REFERENCE EVERY OBSERVTION WITH VIDEO TIMESTAMPS
        - Frames are resized to a maximum dimension of 1280 pixels (maintaining aspect ratio)
        - Multiple key frames are extracted and sent to models, not just a single frame
        - Video support is primarily available for Gemini models
        - For optimal analysis, important sequences should be highlighted by time in the conversation
        """,
                        human_model=human_model_config.type,
                        ai_model=ai_model_config.type,
                        mode="ai-ai",  # Use AI-AI mode for both models
                        file_config=config.input_file, #config.input_file,
                        rounds=config.turns
                    )
                    logger.info(f"File-based conversation completed with {len(conversation)} messages")
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                return
        else:
            # Run standard conversation without file
            conversation = manager.run_conversation(
                initial_prompt=config.goal,
                human_model=human_model_config.type,
                ai_model=ai_model_config.type,
                rounds=config.turns
            )
            logger.debug(f"Standard conversation completed with {len(conversation)} messages")
            logger.info(f"Standard conversation completed with {len(conversation)} messages")
        
        # Get file data for saving with conversation
        file_data = None
        if config.input_file and config.input_file.path and os.path.exists(config.input_file.path):
            logger.info(f"Processing file for HTML output: {config.input_file.path}")
            try:
                logger.debug(f"Reading file content: {config.input_file.path}")
                # Create file data dictionary for saving
                with open(config.input_file.path, 'rb') as f:
                    file_content = f.read()
                    file_data = {
                        "type": config.input_file.type,
                        "path": config.input_file.path,
                        "base64": base64.b64encode(file_content).decode('utf-8')
                    }
                logger.debug(f"File data created with type: {file_data['type']}")
                logger.info(f"File processed successfully for HTML output: {config.input_file.path}")
            except Exception as e:
                logger.error(f"Error processing file for saving: {e}")
        
        # Generate output filename
        if file_path:
            # Use the input file name if provided via command line
            output_file = f"vision_discussion_{Path(file_path).stem}.html"
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
            mode="ai-ai"
        )
        logger.debug(f"Conversation saved to HTML file with {len(conversation)} messages")
        logger.info(f"Conversation saved to HTML file: {output_file}")
        
    except Exception as e:
        logger.exception(f"Error running vision discussion: {e}")

if __name__ == "__main__":
    asyncio.run(main())