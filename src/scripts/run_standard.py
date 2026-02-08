#!/usr/bin/env python3
"""
Standard conversation runner.

Usage:
    python -m scripts.run_standard [config.yaml]
    python -m scripts.run_standard config.yaml --rounds 4
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# TODO: Once ConversationManager is refactored, import from core.conversation_manager
from ai_battle import ConversationManager, save_conversation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    """Run a standard conversation from configuration."""
    parser = argparse.ArgumentParser(description="Run standard AI conversation")
    parser.add_argument(
        "config",
        nargs="?",
        default="docs/examples/configs/discussion_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--rounds", type=int, help="Override number of rounds")
    parser.add_argument("--output", help="Output filename")
    parser.add_argument("--mode", default="ai-ai", choices=["ai-ai", "human-ai", "no-meta"])
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Loading configuration: {args.config}")

    try:
        manager = ConversationManager.from_config(args.config)
        config = manager.config
        models = list(config.models.items())

        if len(models) < 2:
            logger.error("At least two models must be configured")
            return 1

        human_model = models[0][1].type
        ai_model = models[1][1].type
        rounds = args.rounds or config.turns

        logger.info(f"Starting: {human_model} <-> {ai_model} ({rounds} rounds)")

        conversation = manager.run_conversation(
            initial_prompt=config.goal,
            human_model=human_model,
            ai_model=ai_model,
            mode=args.mode,
            rounds=rounds,
        )

        output_file = args.output or f"conversation_{Path(args.config).stem}.html"
        await save_conversation(
            conversation=conversation,
            filename=output_file,
            human_model=human_model,
            ai_model=ai_model,
            mode=args.mode,
            signal_history=manager.signal_history,
        )

        logger.info(f"✓ Saved: {output_file}")
        return 0

    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
