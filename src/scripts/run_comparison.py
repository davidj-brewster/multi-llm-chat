#!/usr/bin/env python3
"""
Multi-mode comparison runner - replicates ai_battle.py main() behavior.

Usage:
    python -m scripts.run_comparison [config.yaml]
    python -m scripts.run_comparison config.yaml --modes ai-ai human-ai
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# TODO: Once modules are refactored, update these imports
from ai_battle import ConversationManager, save_conversation, save_arbiter_report, save_metrics_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    """Run multi-mode conversation comparison."""
    parser = argparse.ArgumentParser(description="Run multi-mode conversation comparison")
    parser.add_argument(
        "config",
        nargs="?",
        default="docs/examples/configs/discussion_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["ai-ai", "human-ai", "no-meta"],
        choices=["ai-ai", "human-ai", "no-meta"],
    )
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Multi-Mode Comparison")
    logger.info(f"Config: {args.config}")
    logger.info(f"Modes: {', '.join(args.modes)}")
    logger.info(f"Output: {output_dir}")

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

        conversations = {}

        # Run conversations in each mode
        for i, mode in enumerate(args.modes, 1):
            logger.info(f"[{i}/{len(args.modes)}] Running {mode} mode...")

            conversation = manager.run_conversation(
                initial_prompt=config.goal,
                human_model=human_model,
                ai_model=ai_model,
                mode=mode,
                rounds=rounds,
            )

            conversations[mode] = conversation

            # Save individual conversation
            output_file = str(output_dir / f"conversation_{mode}.html")
            await save_conversation(
                conversation=conversation,
                filename=output_file,
                human_model=human_model,
                ai_model=ai_model,
                mode=mode,
                signal_history=manager.signal_history,
            )
            logger.info(f"  ✓ Saved: {output_file}")

        # Run arbiter if we have multiple conversations
        if len(conversations) >= 2:
            logger.info("Running arbiter evaluation...")
            try:
                from arbiter_v4 import evaluate_conversations
                evaluation = await evaluate_conversations(
                    conversations=list(conversations.values()),
                    modes=args.modes,
                )
                await save_arbiter_report(evaluation)
                logger.info("  ✓ Arbiter report saved")
            except Exception as e:
                logger.warning(f"  Arbiter failed: {e}")

        # Run metrics if we have ai-ai and human-ai
        if "ai-ai" in conversations and "human-ai" in conversations:
            logger.info("Running metrics analysis...")
            try:
                await save_metrics_report(
                    conversations["ai-ai"],
                    conversations["human-ai"],
                )
                logger.info("  ✓ Metrics report saved")
            except Exception as e:
                logger.warning(f"  Metrics failed: {e}")

        logger.info(f"✓ Comparison complete! Results in {output_dir}")
        return 0

    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
