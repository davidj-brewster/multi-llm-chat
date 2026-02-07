import unittest
import os
import sys
import json
import logging

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, "src"))

from adaptive_instructions import AdaptiveInstructionManager
from context_analysis import ContextAnalyzer


def test_adaptive_system():
    # Create test conversation
    conversation = [
        {"role": "user", "content": "Tell me about machine learning"},
        {
            "role": "assistant",
            "content": "Machine learning is a subset of AI that focuses on creating systems that can learn from data.",
        },
    ]

    # Initialize manager
    manager = AdaptiveInstructionManager()

    # Test instruction generation
    instructions = manager.generate_instructions(conversation, "machine learning")
    print("\nGenerated Instructions:")
    print("----------------------")
    print(instructions)

    # Test with empty conversation
    initial_instructions = manager.generate_instructions([], "machine learning")
    print("\nInitial Instructions:")
    print("-------------------")
    print(initial_instructions)

    # Test with longer conversation
    conversation.append(
        {"role": "user", "content": "How does neural network training work?"}
    )
    conversation.append(
        {
            "role": "assistant",
            "content": "Neural networks are trained through backpropagation, adjusting weights based on error gradients.",
        }
    )

    updated_instructions = manager.generate_instructions(
        conversation, "machine learning"
    )
    print("\nUpdated Instructions:")
    print("-------------------")
    print(updated_instructions)


if __name__ == "__main__":
    test_adaptive_system()
