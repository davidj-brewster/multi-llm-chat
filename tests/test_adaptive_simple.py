"""Simple integration test for the adaptive instruction system."""

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, "src"))

from adaptive_instructions import AdaptiveInstructionManager, InstructionSet
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

    # Initialize manager with persona
    manager = AdaptiveInstructionManager(
        mode="ai-ai",
        persona="You are a senior ML engineer who values precision and challenges vague claims."
    )

    # Test instruction generation
    instructions = manager.generate_instructions(conversation, "machine learning")
    print("\nGenerated Instructions:")
    print("----------------------")
    print(instructions[:500])
    print(f"\n... ({len(instructions)} chars total)")

    # Check InstructionSet was stored
    iset = manager.last_instruction_set
    assert iset is not None, "InstructionSet should be stored"
    assert isinstance(iset, InstructionSet), f"Expected InstructionSet, got {type(iset)}"
    print(f"\nInstructionSet fields:")
    print(f"  persona: {iset.persona[:100]}...")
    print(f"  template: {iset.template[:100]}...")
    print(f"  interventions: '{iset.interventions[:100]}'")
    print(f"  constraints: {iset.constraints[:100]}...")

    # Check context was stored
    ctx = manager.last_context
    assert ctx is not None, "Last context should be stored"
    print(f"\nContext signals:")
    print(f"  flesch_reading_ease: {ctx.flesch_reading_ease:.1f}")
    print(f"  gunning_fog_index: {ctx.gunning_fog_index:.1f}")
    print(f"  vocabulary_richness: {ctx.vocabulary_richness:.2f}")
    print(f"  sentence_variety: {ctx.sentence_variety:.2f}")
    print(f"  repetition_score: {ctx.repetition_score:.2f}")
    print(f"  agreement_saturation: {ctx.agreement_saturation:.2f}")
    print(f"  formulaic_score: {ctx.formulaic_score:.2f}")
    print(f"  conversation_phase: {ctx.conversation_phase:.2f}")
    print(f"  semantic_coherence: {ctx.semantic_coherence:.2f}")
    print(f"  cognitive_load (compat): {ctx.cognitive_load:.2f}")
    print(f"  knowledge_depth (compat): {ctx.knowledge_depth:.2f}")

    # Test with empty conversation
    initial_instructions = manager.generate_instructions([], "machine learning")
    print("\nInitial Instructions (empty convo):")
    print("-------------------")
    print(initial_instructions[:300])

    # Test with longer conversation
    conversation.extend([
        {"role": "user", "content": "How does neural network training work?"},
        {
            "role": "assistant",
            "content": "Neural networks are trained through backpropagation, adjusting weights based on error gradients.",
        },
    ])

    updated_instructions = manager.generate_instructions(
        conversation, "machine learning"
    )
    print("\nUpdated Instructions (longer convo):")
    print("-------------------")
    print(updated_instructions[:300])

    # Test pathology escalation
    print("\n\nPathology Escalation Test:")
    print("=" * 40)
    manager._update_tracker("repetition", True)
    print(f"Level 1: {manager._get_intervention('repetition')}")
    manager._update_tracker("repetition", True)
    print(f"Level 2: {manager._get_intervention('repetition')}")
    manager._update_tracker("repetition", True)
    manager._update_tracker("repetition", True)
    print(f"Level 3: {manager._get_intervention('repetition')}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_adaptive_system()
