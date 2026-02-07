"""Tests for the adaptive instruction manager with InstructionSet,
persona support, pathology escalation, and updated thresholds.
"""

import unittest
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, "src"))

from adaptive_instructions import AdaptiveInstructionManager, InstructionSet
from context_analysis import ContextVector, ContextAnalyzer


class TestInstructionSet(unittest.TestCase):
    """Test InstructionSet dataclass."""

    def test_str_concatenation(self):
        iset = InstructionSet(
            persona="You are a bold leader.",
            template="Discuss the topic critically.",
            interventions="Stop repeating yourself.",
            constraints="Max 3000 tokens.",
        )
        result = str(iset)
        self.assertIn("bold leader", result)
        self.assertIn("critically", result)
        self.assertIn("repeating", result)
        self.assertIn("3000", result)

    def test_str_skips_empty(self):
        iset = InstructionSet(
            persona="You are a bold leader.",
            template="",
            interventions="",
            constraints="Max tokens.",
        )
        result = str(iset)
        self.assertIn("bold leader", result)
        self.assertIn("Max tokens", result)
        # Should not have empty sections joined
        self.assertNotIn("\n\n\n\n", result)

    def test_empty_instruction_set(self):
        iset = InstructionSet()
        self.assertEqual(str(iset), "")


class TestAdaptiveInstructionManager(unittest.TestCase):
    """Test the main manager with persona, pathology, and escalation."""

    def setUp(self):
        self.manager = AdaptiveInstructionManager(mode="ai-ai")
        self.domain = "machine learning"
        self.convo = [
            {"role": "user", "content": "What are the fundamental principles of machine learning?"},
            {"role": "assistant", "content": "The key principles include training data quality, model selection, and avoiding overfitting."},
            {"role": "user", "content": "Can you explain overfitting and regularization?"},
            {"role": "assistant", "content": "Overfitting occurs when a model memorizes noise. Regularization techniques like L1 and L2 help."},
        ]

    def test_generate_instructions_returns_string(self):
        result = self.manager.generate_instructions(self.convo, self.domain)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_generate_stores_last_instruction_set(self):
        self.manager.generate_instructions(self.convo, self.domain)
        self.assertIsNotNone(self.manager.last_instruction_set)
        self.assertIsInstance(self.manager.last_instruction_set, InstructionSet)

    def test_generate_stores_last_context(self):
        self.manager.generate_instructions(self.convo, self.domain)
        self.assertIsNotNone(self.manager.last_context)

    def test_domain_in_output(self):
        result = self.manager.generate_instructions(self.convo, self.domain)
        self.assertIn(self.domain, result)

    def test_no_meta_prompting_mode(self):
        manager = AdaptiveInstructionManager(mode="no-meta-prompting")
        result = manager.generate_instructions(self.convo, self.domain)
        self.assertIn(self.domain, result)
        self.assertIn("RESTRICT OUTPUTS", result)

    def test_empty_conversation(self):
        result = self.manager.generate_instructions([], self.domain)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestPersonaIntegration(unittest.TestCase):
    """Test persona wiring through instructions."""

    def test_persona_in_output(self):
        manager = AdaptiveInstructionManager(mode="ai-ai", persona="You are a senior architect with 20 years experience. You are direct, commanding, and intolerant of vague thinking.")
        result = manager.generate_instructions(
            [{"role": "user", "content": "Discuss system design."},
             {"role": "assistant", "content": "Let us analyze the architecture."}],
            "system design",
        )
        self.assertIn("senior architect", result)
        self.assertIn("IDENTITY AND PERSONA", result)

    def test_default_persona_when_none(self):
        manager = AdaptiveInstructionManager(mode="ai-ai", persona=None)
        result = manager.generate_instructions(
            [{"role": "user", "content": "Hello"},
             {"role": "assistant", "content": "Hi there"}],
            "general",
        )
        self.assertIn("human expert", result.lower())

    def test_persona_never_break_character(self):
        manager = AdaptiveInstructionManager(mode="ai-ai", persona="Bold leader persona")
        manager.generate_instructions(
            [{"role": "user", "content": "Test"},
             {"role": "assistant", "content": "Response"}],
            "test",
        )
        iset = manager.last_instruction_set
        self.assertIn("stay in character", iset.persona)


class TestPathologyEscalation(unittest.TestCase):
    """Test pathology detection and escalation levels."""

    def setUp(self):
        self.manager = AdaptiveInstructionManager(mode="ai-ai")

    def test_pathology_tracker_initialized(self):
        for key in ["repetition", "agreement", "formulaic", "readability_drift"]:
            self.assertIn(key, self.manager._pathology_tracker)
            self.assertEqual(self.manager._pathology_tracker[key]["count"], 0)

    def test_tracker_increments_on_detection(self):
        self.manager._update_tracker("repetition", True)
        self.assertEqual(self.manager._pathology_tracker["repetition"]["count"], 1)
        self.manager._update_tracker("repetition", True)
        self.assertEqual(self.manager._pathology_tracker["repetition"]["count"], 2)

    def test_tracker_resets_on_no_detection(self):
        self.manager._update_tracker("repetition", True)
        self.manager._update_tracker("repetition", True)
        self.manager._update_tracker("repetition", False)
        self.assertEqual(self.manager._pathology_tracker["repetition"]["count"], 0)

    def test_level_1_intervention(self):
        self.manager._update_tracker("repetition", True)
        text = self.manager._get_intervention("repetition")
        self.assertIsNotNone(text)
        self.assertIn("new angle", text.lower())

    def test_level_2_intervention(self):
        self.manager._update_tracker("repetition", True)
        self.manager._update_tracker("repetition", True)
        text = self.manager._get_intervention("repetition")
        self.assertIsNotNone(text)
        self.assertIn("STOP", text)

    def test_level_3_intervention(self):
        for _ in range(4):
            self.manager._update_tracker("repetition", True)
        text = self.manager._get_intervention("repetition")
        self.assertIsNotNone(text)
        self.assertIn("CRITICAL", text)

    def test_no_intervention_when_clear(self):
        text = self.manager._get_intervention("repetition")
        self.assertIsNone(text)


class TestTemplateSelection(unittest.TestCase):
    """Test updated template selection thresholds."""

    def setUp(self):
        self.manager = AdaptiveInstructionManager(mode="ai-ai")

    def test_early_phase_selects_exploratory(self):
        ctx = ContextVector(conversation_phase=0.1, semantic_coherence=0.8)
        template = self.manager._select_template(ctx, "ai-ai")
        self.assertIsInstance(template, str)
        self.assertGreater(len(template), 0)

    def test_low_coherence_selects_structured(self):
        ctx = ContextVector(conversation_phase=0.5, semantic_coherence=0.2)
        template = self.manager._select_template(ctx, "ai-ai")
        self.assertIsInstance(template, str)

    def test_high_fog_selects_synthesis(self):
        ctx = ContextVector(conversation_phase=0.5, semantic_coherence=0.8, gunning_fog_index=16.0)
        template = self.manager._select_template(ctx, "ai-ai")
        self.assertIsInstance(template, str)

    def test_rich_vocab_mature_selects_critical(self):
        ctx = ContextVector(
            conversation_phase=0.8, semantic_coherence=0.8,
            gunning_fog_index=10.0, vocabulary_richness=0.8,
        )
        template = self.manager._select_template(ctx, "ai-ai")
        self.assertIsInstance(template, str)

    def test_goal_detection_in_domain(self):
        ctx = ContextVector(domain_info="GOAL: Write a comprehensive essay")
        template = self.manager._select_template(ctx, "ai-ai")
        self.assertIsInstance(template, str)


class TestModeHandling(unittest.TestCase):
    """Test different conversation modes."""

    def test_ai_ai_mode(self):
        manager = AdaptiveInstructionManager(mode="ai-ai")
        result = manager.generate_instructions(
            [{"role": "user", "content": "Discuss AI safety."},
             {"role": "assistant", "content": "AI safety involves alignment and robustness."}],
            "AI safety",
        )
        self.assertIn("HUMAN EXPERT", result)

    def test_human_ai_mode(self):
        manager = AdaptiveInstructionManager(mode="human-ai")
        result = manager.generate_instructions(
            [{"role": "user", "content": "Tell me about ML."},
             {"role": "assistant", "content": "ML is about learning from data."}],
            "machine learning",
            role="user",
        )
        self.assertIsInstance(result, str)

    def test_default_mode(self):
        manager = AdaptiveInstructionManager(mode="default")
        result = manager.generate_instructions(
            [{"role": "user", "content": "Hello"},
             {"role": "assistant", "content": "Hi"}],
            "general",
        )
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
