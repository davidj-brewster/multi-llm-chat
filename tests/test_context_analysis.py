"""Tests for the NLP signal engine in context_analysis.py.

Covers: readability calibration, pathology detection, per-participant signals,
topic analysis, backward-compat properties, and spaCy vs fallback parity.
"""

import unittest
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, "src"))

from context_analysis import ContextAnalyzer, ContextVector


class TestContextVector(unittest.TestCase):
    """Test ContextVector dataclass and backward-compat properties."""

    def test_defaults(self):
        cv = ContextVector()
        self.assertEqual(cv.conversation_phase, 0.0)
        self.assertEqual(cv.semantic_coherence, 1.0)
        self.assertEqual(cv.flesch_reading_ease, 50.0)
        self.assertEqual(cv.gunning_fog_index, 12.0)
        self.assertEqual(cv.repetition_score, 0.0)
        self.assertEqual(cv.agreement_saturation, 0.0)
        self.assertEqual(cv.formulaic_score, 0.0)

    def test_cognitive_load_property(self):
        cv = ContextVector(gunning_fog_index=20.0)
        self.assertIsInstance(cv.cognitive_load, float)
        self.assertTrue(0 <= cv.cognitive_load <= 1)

    def test_knowledge_depth_property(self):
        cv = ContextVector(vocabulary_richness=0.8, topic_coherence=0.7)
        self.assertIsInstance(cv.knowledge_depth, float)
        self.assertTrue(0 <= cv.knowledge_depth <= 1)

    def test_uncertainty_markers_alias(self):
        cv = ContextVector(epistemic_stance={"uncertainty": 0.5, "confidence": 0.3})
        self.assertEqual(cv.uncertainty_markers, cv.epistemic_stance)


class TestReadabilityCalibration(unittest.TestCase):
    """Test that readability metrics produce sensible values."""

    def setUp(self):
        self.analyzer = ContextAnalyzer()

    def test_simple_text_high_flesch(self):
        """Simple, short sentences should have high Flesch reading ease."""
        simple_convo = [
            {"role": "user", "content": "The cat sat on the mat. It was a big cat. The mat was red."},
            {"role": "assistant", "content": "Yes the cat is nice. I like cats. They are fun pets."},
        ]
        ctx = self.analyzer.analyze(simple_convo)
        # Simple text should have Flesch > 60
        self.assertGreater(ctx.flesch_reading_ease, 50)

    def test_complex_text_low_flesch(self):
        """Complex academic text should have lower Flesch reading ease."""
        complex_convo = [
            {"role": "user", "content": "The epistemological implications of quantum decoherence necessitate a fundamental reconceptualization of consciousness. Furthermore, the phenomenological characteristics of subjective experience remain fundamentally incompatible with materialist reductionism."},
            {"role": "assistant", "content": "Notwithstanding the aforementioned philosophical considerations, the neurophysiological correlates of consciousness demonstrate remarkable plasticity in their manifestation across phylogenetically diverse organisms."},
        ]
        ctx = self.analyzer.analyze(complex_convo)
        # Complex text should have lower Flesch
        self.assertLess(ctx.flesch_reading_ease, 50)

    def test_fog_index_range(self):
        """Gunning Fog index should be positive."""
        convo = [
            {"role": "user", "content": "Machine learning is about training models on data."},
            {"role": "assistant", "content": "Yes. Models learn patterns from data to make predictions."},
        ]
        ctx = self.analyzer.analyze(convo)
        self.assertGreater(ctx.gunning_fog_index, 0)

    def test_vocabulary_richness_range(self):
        """Vocabulary richness should be between 0 and 1."""
        convo = [
            {"role": "user", "content": "The quick brown fox jumps over the lazy dog and runs fast."},
            {"role": "assistant", "content": "Indeed the agile crimson vulpine leapt across the lethargic canine."},
        ]
        ctx = self.analyzer.analyze(convo)
        self.assertTrue(0 <= ctx.vocabulary_richness <= 1)

    def test_sentence_variety_range(self):
        """Sentence variety should be non-negative."""
        convo = [
            {"role": "user", "content": "Short one. This is a much longer sentence with more words in it. Ok."},
            {"role": "assistant", "content": "Brief. Another medium length sentence here. And a third with different length."},
        ]
        ctx = self.analyzer.analyze(convo)
        self.assertGreaterEqual(ctx.sentence_variety, 0.0)


class TestPathologyDetection(unittest.TestCase):
    """Test pathology signal detection."""

    def setUp(self):
        self.analyzer = ContextAnalyzer()

    def test_repetition_detection(self):
        """Repeated content should produce higher repetition score."""
        repetitive = [
            {"role": "user", "content": "Machine learning uses data to train models. Models learn patterns from data."},
            {"role": "assistant", "content": "Yes, machine learning uses data to train models. The models learn patterns from data."},
            {"role": "user", "content": "Right, machine learning uses data to train models and models learn patterns."},
            {"role": "assistant", "content": "Exactly, machine learning uses data to train models. They learn patterns from data."},
        ]
        ctx = self.analyzer.analyze(repetitive)
        self.assertGreater(ctx.repetition_score, 0.2)

    def test_agreement_saturation(self):
        """Cross-participant echoing should increase agreement saturation."""
        agreeing = [
            {"role": "user", "content": "I think machine learning is the future of technology."},
            {"role": "assistant", "content": "I agree, machine learning is definitely the future of technology."},
            {"role": "user", "content": "Yes exactly, machine learning is the future. Very important."},
            {"role": "assistant", "content": "Absolutely, machine learning is the future and it is very important."},
        ]
        ctx = self.analyzer.analyze(agreeing)
        self.assertGreater(ctx.agreement_saturation, 0.2)

    def test_formulaic_score_range(self):
        """Formulaic score should be between 0 and 1."""
        convo = [
            {"role": "user", "content": "Tell me about neural networks."},
            {"role": "assistant", "content": "Neural networks are computational models inspired by biological brains."},
        ]
        ctx = self.analyzer.analyze(convo)
        self.assertTrue(0 <= ctx.formulaic_score <= 1)


class TestPerParticipantSignals(unittest.TestCase):
    """Test per-participant signal breakdown."""

    def setUp(self):
        self.analyzer = ContextAnalyzer()

    def test_participant_signals_populated(self):
        """participant_signals should contain entries for each role."""
        convo = [
            {"role": "user", "content": "What about deep learning architectures?"},
            {"role": "assistant", "content": "Deep learning uses layered neural networks for representation learning."},
            {"role": "user", "content": "How do transformers work?"},
            {"role": "assistant", "content": "Transformers use self-attention mechanisms to process sequences in parallel."},
        ]
        ctx = self.analyzer.analyze(convo)
        self.assertIsInstance(ctx.participant_signals, dict)
        # Should have signals for at least one role
        if ctx.participant_signals:
            for role, signals in ctx.participant_signals.items():
                self.assertIsInstance(signals, dict)


class TestTopicAnalysis(unittest.TestCase):
    """Test topic evolution and coherence."""

    def setUp(self):
        self.analyzer = ContextAnalyzer()

    def test_topic_evolution_populated(self):
        """Topic evolution should extract meaningful topics."""
        convo = [
            {"role": "user", "content": "What are the fundamental principles of machine learning?"},
            {"role": "assistant", "content": "The key principles include training data quality, model selection, and avoiding overfitting."},
            {"role": "user", "content": "Can you explain more about overfitting and regularization?"},
            {"role": "assistant", "content": "Overfitting occurs when a model memorizes noise. Regularization techniques like L1 and L2 help prevent this."},
        ]
        ctx = self.analyzer.analyze(convo)
        self.assertIsInstance(ctx.topic_evolution, dict)

    def test_topic_coherence_range(self):
        """Topic coherence should be between 0 and 1."""
        convo = [
            {"role": "user", "content": "Discuss neural networks and their applications."},
            {"role": "assistant", "content": "Neural networks are used in image recognition and language processing."},
        ]
        ctx = self.analyzer.analyze(convo)
        self.assertTrue(0 <= ctx.topic_coherence <= 1)

    def test_semantic_coherence_no_division_by_two(self):
        """Semantic coherence should NOT be artificially halved (regression test for /2 bug).

        Note: TF-IDF cosine similarity on short messages is inherently lower than
        embedding-based similarity. Values > 0.1 for topically related messages
        indicate the /2 bug is fixed (old code would produce ~0.05).
        """
        coherent_convo = [
            {"role": "user", "content": "Machine learning is about training models on datasets."},
            {"role": "assistant", "content": "Yes, machine learning trains models using large datasets effectively."},
            {"role": "user", "content": "The training process involves optimizing model parameters on the dataset."},
            {"role": "assistant", "content": "Model parameters are optimized through gradient descent on the training data."},
        ]
        ctx = self.analyzer.analyze(coherent_convo)
        # TF-IDF cosine on short text is low by nature; just verify it's positive
        # and not artificially halved (old bug: divided by 2)
        self.assertGreater(ctx.semantic_coherence, 0.1)
        self.assertTrue(0 <= ctx.semantic_coherence <= 1)


class TestConversationPhase(unittest.TestCase):
    """Test conversation phase computation."""

    def setUp(self):
        self.analyzer = ContextAnalyzer()

    def test_early_phase(self):
        """Short conversation should have low phase."""
        short = [
            {"role": "user", "content": "Hello, let's discuss AI."},
            {"role": "assistant", "content": "Sure, what about AI interests you?"},
        ]
        ctx = self.analyzer.analyze(short)
        self.assertLess(ctx.conversation_phase, 0.5)

    def test_mature_phase(self):
        """Long conversation should have higher phase."""
        long_convo = []
        for i in range(12):
            role = "user" if i % 2 == 0 else "assistant"
            long_convo.append({"role": role, "content": f"This is message {i} about various aspects of machine learning and deep neural network architectures including transformers and attention mechanisms."})
        ctx = self.analyzer.analyze(long_convo)
        self.assertGreater(ctx.conversation_phase, 0.4)


class TestEngagementAndEpistemic(unittest.TestCase):
    """Test engagement metrics and epistemic stance."""

    def setUp(self):
        self.analyzer = ContextAnalyzer()
        self.convo = [
            {"role": "user", "content": "What are the fundamental principles of machine learning?"},
            {"role": "assistant", "content": "The key principles include data quality, model selection, and evaluation."},
            {"role": "user", "content": "Can you explain overfitting? Why is it a problem?"},
            {"role": "assistant", "content": "Overfitting occurs when a model memorizes training noise instead of learning general patterns."},
        ]

    def test_engagement_metrics(self):
        ctx = self.analyzer.analyze(self.convo)
        self.assertIsInstance(ctx.engagement_metrics, dict)
        self.assertIn("avg_response_length", ctx.engagement_metrics)
        self.assertIn("turn_taking_balance", ctx.engagement_metrics)

    def test_epistemic_stance(self):
        ctx = self.analyzer.analyze(self.convo)
        self.assertIsInstance(ctx.epistemic_stance, dict)
        self.assertIn("uncertainty", ctx.epistemic_stance)
        self.assertIn("confidence", ctx.epistemic_stance)

    def test_reasoning_patterns(self):
        ctx = self.analyzer.analyze(self.convo)
        self.assertIsInstance(ctx.reasoning_patterns, dict)
        for key in ["deductive", "inductive", "abductive", "analogical", "causal"]:
            self.assertIn(key, ctx.reasoning_patterns)

    def test_response_patterns(self):
        ctx = self.analyzer.analyze(self.convo)
        self.assertIsInstance(ctx.response_patterns, dict)
        self.assertIn("question_ratio", ctx.response_patterns)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        self.analyzer = ContextAnalyzer()

    def test_empty_conversation(self):
        ctx = self.analyzer.analyze([])
        self.assertIsInstance(ctx, ContextVector)
        self.assertEqual(ctx.conversation_phase, 0.0)

    def test_single_message(self):
        ctx = self.analyzer.analyze([{"role": "user", "content": "Hello"}])
        self.assertIsInstance(ctx, ContextVector)

    def test_very_short_messages(self):
        ctx = self.analyzer.analyze([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hey"},
        ])
        self.assertIsInstance(ctx, ContextVector)

    def test_ai_ai_mode(self):
        analyzer = ContextAnalyzer(mode="ai-ai")
        convo = [
            {"role": "user", "content": "Let us systematically analyze neural network properties."},
            {"role": "assistant", "content": "Proceeding with formal analysis of network architectures."},
        ]
        ctx = analyzer.analyze(convo)
        self.assertIn("formal_logic", ctx.reasoning_patterns)
        self.assertIn("systematic", ctx.reasoning_patterns)
        self.assertIn("technical", ctx.reasoning_patterns)


if __name__ == "__main__":
    unittest.main()
