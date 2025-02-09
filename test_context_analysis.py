import unittest
from context_analysis import ContextAnalyzer, ContextVector

class TestContextAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = ContextAnalyzer()
        self.test_conversation = [
            {
                "role": "user",
                "content": "What are the fundamental principles of machine learning?"
            },
            {
                "role": "assistant",
                "content": "The key principles of machine learning include: 1) Quality of training data is crucial 2) Model selection matters 3) Avoiding overfitting is essential"
            },
            {
                "role": "user",
                "content": "Can you explain more about overfitting? Why is it a problem?"
            },
            {
                "role": "assistant",
                "content": "Overfitting occurs when a model learns the noise in the training data too well. While it performs excellently on training data, it fails to generalize to new, unseen data. This happens because the model is too complex relative to the underlying pattern it needs to learn."
            }
        ]

    def test_semantic_coherence(self):
        context = self.analyzer.analyze(self.test_conversation)
        self.assertIsInstance(context.semantic_coherence, float)
        self.assertTrue(0 <= context.semantic_coherence <= 1)
        
    def test_topic_evolution(self):
        context = self.analyzer.analyze(self.test_conversation)
        self.assertIsInstance(context.topic_evolution, dict)
        self.assertTrue(len(context.topic_evolution) > 0)
        self.assertTrue(all(isinstance(v, float) for v in context.topic_evolution.values()))
        
    def test_response_patterns(self):
        context = self.analyzer.analyze(self.test_conversation)
        self.assertIsInstance(context.response_patterns, dict)
        expected_patterns = {'question_frequency', 'elaboration_frequency', 
                           'challenge_frequency', 'agreement_frequency'}
        self.assertTrue(all(pattern in context.response_patterns for pattern in expected_patterns))
        
    def test_engagement_metrics(self):
        context = self.analyzer.analyze(self.test_conversation)
        self.assertIsInstance(context.engagement_metrics, dict)
        expected_metrics = {'avg_response_length', 'turn_taking_balance'}
        self.assertTrue(all(metric in context.engagement_metrics for metric in expected_metrics))
        
    def test_cognitive_load(self):
        context = self.analyzer.analyze(self.test_conversation)
        self.assertIsInstance(context.cognitive_load, float)
        self.assertTrue(0 <= context.cognitive_load <= 1)
        
    def test_knowledge_depth(self):
        context = self.analyzer.analyze(self.test_conversation)
        self.assertIsInstance(context.knowledge_depth, float)
        self.assertTrue(0 <= context.knowledge_depth <= 1)
        
    def test_reasoning_patterns(self):
        context = self.analyzer.analyze(self.test_conversation)
        self.assertIsInstance(context.reasoning_patterns, dict)
        expected_patterns = {'deductive', 'inductive', 'abductive', 
                           'analogical', 'causal'}
        self.assertTrue(all(pattern in context.reasoning_patterns for pattern in expected_patterns))
        
    def test_uncertainty_markers(self):
        context = self.analyzer.analyze(self.test_conversation)
        self.assertIsInstance(context.uncertainty_markers, dict)
        expected_markers = {'uncertainty', 'confidence', 'qualification'}
        self.assertTrue(all(marker in context.uncertainty_markers for marker in expected_markers))
        
    def test_empty_conversation(self):
        empty_conversation = []
        context = self.analyzer.analyze(empty_conversation)
        self.assertIsInstance(context, ContextVector)
        
    def test_single_message_conversation(self):
        single_message = [{"role": "user", "content": "Hello"}]
        context = self.analyzer.analyze(single_message)
        self.assertIsInstance(context, ContextVector)

if __name__ == '__main__':
    unittest.main()