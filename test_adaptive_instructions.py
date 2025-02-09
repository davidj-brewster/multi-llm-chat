import unittest
from adaptive_instructions import AdaptiveInstructionManager
from context_analysis import ContextVector

class TestAdaptiveInstructions(unittest.TestCase):
    def setUp(self):
        self.manager = AdaptiveInstructionManager()
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
                "content": "Overfitting occurs when a model learns the noise in the training data too well. While it performs excellently on training data, it fails to generalize to new, unseen data."
            }
        ]
        self.domain = "machine learning"

    def test_instruction_generation(self):
        instructions = self.manager.generate_instructions(
            self.test_conversation,
            self.domain
        )
        self.assertIsInstance(instructions, str)
        self.assertGreater(len(instructions), 0)
        self.assertIn(self.domain, instructions)

    def test_template_selection(self):
        # Test exploratory template for new conversation
        short_conversation = self.test_conversation[:1]
        instructions = self.manager.generate_instructions(
            short_conversation,
            self.domain
        )
        self.assertIn("exploring", instructions.lower())

        # Test structured template for low coherence
        context = self.manager.context_analyzer.analyze(self.test_conversation)
        template = self.manager._select_template(context)
        self.assertIsInstance(template, str)
        self.assertGreater(len(template), 0)

    def test_template_customization(self):
        context = self.manager.context_analyzer.analyze(self.test_conversation)
        template = self.manager._select_template(context)
        customized = self.manager._customize_template(
            template,
            context,
            self.domain
        )
        self.assertIsInstance(customized, str)
        self.assertGreater(len(customized), len(template))
        self.assertIn(self.domain, customized)

    def test_empty_conversation(self):
        empty_conversation = []
        instructions = self.manager.generate_instructions(
            empty_conversation,
            self.domain
        )
        self.assertIsInstance(instructions, str)
        self.assertGreater(len(instructions), 0)
        self.assertIn("exploring", instructions.lower())

    def test_high_complexity_conversation(self):
        # Simulate a complex conversation with technical terms
        complex_conversation = [
            {
                "role": "user",
                "content": "Explain the relationship between bias-variance tradeoff and model complexity in machine learning"
            },
            {
                "role": "assistant",
                "content": "The bias-variance tradeoff involves balancing underfitting and overfitting. Higher model complexity typically reduces bias but increases variance, while simpler models have higher bias but lower variance."
            }
        ]
        instructions = self.manager.generate_instructions(
            complex_conversation,
            self.domain
        )
        self.assertIn("systematic", instructions.lower())

    def test_uncertainty_handling(self):
        uncertain_conversation = [
            {
                "role": "user",
                "content": "I'm not sure about the difference between supervised and unsupervised learning"
            },
            {
                "role": "assistant",
                "content": "Let me explain the key differences..."
            }
        ]
        instructions = self.manager.generate_instructions(
            uncertain_conversation,
            self.domain
        )
        self.assertIn("clarification", instructions.lower())

    def test_domain_integration(self):
        domains = ["machine learning", "quantum computing", "cybersecurity"]
        for domain in domains:
            instructions = self.manager.generate_instructions(
                self.test_conversation,
                domain
            )
            self.assertIn(domain, instructions)

if __name__ == '__main__':
    unittest.main()