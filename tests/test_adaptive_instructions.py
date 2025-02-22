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

    def test_mode_initialization(self):
        # Test default mode
        default_manager = AdaptiveInstructionManager()
        self.assertEqual(default_manager.mode, "human-ai")
        
        # Test AI-AI mode
        ai_manager = AdaptiveInstructionManager(mode="ai-ai")
        self.assertEqual(ai_manager.mode, "ai-ai")

    def test_ai_ai_mode_templates(self):
        manager = AdaptiveInstructionManager(mode="ai-ai")
        
        # Test AI-AI exploratory template
        short_conversation = self.test_conversation[:1]
        instructions = manager.generate_instructions(
            short_conversation,
            self.domain
        )
        self.assertIn("AI system", instructions)
        self.assertIn("structured knowledge", instructions.lower())
        
        # Test template selection in AI-AI mode
        context = manager.context_analyzer.analyze(self.test_conversation)
        template = manager._select_template(context, "ai-ai")
        self.assertIn("AI system", template)

    def test_mode_specific_customization(self):
        # Test human-ai mode
        human_manager = AdaptiveInstructionManager(mode="human-ai")
        human_instructions = human_manager.generate_instructions(
            self.test_conversation,
            self.domain
        )
        self.assertIn("human expert", human_instructions.lower())
        
        # Test ai-ai mode
        ai_manager = AdaptiveInstructionManager(mode="ai-ai")
        ai_instructions = ai_manager.generate_instructions(
            self.test_conversation,
            self.domain
        )
        self.assertIn("AI system", ai_instructions)
        self.assertNotIn("human expert", ai_instructions.lower())
        
        # Verify different content between modes

    def test_mode_aware_context_analysis(self):
        # Test AI-AI mode context analysis
        ai_conversation = [
            {
                "role": "user",
                "content": "Let us systematically analyze the formal properties of neural networks. Given the axioms of backpropagation..."
            },
            {
                "role": "assistant",
                "content": "Proceeding with formal analysis. Let's define our theorem precisely: neural networks with non-linear activation functions are universal approximators..."
            }
        ]
        
        # Create analyzers for both modes
        human_analyzer = ContextAnalyzer(mode="human-ai")
        ai_analyzer = ContextAnalyzer(mode="ai-ai")
        
        # Analyze same conversation with both modes
        human_context = human_analyzer.analyze(ai_conversation)
        ai_context = ai_analyzer.analyze(ai_conversation)
        
        # Verify AI-AI specific patterns are detected
        self.assertIn('formal_logic', ai_context.reasoning_patterns)
        self.assertIn('systematic', ai_context.reasoning_patterns)
        self.assertIn('technical', ai_context.reasoning_patterns)
        
        # Verify AI-AI mode has higher formal reasoning scores
        self.assertGreater(
            ai_context.reasoning_patterns.get('formal_logic', 0),
            human_context.reasoning_patterns.get('deductive', 0)
        )


        self.assertNotEqual(human_instructions, ai_instructions)

if __name__ == '__main__':
    unittest.main()