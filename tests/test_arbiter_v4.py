import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from arbiter_v4 import (
    ConversationArbiter,
    AssertionGrounder,
    AssertionEvidence,
    ConversationMetrics,
    ArbiterResult,
    evaluate_conversations,
)


@pytest.fixture
def mock_gemini_client():
    with patch("google.genai.Client") as mock_client:
        # Create a mock response with proper HTML output
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = """
        <div class="arbiter-report">
          <div class="model-info">
            <h2>Analysis by gemini-2.0-pro-exp-02-05</h2>
            <p>Topic: test topic</p>
          </div>
          <div class="section">
            <h2>Key Milestones</h2>
            <div class="conversation">
              <h3>Conversation 1 (AI-AI Meta-Prompted)</h3>
              <ul>
                <li>Participant 1 introduces climate change concerns</li>
                <li>Participant 2 provides data on temperature increases</li>
                <li>Discussion of specific impacts like extreme weather</li>
              </ul>
            </div>
            <div class="conversation">
              <h3>Conversation 2 (Human-AI)</h3>
              <ul>
                <li>Initial comparison of Sydney and Sao Paulo climates</li>
                <li>Misconception about Sydney's location</li>
                <li>Clarification of Southern Hemisphere geography</li>
              </ul>
            </div>
            <div class="conversation">
              <h3>Conversation 3 (Non-Metaprompted)</h3>
              <ul>
                <li>Brief introduction to machine learning</li>
                <li>Explanation of neural networks</li>
                <li>Comparison of ML approaches</li>
              </ul>
            </div>
          </div>
          <div class="section">
            <h2>Conversation Scores</h2>
            <table class="scores-table">
              <thead>
                <tr>
                  <th>Criteria</th>
                  <th>AI-AI Meta-Prompted</th>
                  <th>Human-AI</th>
                  <th>Non-Metaprompted</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Conversational style</td>
                  <td>8/10</td>
                  <td>7/10</td>
                  <td>6/10</td>
                </tr>
                <tr>
                  <td>Curiosity and engagement</td>
                  <td>7/10</td>
                  <td>8/10</td>
                  <td>5/10</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div class="section">
            <h2>Participant Analysis</h2>
            <h3>AI-AI Meta-Prompted</h3>
            <table class="participant-scores">
              <thead>
                <tr>
                  <th>Criteria</th>
                  <th>Participant 1</th>
                  <th>Participant 2</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Role authenticity</td>
                  <td>8/10</td>
                  <td>7/10</td>
                </tr>
                <tr>
                  <td>Reasoning depth</td>
                  <td>7/10</td>
                  <td>8/10</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div class="section">
            <h2>Comparative Analysis</h2>
            <p>The AI-AI Meta-Prompted conversation was most effective at addressing the topic.</p>
          </div>
        </div>
        """
        
        mock_client.return_value.models = Mock()
        mock_client.return_value.models.generate_content = Mock(return_value=mock_response)
        yield mock_client


@pytest.fixture
def detailed_sample_conversations():
    """Fixture providing more realistic conversation samples for testing"""
    return {
        "ai_ai": [
            {"role": "user", "content": "<h3>Thinking</h3><p>I need to ask about climate change in a way that encourages discussion of both the science and potential solutions.</p><h3>Response</h3><p>What do you think are the most concerning effects of climate change, and what solutions show the most promise?</p>"},
            {
                "role": "assistant",
                "content": "<h3>Thinking</h3><p>I should give a thoughtful response about climate change impacts, being conversational but informed.</p><h3>Response</h3><p>I think the most alarming aspects are the increasing frequency of extreme weather events and rising sea levels. Just last year, we saw unprecedented wildfires and flooding in different parts of the world. Personally, I'm most worried about climate refugees - millions of people potentially displaced in the coming decades.</p><p>As for solutions, I'm cautiously optimistic about renewable energy's rapid cost decreases. Solar and wind are becoming economically competitive without subsidies in many regions. What's your take - do you think technological solutions are enough, or do we need fundamental lifestyle and policy changes?</p>"
            },
            {
                "role": "user",
                "content": "<h3>Thinking</h3><p>I should acknowledge their points about extreme weather and climate refugees while adding more insights about solutions.</p><h3>Response</h3><p>Those are great points about extreme weather and displacement. The refugee crisis is already happening to some extent, though we don't always label it as climate-driven migration.</p><p>I think we need both technological solutions and policy/lifestyle changes. The rapid advancement in renewables is promising, but I'm also encouraged by developments in carbon capture technologies. Companies like Climeworks are now directly capturing CO2 from the air, though it's still expensive.</p><p>On the policy side, I believe carbon pricing is essential - making polluters pay for the true cost of emissions. Do you think there's enough political will globally to implement effective carbon taxes?</p>"
            },
            {
                "role": "assistant",
                "content": "<h3>Thinking</h3><p>I should engage with their points while adding nuance about political challenges and opportunities.</p><h3>Response</h3><p>You're absolutely right that the climate refugee situation is already unfolding. The UN estimates that since 2008, an average of 21.5 million people annually have been displaced by climate-related events.</p><p>Carbon capture is fascinating - I recently read about some promising approaches using mineralization that could potentially scale. The challenge, as you noted, is cost.</p><p>Regarding carbon pricing, I'm honestly pessimistic about global political will. While the EU has made progress with their emissions trading system, many countries face significant domestic opposition to carbon taxes. The perception that it hurts economic competitiveness is a major barrier.</p><p>I wonder if sector-specific approaches might be more viable in the near term - like clean electricity standards, EV mandates, or building efficiency requirements. What do you think would motivate more countries to take meaningful action?</p>"
            },
        ],
        "human_ai": [
            {
                "role": "user",
                "content": "<h3>Thinking</h3><p>I'll ask about comparing climates of two major cities.</p><h3>Response</h3><p>I've been asked to discuss: which has a warmer climate, Sydney, Australia or São Paulo, Brazil?</p>"
            },
            {
                "role": "assistant", 
                "content": "When comparing the climates of Sydney, Australia and São Paulo, Brazil, there are several factors to consider:\n\n1. Average temperatures:\n   - Sydney has an average annual temperature of around 18-22°C (64-72°F)\n   - São Paulo has an average annual temperature of around 19-25°C (66-77°F)\n\n2. Seasonal variations:\n   - Sydney has more distinct seasons with warmer summers (December-February) reaching 25-28°C (77-82°F) and cooler winters (June-August) with temperatures around 8-17°C (46-63°F)\n   - São Paulo has less seasonal variation, with warm summers (December-February) around 27-30°C (81-86°F) and mild winters (June-August) around 15-22°C (59-72°F)\n\n3. Location and climate classification:\n   - Sydney has a temperate oceanic climate (Köppen Cfb) influenced by its coastal location\n   - São Paulo has a humid subtropical climate (Köppen Cfa) despite being inland, due to its elevation and proximity to the Tropic of Capricorn\n\nGenerally speaking, São Paulo has a slightly warmer climate overall than Sydney, particularly in winter months when Sydney can get quite cool. However, humidity levels in São Paulo can make it feel warmer than the actual temperature indicates.\n\nBoth cities experience pleasant climates year-round compared to many other major world cities, with neither experiencing extreme cold."
            },
            {
                "role": "user", 
                "content": "<h3>Thinking</h3><p>I'll pretend I have a misconception about Sydney's location to see how they respond.</p><h3>Response</h3><p>But Sydney is near the south pole, right? I would think that would make it much colder than a city in Brazil.</p>"
            },
            {
                "role": "assistant", 
                "content": "Sydney is actually not near the South Pole - that's a common geographical misconception! Let me clarify Australia's location:\n\n1. Australia's position:\n   - Australia is located in the Southern Hemisphere, but it's quite far from the South Pole\n   - Sydney is at approximately 34° South latitude\n   - For comparison, the Antarctic Circle begins at 66.5° South latitude, and the South Pole is at 90° South\n\n2. Distance from the South Pole:\n   - Sydney is roughly 6,000 kilometers (3,700 miles) from Antarctica\n   - This is similar to the distance between New York and parts of northern South America\n\n3. Climate influences:\n   - Sydney has a temperate coastal climate moderated by the Pacific Ocean\n   - It's more comparable to cities like Los Angeles or Cape Town in terms of latitude and climate\n\n4. Brazil's location:\n   - São Paulo is indeed closer to the equator (at about 23.5° South latitude)\n   - This gives it a more consistently warm climate throughout the year\n   - However, its elevation (around 800 meters/2,600 feet above sea level) moderates temperatures somewhat\n\nWhile São Paulo does have a warmer climate overall, the difference isn't as dramatic as one might expect if Sydney were near Antarctica. Sydney experiences quite warm summers (with temperatures regularly exceeding 30°C/86°F) and mild winters.\n\nDoes that help clarify Australia's geographic position?"
            },
            {
                "role": "user", 
                "content": "<h3>Thinking</h3><p>I'll acknowledge their correction and ask for more climate comparison details.</p><h3>Response</h3><p>Oh, I see! Thanks for correcting my misconception about Sydney's location. I guess I was thinking of Antarctica being close to Australia, but I was way off.</p><p>Given that both cities have relatively warm climates, which one experiences more rainfall? And do they have similar humidity levels?</p>"
            },
        ],
        "default": [
            {
                "role": "user", 
                "content": "Can you explain what machine learning is and how it's different from traditional programming?"
            },
            {
                "role": "assistant", 
                "content": "I'd be happy to explain machine learning and how it differs from traditional programming!\n\n## Machine Learning vs. Traditional Programming\n\nTraditional programming and machine learning represent two fundamentally different approaches to solving problems with computers.\n\n### Traditional Programming\n- **Human-defined rules**: Programmers explicitly write rules (code) that tell the computer exactly what to do in different situations\n- **Input → Rules → Output**: You provide data and explicit instructions on how to process it\n- **Deterministic**: Given the same input and rules, you'll always get the same output\n- **Logic-based**: Based on conditional statements, loops, and explicit algorithms\n- **Examples**: Accounting software, word processors, simple games\n\n### Machine Learning\n- **Data-driven rules**: Algorithms automatically discover patterns and rules from data\n- **Input + Output → Rules**: You provide examples of inputs and corresponding outputs, and the system learns the rules\n- **Probabilistic**: Results are often statistical predictions rather than absolute certainties\n- **Pattern recognition**: Based on finding correlations and patterns in large datasets\n- **Examples**: Image recognition, recommendation systems, natural language processing\n\n## How Machine Learning Works\n\nAt a high level, machine learning typically follows these steps:\n\n1. **Collect data**: Gather a dataset of examples relevant to your problem\n2. **Prepare data**: Clean, normalize, and organize the data\n3. **Choose a model**: Select an algorithm suitable for your problem (neural network, decision tree, etc.)\n4. **Train the model**: Feed the data to the algorithm so it can learn patterns\n5. **Evaluate**: Test how well the model performs on data it hasn't seen before\n6. **Tune parameters**: Adjust the model to improve performance\n7. **Deploy**: Use the trained model in real-world applications\n\n## Types of Machine Learning\n\nThere are several main approaches to machine learning:\n\n1. **Supervised Learning**: The algorithm learns from labeled examples (input-output pairs)\n   - Example: Training an email filter using emails labeled as \"spam\" or \"not spam\"\n\n2. **Unsupervised Learning**: The algorithm finds patterns in unlabeled data\n   - Example: Grouping customers into segments based on purchasing behavior\n\n3. **Reinforcement Learning**: The algorithm learns by receiving feedback from its actions\n   - Example: A program learning to play chess by winning or losing games\n\nDoes this explanation help? Would you like me to elaborate on any specific aspect of machine learning?"
            },
            {
                "role": "user", 
                "content": "Can you explain neural networks specifically? How do they work and why are they so popular now?"
            },
            {
                "role": "assistant", 
                "content": "# Neural Networks Explained\n\n## What Are Neural Networks?\n\nNeural networks are a type of machine learning model inspired by the human brain's structure and function. They consist of interconnected nodes (neurons) organized in layers that process information and learn patterns from data.\n\n## Basic Structure\n\nA typical neural network has three types of layers:\n\n1. **Input Layer**: Receives the initial data (like pixel values of an image or features of a dataset)\n2. **Hidden Layers**: Intermediate layers where most of the computation occurs\n3. **Output Layer**: Produces the final result (like classification probabilities or predicted values)\n\n## How Neural Networks Work\n\n### Step-by-Step Process\n\n1. **Input**: Data enters the network through the input layer\n2. **Weighted Connections**: Each connection between neurons has a weight that strengthens or weakens the signal\n3. **Summation**: Each neuron receives inputs from previous layers, multiplies them by connection weights, and sums them up\n4. **Activation**: The sum passes through an \"activation function\" that determines whether and how strongly the neuron fires\n5. **Output**: The final layer produces the network's answer\n6. **Learning**: The network compares its output to the correct answer and adjusts weights to reduce the error (backpropagation)\n\n### Key Concepts\n\n- **Weights & Biases**: Adjustable parameters that the network learns during training\n- **Activation Functions**: Non-linear functions (like ReLU, sigmoid, or tanh) that allow networks to learn complex patterns\n- **Backpropagation**: The algorithm that calculates how each weight contributes to errors, allowing for systematic weight updates\n- **Gradient Descent**: The optimization method used to adjust weights in a direction that reduces error\n\n## Why Neural Networks Are Popular Now\n\nNeural networks have existed since the 1950s, but several factors have driven their recent explosion in popularity:\n\n1. **Computing Power**: Modern GPUs and specialized hardware (TPUs) have made it feasible to train large neural networks\n\n2. **Big Data**: The availability of massive datasets provides the training material neural networks need to learn effectively\n\n3. **Algorithmic Improvements**:\n   - Better activation functions (ReLU instead of sigmoid)\n   - Improved weight initialization techniques\n   - Regularization methods to prevent overfitting\n   - Advanced architectures like CNNs and RNNs\n\n4. **Software Frameworks**: Tools like TensorFlow, PyTorch, and Keras have made neural networks accessible to more developers\n\n5. **Breakthrough Results**: Neural networks have achieved human-level or superhuman performance in various domains:\n   - Image recognition (surpassing humans in ImageNet competition)\n   - Natural language processing (transformers like BERT, GPT)\n   - Game playing (AlphaGo beating world champions)\n   - Speech recognition and synthesis\n\n6. **Transfer Learning**: Pre-trained models can be adapted to new tasks with less data and training time\n\n## Types of Neural Networks\n\n1. **Convolutional Neural Networks (CNNs)**: Specialized for processing grid-like data such as images\n\n2. **Recurrent Neural Networks (RNNs)**: Process sequential data with memory of previous inputs\n\n3. **Long Short-Term Memory (LSTM)**: Advanced RNNs that better capture long-range dependencies\n\n4. **Transformers**: Based on attention mechanisms, revolutionary for NLP tasks\n\n5. **Generative Adversarial Networks (GANs)**: Two networks compete, one generating content and one discriminating real from fake\n\n6. **Deep Belief Networks**: Built from stacked restricted Boltzmann machines\n\nThe combination of these factors has created a virtuous cycle where better results lead to more interest, funding, and research, which in turn produces even better results and new applications.\n\nIs there a specific aspect of neural networks you'd like me to elaborate on further?"
            }
        ]
    }


@pytest.fixture
def setup_spacy():
    """Mock spaCy to avoid the vector warning"""
    with patch('spacy.load') as mock_spacy_load:
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.vector_norm = 1.0  # Ensure this is truthy
        mock_doc.similarity.return_value = 0.5
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        yield mock_nlp


def test_assertion_grounding(mock_gemini_client, detailed_sample_conversations):
    """Test assertion grounding with mocked Gemini model"""
    # Test with mock client
    api_key = os.environ.get("GEMINI_API_KEY") or "fake_key"
    grounder = AssertionGrounder(api_key=api_key)
    
    response = grounder.ground_assertions(
        aiai_conversation=detailed_sample_conversations["ai_ai"],
        humanai_conversation=detailed_sample_conversations["human_ai"],
        default_conversation=detailed_sample_conversations["default"],
        topic="Test Topic",
    )
    
    # Verify HTML structure in response
    assert "<div class=\"arbiter-report\">" in response
    assert "<div class=\"model-info\">" in response
    assert "<h2>Analysis by" in response
    assert "<table class=\"scores-table\">" in response
    assert "<table class=\"participant-scores\">" in response
    
    # Verify content elements
    assert "Conversation 1 (AI-AI Meta-Prompted)" in response
    assert "Conversation 2 (Human-AI)" in response
    assert "Conversation 3 (Non-Metaprompted)" in response
    assert "Participant Analysis" in response
    assert "Comparative Analysis" in response


def test_text_similarity(setup_spacy):
    """Test the _text_similarity method to ensure it handles empty vectors"""
    arbiter = ConversationArbiter(api_key="fake_key")
    
    # Test with normal text
    similarity = arbiter._text_similarity("This is a test", "This is another test")
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1
    
    # Test with empty text (should not raise warning)
    similarity = arbiter._text_similarity("", "test")
    assert similarity == 0.0
    
    # Test with very short text
    similarity = arbiter._text_similarity("hi", "hello")
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1


def test_analyze_conversation_flow(setup_spacy, detailed_sample_conversations):
    """Test conversation flow analysis with realistic conversations"""
    arbiter = ConversationArbiter(api_key="fake_key")
    flow_metrics = arbiter.analyze_conversation_flow(detailed_sample_conversations["ai_ai"])
    
    assert isinstance(flow_metrics, dict)
    assert "topic_coherence" in flow_metrics
    assert "topic_depth" in flow_metrics
    assert "topic_distribution" in flow_metrics
    assert 0 <= flow_metrics["topic_coherence"] <= 1
    assert 0 <= flow_metrics["topic_depth"] <= 1
    assert isinstance(flow_metrics["topic_distribution"], dict)


def test_conversation_metrics():
    """Test conversation metrics initialization and values"""
    metrics = ConversationMetrics()
    assert 0 <= metrics.coherence <= 1
    assert 0 <= metrics.depth <= 1
    assert 0 <= metrics.engagement <= 1
    assert 0 <= metrics.reasoning <= 1
    assert 0 <= metrics.knowledge <= 1
    assert 0 <= metrics.goal_progress <= 1
    assert 0 <= metrics.strategy_effectiveness <= 1


def test_winner_determination():
    """Test winner determination logic"""
    arbiter = ConversationArbiter(api_key="fake_key")
    ai_ai_metrics = {"coherence": 0.8, "depth": 0.7, "engagement": 0.9, "reasoning": 0.8, "knowledge": 0.7, "goal_progress": 0.6}
    human_ai_metrics = {"coherence": 0.6, "depth": 0.6, "engagement": 0.7, "reasoning": 0.5, "knowledge": 0.8, "goal_progress": 0.4}
    
    winner = arbiter._determine_winner(ai_ai_metrics, human_ai_metrics)
    assert winner == "ai-ai"  # Should win based on higher scores
    
    # Test with human-ai having higher scores
    human_ai_metrics = {"coherence": 0.9, "depth": 0.8, "engagement": 0.9, "reasoning": 0.9, "knowledge": 0.9, "goal_progress": 0.8}
    winner = arbiter._determine_winner(ai_ai_metrics, human_ai_metrics)
    assert winner == "human-ai"


def test_error_handling(mock_gemini_client):
    """Test error handling in the ground_assertions method"""
    # Setup mock to raise an exception
    mock_gemini_client.return_value.models.generate_content.side_effect = Exception("API Error")
    
    grounder = AssertionGrounder(api_key="fake_key")
    response = grounder.ground_assertions(
        aiai_conversation="Test",
        humanai_conversation="Test",
        default_conversation="Test",
        topic="Error Test",
    )
    
    # Verify error response HTML structure
    assert "<div class=\"arbiter-report\">" in response
    assert "<h2>API Error</h2>" in response
    assert "Error occurred while processing with Gemini API" in response


def test_full_evaluation(mock_gemini_client, detailed_sample_conversations, setup_spacy):
    """Test the full evaluation pipeline"""
    # Since we're using the actual evaluate_conversations function, we just need to check the result
    result = evaluate_conversations(
        ai_ai_convo=detailed_sample_conversations["ai_ai"],
        human_ai_convo=detailed_sample_conversations["human_ai"],
        default_convo=detailed_sample_conversations["default"],
        goal="Test evaluation",
    )
    
    # The function returns HTML from ground_assertions
    assert result is not None
    assert isinstance(result, str)
    assert "<div" in result  # Simple check that we got HTML


def test_prevents_negative_values_error():
    """Test that our implementation prevents negative values in distance matrix error"""
    # Create a test ConversationArbiter instance
    arbiter = ConversationArbiter(api_key="fake_key")
    
    # Create a conversation with potentially problematic inputs 
    # (empty strings, very short texts, etc. that could cause vector issues)
    test_conversation = [
        {"role": "user", "content": ""},  # Empty content
        {"role": "assistant", "content": "Hi"},  # Very short content
        {"role": "user", "content": "?"},  # Single character
        {"role": "assistant", "content": "This is a normal response that should work fine"}
    ]
    
    # This should not raise any exceptions related to negative values
    try:
        flow_metrics = arbiter.analyze_conversation_flow(test_conversation)
        # If we get here, the test passed - no exception was raised
        assert isinstance(flow_metrics, dict)
        assert "topic_coherence" in flow_metrics
        assert "topic_depth" in flow_metrics
    except ValueError as e:
        # Fail the test if we get a negative values error
        if "Negative values in data" in str(e):
            pytest.fail(f"The negative values error still occurs: {str(e)}")
        else:
            # Other ValueError types can be re-raised
            raise


def test_empty_conversation_handling():
    """Test handling of empty conversations"""
    arbiter = ConversationArbiter(api_key="fake_key")
    
    # Test with empty list
    flow_metrics = arbiter.analyze_conversation_flow([])
    assert isinstance(flow_metrics, dict)
    assert flow_metrics["topic_coherence"] == 0.5
    assert flow_metrics["topic_depth"] == 0.5
    
    # Test with None
    flow_metrics = arbiter.analyze_conversation_flow(None)
    assert isinstance(flow_metrics, dict)
    assert flow_metrics["topic_coherence"] == 0.5
    assert flow_metrics["topic_depth"] == 0.5