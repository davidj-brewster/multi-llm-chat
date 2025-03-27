import pytest
import json
import os
from unittest.mock import Mock, patch
from datetime import datetime
from pathlib import Path

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
        # Mock search results
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].grounding_metadata = Mock()
        mock_response.candidates[0].grounding_metadata.search_entry_point = Mock()
        mock_response.candidates[
            0
        ].grounding_metadata.search_entry_point.rendered_content = [
            {
                "link": "https://example.edu/article1",
                "title": "Test Article",
                "snippet": "Relevant content for testing",
            }
        ]
        mock_client.return_value.generate_content.return_value = mock_response
        yield mock_client


@pytest.fixture
def sample_conversations():
    return {
        "ai_ai": [
            {"role": "user", "content": "What are the effects of climate change?"},
            {
                "role": "assistant",
                "content": "Climate change leads to rising temperatures.",
            },
            {
                "role": "user",
                "content": "Numerous studies showing temperature increases globally over the last 100 years",
            },
            {
                "role": "assistant",
                "content": "Yes, that's correct. In fact there has been a global 1 degree Celsius increase in temperature over the last century.",
            },
            {
                "role": "user",
                "content": "What are the implications of this temperature increase?",
            },
            {
                "role": "assistant",
                "content": "The implications of this temperature increase are far-reaching, including more frequent and severe weather events.",
            },
            {
                "role": "user",
                "content": "What are some examples of these weather events?",
            },
            {
                "role": "assistant",
                "content": "Some examples of severe weather events include hurricanes, droughts, and heatwaves.",
            },
        ],
        "human_ai": [
            {
                "role": "user",
                "content": "I've been asked to discuss: which is a warmer climate? Sydney, Australia or Sao Paolo, Brazil?",
            },
            {
                "role": "assistant",
                "content": "Of course this depends on seasonality and other factors, but generally Sydney has a warmer climate.",
            },
            {"role": "user", "content": "But Sydney is near the south pole?"},
        ],
    }


def test_assertion_grounding(mock_gemini_client):
    """Test assertion grounding with mocked search results"""
    api_key = os.environ.get("GEMINI_API_KEY")
    grounder = AssertionGrounder(api_key=api_key)
    evidence = grounder.ground_assertions(
        grounder,
        ai_ai_convo=sample_conversations["ai_ai"],
        humanai_convo=sample_conversations["human_ai"],
    )

    assert isinstance(evidence, AssertionEvidence)
    assert evidence.confidence > 0
    assert len(evidence.sources) > 0
    assert evidence.verification_method == "gemini_search"


def test_conversation_analysis(mock_gemini_client, sample_conversations):
    """Test conversation analysis and evaluation"""
    arbiter = ConversationArbiter(api_key="fake_key")
    result = evaluate_conversations(
        ai_ai_convo=str(sample_conversations["ai_ai"]),
        human_ai_convo=str(sample_conversations["human_ai"]),
        goal="world knowledge",
    )

    assert isinstance(result, ArbiterResult)
    assert result.winner in ["ai-ai", "human-ai"]
    assert "ai-ai" in result.conversation_metrics
    assert "human-ai" in result.conversation_metrics
    assert len(result.key_insights) >= 0
    assert isinstance(result.execution_timestamp, str)


def test_conversation_metrics():
    """Test conversation metrics initialization and values"""
    metrics = ConversationMetrics()
    assert 0 <= metrics.coherence <= 1
    assert 0 <= metrics.depth <= 1
    assert 0 <= metrics.engagement <= 1
    assert hasattr(metrics, "goal_progress")


@pytest.fixture
async def test_flow_analysis(sample_conversations):
    """Test conversation flow analysis"""
    arbiter = ConversationArbiter(api_key="fake_key")
    flow_metrics = arbiter.analyze_conversation_flow(sample_conversations["ai_ai"])

    assert isinstance(flow_metrics, dict)
    assert "topic_coherence" in flow_metrics
    assert "topic_depth" in flow_metrics
    assert 0 <= flow_metrics["topic_coherence"] <= 1


def test_winner_determination():
    """Test winner determination logic"""
    arbiter = ConversationArbiter(api_key="fake_key")
    ai_ai_metrics = {"coherence": 0.8, "depth": 0.7, "engagement": 0.9}
    human_ai_metrics = {"coherence": 0.6, "depth": 0.6, "engagement": 0.7}

    winner = arbiter._determine_winner(ai_ai_metrics, human_ai_metrics)
    assert winner in ["ai-ai", "human-ai"]
