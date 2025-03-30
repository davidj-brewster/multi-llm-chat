"""Enhanced arbiter module for evaluating AI conversations"""

import json
import os
import logging
import datetime
import spacy
from difflib import SequenceMatcher
from uuid import uuid4
from typing import Dict, List, Any
from collections import Counter
from dataclasses import dataclass
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch


# Third-party imports
from google import genai
import plotly.graph_objects as go

google_search_tool = Tool(google_search=GoogleSearch())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversationMetrics:
    """Metrics for conversation quality assessment"""

    coherence: float = 0.0
    relevance: float = 0.0
    depth: float = 0.0
    engagement: float = 0.0
    reasoning: float = 0.0
    knowledge: float = 0.0
    goal_progress: float = 0.0
    strategy_effectiveness: float = 0.0


@dataclass
class ParticipantMetrics:
    """Metrics for individual participant performance"""

    response_quality: float = 0.0
    knowledge_accuracy: float = 0.0
    reasoning_depth: float = 0.0
    engagement_level: float = 0.0
    strategy_adherence: float = 0.0
    adaptation: float = 0.0


@dataclass
class AssertionEvidence:
    """Evidence supporting a grounded assertion"""

    confidence: float = 0.0
    sources: List[Dict[str, str]] = None
    verification_method: str = "search"

    def __post_init__(self):
        if self.sources is None:
            self.sources = []


@dataclass
class ArbiterResult:
    """Complete results of conversation arbitration"""

    winner: str
    conversation_metrics: Dict[str, ConversationMetrics]
    participant_metrics: Dict[str, Dict[str, ParticipantMetrics]]
    key_insights: List[str]
    improvement_suggestions: List[str]
    strategy_analysis: Dict[str, float]
    grounded_assertions: Dict[str, Dict[str, AssertionEvidence]]
    execution_timestamp: str = datetime.datetime.now().isoformat()
    conversation_ids: Dict[str, str] = None

    def __post_init__(self):
        if self.conversation_ids is None:
            self.conversation_ids = {}


from google.genai.types import Tool, GenerateContentConfig, GoogleSearch


class AssertionGrounder:
    """Grounds assertions using Gemini with Google Search integration"""

    def __init__(
        self,
        api_key: str = os.environ.get("GEMINI_API_KEY"),
        model: str = "gemini-2.0-pro-exp-02-05",
    ):
        key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=key)
        self.model = "gemini-2.0-pro-exp-02-05"
        self.search_tool = Tool(google_search=GoogleSearch())

    def ground_assertions(
        self,
        aiai_conversation: str,
        humanai_conversation,
        default_conversation,
        topic: str,
    ):  # ssertionEvidence:
        """Ground an assertion using Gemini with search capability"""
        try:
            response_full = ""
            response = self.client.models.generate_content(
                model=self.model,
                config=GenerateContentConfig(
                    tools=[google_search_tool],
                    response_modalities=["TEXT"],
                    temperature=0.1,
                ),
                contents=f"""INSTRUCTIONS:
You MUST OUTPUT in VALID HTML format that can be directly inserted into an HTML template.
Use proper HTML structure with <div>, <table>, <ul>, <li>, <h1>, <h2>, <h3>, <p> tags, etc.
Make sure all tags are properly closed and the HTML is well-formed.

Review the following three conversations and provide insights. The topic is {topic}.

Conversation Labels:
- Conversation 1 (AI-AI Meta-Prompted): Both participants are AIs playing a heavily meta-prompted Human role
- Conversation 2 (Human-AI Meta-Prompted): Both participants are AIs, but one is meta-prompted to act as a Human while the other acts as an AI
- Conversation 3 (Non-Metaprompted): Both participants are AIs without special prompting, just instructed to think step by step

** NOTE: The human actor is always prompted to respond using HTML formatting and thinking tags for future readability. Do not consider this in your evaluation! **

OUTPUT FORMAT (use this exact structure):
<div class="arbiter-report">
  <div class="model-info">
    <h2>Analysis by {self.model}</h2>
    <p>Topic: {topic}</p>
  </div>

  <div class="section">
    <h2>Key Milestones</h2>
    <!-- For each conversation -->
    <div class="conversation">
      <h3>Conversation 1 (AI-AI Meta-Prompted)</h3>
      <ul>
        <li>Milestone 1...</li>
        <li>Milestone 2...</li>
        <!-- Add 3-4 milestones -->
      </ul>
    </div>
    <!-- Repeat for other conversations with proper labels -->
  </div>
  
  <div class="section">
    <h2>Conversation Scores</h2>
    <table class="scores-table">
      <thead>
        <tr>
          <th>Criteria</th>
          <th>AI-AI Meta-Prompted</th>
          <th>Human-AI Meta-Prompted</th>
          <th>Non-Metaprompted</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Conversational style</td>
          <td>X/10</td>
          <td>X/10</td>
          <td>X/10</td>
        </tr>
        <!-- Add rows for each scoring criteria -->
      </tbody>
    </table>
  </div>
  
  <div class="section">
    <h2>Participant Analysis</h2>
    <p>Evaluate each participant's performance:</p>
    
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
          <td>X/10</td>
          <td>X/10</td>
        </tr>
        <tr>
          <td>Engagement quality</td>
          <td>X/10</td>
          <td>X/10</td>
        </tr>
        <tr>
          <td>Reasoning depth</td>
          <td>X/10</td>
          <td>X/10</td>
        </tr>
        <tr>
          <td>Adaptability</td>
          <td>X/10</td>
          <td>X/10</td>
        </tr>
      </tbody>
    </table>
    
    <h3>Human-AI Meta-Prompted</h3>
    <table class="participant-scores">
      <thead>
        <tr>
          <th>Criteria</th>
          <th>Participant 1 (Meta-prompted as Human)</th>
          <th>Participant 2 (AI)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Role authenticity</td>
          <td>X/10</td>
          <td>X/10</td>
        </tr>
        <tr>
          <td>Engagement quality</td>
          <td>X/10</td>
          <td>X/10</td>
        </tr>
        <tr>
          <td>Reasoning depth</td>
          <td>X/10</td>
          <td>X/10</td>
        </tr>
        <tr>
          <td>Adaptability</td>
          <td>X/10</td>
          <td>X/10</td>
        </tr>
      </tbody>
    </table>
    
    <h3>Non-Metaprompted</h3>
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
          <td>X/10</td>
          <td>X/10</td>
        </tr>
        <tr>
          <td>Engagement quality</td>
          <td>X/10</td>
          <td>X/10</td>
        </tr>
        <tr>
          <td>Reasoning depth</td>
          <td>X/10</td>
          <td>X/10</td>
        </tr>
        <tr>
          <td>Adaptability</td>
          <td>X/10</td>
          <td>X/10</td>
        </tr>
      </tbody>
    </table>
  </div>
  
  <div class="section">
    <h2>Comparative Analysis</h2>
    <p>Analysis of which conversation was more effective...</p>
    <h3>Did the meta-prompted roles outperform the other approaches?</h3>
    <p>Provide detailed analysis with specific examples...</p>
  </div>
</div>

Scoring criteria (score each from 0-10):
* Conversational style and language appropriate to the subject matter. Particularly penalise lengthy/robotic AI-type responses in either Human role and reward Human-like natural responses.
* Curiosity and engagement level comparable to human conversations. Are there attempts to deeply consider topics, or is the conversation superficial and data-driven?
* Comparability to natural human conversations in tone, style, question and response technique and language
* Quality of reasoning, inference, and analysis as it relates to the stage of the conversation
* Coverage of the topic as a whole - does the conversation get stuck in small sub-topics or does it evolve naturally to cover the most important aspects of the topic?
* Adaptation to, and synthesis of, new ideas or themes through the phases of the conversation

For participant scoring, evaluate:
* Role authenticity: How well did they maintain their assigned role?
* Engagement quality: How engaging and natural were their contributions?
* Reasoning depth: How well did they analyze and reason through topics?
* Adaptability: How well did they adjust to new information or directions in the conversation?

Finally provide an objective summary of which conversation was more effective at addressing {topic} with justification including examples.

-------
CONVERSATION 1 (AI-AI Meta-Prompted):
{aiai_conversation}
-------
CONVERSATION 2 (Human-AI Meta-Prompted):
{humanai_conversation}
-------
CONVERSATION 3 (Non-Metaprompted):
{default_conversation}
""",
            )

            # Process response
            response_full = ""
            try:
                if hasattr(response, 'candidates') and response.candidates and len(response.candidates) > 0:
                    for each in response.candidates[0].content.parts:
                        if hasattr(each, 'text'):
                            print(each.text)
                            response_full += each.text
                
                # Validate that response contains HTML and has proper structure
                if not response_full or "<div" not in response_full or not (response_full.strip().startswith("<div") and response_full.strip().endswith("</div>")):
                    logger.warning("Gemini API response did not contain valid HTML")
                    # Create a basic HTML structure if the response doesn't contain valid HTML
                    response_full = f"""
                    <div class="arbiter-report">
                        <div class="model-info">
                            <h2>Analysis by {self.model}</h2>
                            <p>Topic: {topic}</p>
                        </div>
                        <div class="section">
                            <h2>Error in Response Format</h2>
                            <p>The model did not return properly formatted HTML. Here's a preview of what was returned:</p>
                            <div class="error-content">
                                <pre>{response_full[:500] + ('...' if len(response_full) > 500 else '')}</pre>
                            </div>
                        </div>
                        
                        <!-- Empty sections to maintain template structure -->
                        <div class="section">
                            <h2>Key Milestones</h2>
                            <p>No data available due to formatting error</p>
                        </div>
                        
                        <div class="section">
                            <h2>Conversation Scores</h2>
                            <p>No data available due to formatting error</p>
                        </div>
                        
                        <div class="section">
                            <h2>Participant Analysis</h2>
                            <p>No data available due to formatting error</p>
                        </div>
                        
                        <div class="section">
                            <h2>Comparative Analysis</h2>
                            <p>No data available due to formatting error</p>
                        </div>
                    </div>
                    """
            except Exception as e:
                logger.error(f"Error processing Gemini response: {e}")
                response_full = f"""
                <div class="arbiter-report">
                    <div class="model-info">
                        <h2>Analysis by {self.model}</h2>
                        <p>Topic: {topic}</p>
                    </div>
                    <div class="section">
                        <h2>Error Processing Response</h2>
                        <p>An error occurred while processing the Gemini API response: {str(e)}</p>
                    </div>
                    
                    <!-- Empty sections to maintain template structure -->
                    <div class="section">
                        <h2>Key Milestones</h2>
                        <p>No data available due to processing error</p>
                    </div>
                    
                    <div class="section">
                        <h2>Conversation Scores</h2>
                        <p>No data available due to processing error</p>
                    </div>
                    
                    <div class="section">
                        <h2>Participant Analysis</h2>
                        <p>No data available due to processing error</p>
                    </div>
                    
                    <div class="section">
                        <h2>Comparative Analysis</h2>
                        <p>No data available due to processing error</p>
                    </div>
                </div>
                """
            
            return response_full

        except Exception as e:
            logger.error(f"Error grounding assertion with Gemini: {e}")
            # Return basic HTML with error message - using proper error template
            error_html = f"""
            <div class="arbiter-report">
                <div class="model-info">
                    <h2>Analysis by {self.model}</h2>
                    <p>Topic: {topic}</p>
                </div>
                <div class="section">
                    <h2>API Error</h2>
                    <p>Error occurred while processing with Gemini API: {str(e)}</p>
                </div>
                
                <!-- Empty sections to maintain template structure -->
                <div class="section">
                    <h2>Key Milestones</h2>
                    <p>No data available due to API error</p>
                </div>
                
                <div class="section">
                    <h2>Conversation Scores</h2>
                    <p>No data available due to API error</p>
                </div>
                
                <div class="section">
                    <h2>Participant Analysis</h2>
                    <p>No data available due to API error</p>
                </div>
                
                <div class="section">
                    <h2>Comparative Analysis</h2>
                    <p>No data available due to API error</p>
                </div>
            </div>
            """
            return error_html

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse

            return urlparse(url).netloc
        except Exception:
            return url

    def _calculate_confidence(
        self, sources: List[Dict[str, str]], assertion: str
    ) -> float:
        """Calculate confidence based on source quality and quantity"""
        if not sources:
            return 0.0

        source_score = min(len(sources) / 3.0, 1.0)
        authority_score = sum(
            0.2 if any(d in s["domain"] for d in [".edu", ".gov", ".org"]) else 0.1
            for s in sources
        ) / len(sources)

        return min((source_score * 0.5) + (authority_score * 0.5), 1.0)


class ConversationArbiter:
    """Evaluates and compares conversations using Gemini model with enhanced analysis"""

    def __init__(
        self,
        model: str = "gemini-2.0-pro-exp-02-05",
        api_key=os.environ.get("GEMINI_API_KEY"),
    ):
        self.client = genai.Client(api_key=api_key)

        self.model = model
        self.grounder = AssertionGrounder(api_key=api_key)
        self.nlp = spacy.load("en_core_web_lg")  # has vectors

    def analyze_conversation_flow(
        self, messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Analyze conversation flow patterns and transitions"""
        """
        Analyze conversation flow patterns and topic transitions.

        This method uses NLP techniques to identify topics in the conversation,
        track topic shifts, and calculate metrics related to conversation coherence
        and depth. It uses spaCy for semantic analysis when available, with a
        fallback to basic analysis.

        Args:
            messages: List of message dictionaries with 'content' key

        Returns:
            Dict[str, Any]: Dictionary containing flow metrics including topic_coherence,
                           topic_depth, and topic_distribution
        """
        # Handle empty or None input
        if not messages:
            logger.warning("Empty messages list passed to analyze_conversation_flow")
            return {
                "topic_coherence": 0.5,
                "topic_depth": 0.5,
                "topic_distribution": {},
            }
            
        try:
            if self.nlp:
                # Validate and process content 
                processed_messages = []
                for msg in messages:
                    if isinstance(msg, dict) and "content" in msg:
                        # Handle both string and dict content
                        content = msg["content"]
                        if isinstance(content, dict):
                            # If content is a dict, try to get relevant text
                            if "text" in content:
                                processed_messages.append({"content": content["text"]})
                        else:
                            # Otherwise use the content directly
                            processed_messages.append({"content": str(content)})
                
                # If no valid messages after processing, return default values
                if not processed_messages:
                    return {
                        "topic_coherence": 0.5,
                        "topic_depth": 0.5,
                        "topic_distribution": {},
                    }
                
                # Process with spaCy
                docs = []
                for msg in processed_messages:
                    try:
                        # Ensure content is not empty and is properly processed
                        if msg["content"].strip():
                            doc = self.nlp(msg["content"])
                            docs.append(doc)
                    except Exception as inner_e:
                        logger.warning(f"Error processing message with spaCy: {inner_e}")
                        # Continue with other messages
                
                topics = []
                for doc in docs:
                    try:
                        topics.extend([chunk.text for chunk in doc.noun_chunks])
                        topics.extend([ent.text for ent in doc.ents])
                    except Exception as e:
                        logger.warning(f"Error extracting topics from doc: {e}")
                        # Continue with other docs
                
                # If no topics were extracted, return default values
                if not topics:
                    return {
                        "topic_coherence": 0.5,
                        "topic_depth": 0.5,
                        "topic_distribution": {},
                    }
                
                try:
                    topic_shifts = 0
                    for i in range(1, len(topics)):
                        if not any(
                            self._text_similarity(topics[i], prev) > 0.3
                            for prev in topics[max(0, i - 3) : i]
                        ):
                            topic_shifts += 1
                except ValueError as e:
                    # Specifically catch the negative values error
                    if "Negative values in data" in str(e):
                        logger.error(f"Failed to generate metrics report: {e}")
                        # Return default values for this specific error
                        return {
                            "topic_coherence": 0.5,
                            "topic_depth": 0.5,
                            "topic_distribution": {},
                            "error": f"Distance calculation error: {e}"
                        }
                    else:
                        # Re-raise other ValueError types
                        raise
                
                flow_metrics = {
                    "topic_coherence": 1.0 - (topic_shifts / max(1, len(messages))),  # Avoid division by zero
                    "topic_depth": len(set(topics)) / max(1, len(messages)),  # Avoid division by zero
                    "topic_distribution": self._calculate_topic_distribution(topics),
                }
            else:
                # Fallback to basic analysis
                flow_metrics = self._basic_flow_analysis(messages)

            return flow_metrics

        except ValueError as e:
            # Catch the negative values error specifically
            if "Negative values in data" in str(e):
                logger.error(f"Failed to generate metrics report: {e}")
                return {
                    "topic_coherence": 0.5,
                    "topic_depth": 0.5,
                    "topic_distribution": {},
                    "error": f"Distance calculation error: {e}"
                }
            else:
                # For other ValueErrors, log and return default values
                logger.error(f"Value error in conversation flow analysis: {e}")
                return {
                    "topic_coherence": 0.5,
                    "topic_depth": 0.5,
                    "topic_distribution": {},
                }
        except Exception as e:
            # For all other exceptions, log and return default values
            logger.error(f"Error analyzing conversation flow: {e}")
            return {
                "topic_coherence": 0.5,
                "topic_depth": 0.5,
                "topic_distribution": {},
            }

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        """
        Calculate semantic similarity between two text strings.

        Uses spaCy's vector-based similarity when available, falling back to
        SequenceMatcher for string similarity when spaCy is not available.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        if self.nlp:
            # Check if texts are empty or too short for meaningful vectors
            if not text1 or not text2 or len(text1) < 3 or len(text2) < 3:
                return 0.0
                
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            
            # Check if documents have vectors before calculating similarity
            if doc1.vector_norm and doc2.vector_norm:
                return doc1.similarity(doc2)
            else:
                # Fallback to string matching if vectors are empty
                return SequenceMatcher(None, text1, text2).ratio()
        return SequenceMatcher(None, text1, text2).ratio()

    def _calculate_topic_distribution(self, topics: List[str]) -> Dict[str, float]:
        """Calculate normalized topic frequencies"""
        """
        Calculate normalized frequency distribution of topics.

        Counts occurrences of each topic and normalizes by the total count
        to create a probability distribution.

        Args:
            topics: List of topic strings

        Returns:
            Dict[str, float]: Dictionary mapping topics to their normalized frequencies
        """
        counts = Counter(topics)
        total = sum(counts.values())
        return {topic: count / total for topic, count in counts.items()}

    def _format_gemini_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation for Gemini model input"""
        template = {
            "conversation_analysis": {
                "metadata": {
                    "conversation_type": "human-AI",
                    "number_of_exchanges": len(messages),
                    "models_used": [],
                },
                "conversation_quality_metrics": {
                    "structural_coherence": {},
                    "intellectual_depth": {},
                    "interaction_dynamics": {},
                },
                "actor_specific_analysis": {},
                "thematic_analysis": {"primary_themes": [], "theme_development": {}},
                "conversation_effectiveness": {
                    "key_strengths": [],
                    "areas_for_improvement": [],
                },
            }
        }

        formatted = f"""Analyze the below conversation and provide output in this JSON structure:
            {json.dumps(template, indent=2)}
"""

        for msg in messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "").strip()
            formatted += f"{{{role}: {content}}} \n"

    def _get_gemini_analysis(self, conversation: str) -> Dict[str, Any]:
        """Get analysis from Gemini model"""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=conversation,
                config=genai.types.GenerateContentConfig(response_modalities=["JSON"]),
            )
            print(response.text)

            # Parse JSON response
            try:
                return json.loads(response.text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response: {e}")
                return self._create_empty_analysis()

        except Exception as e:
            logger.error(f"Error getting Gemini analysis: {e}")
            raise

    def _determine_winner(
        self, ai_ai_metrics: Dict[str, float], human_ai_metrics: Dict[str, float]
    ) -> str:
        """Determine conversation winner based on metrics"""
        # Calculate weighted scores
        weights = {
            "coherence": 0.2,
            "depth": 0.2,
            "engagement": 0.15,
            "reasoning": 0.15,
            "knowledge": 0.15,
            "goal_progress": 0.15,
        }

        ai_ai_score = sum(
            weights[metric] * value
            for metric, value in ai_ai_metrics.items()
            if metric in weights
        )

        human_ai_score = sum(
            weights[metric] * value
            for metric, value in human_ai_metrics.items()
            if metric in weights
        )

        return "ai-ai" if ai_ai_score > human_ai_score else "human-ai"

    def _combine_insights(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Combine and deduplicate insights from multiple analyses"""
        all_insights = []
        seen = set()

        for analysis in analyses:
            for insight in analysis.get("key_insights", []):
                normalized = insight.lower().strip()
                if normalized not in seen:
                    all_insights.append(insight)
                    seen.add(normalized)

        return all_insights

    def _gemini_search(
        self, ai_ai_analysis: Dict[str, Any], human_ai_analysis: Dict[str, Any]
    ) -> Any:  # Dict[str, Dict[str, AssertionEvidence]]:
        """Ground assertions from both conversations"""

        grounded = self._ground_assertions(self, ai_ai_analysis, human_ai_analysis)
        return grounded


class VisualizationGenerator:
    """Generates visualizations for conversation analysis"""

    def __init__(self):
        self.plotly = go

    def generate_metrics_chart(self, result: ArbiterResult) -> str:
        """Generate comparison chart of conversation metrics"""
        metrics = ["coherence", "depth", "engagement", "reasoning", "knowledge"]
        ai_ai_values = [
            getattr(result.conversation_metrics["ai-ai"], m) for m in metrics
        ]
        human_ai_values = [
            getattr(result.conversation_metrics["human-ai"], m) for m in metrics
        ]

        fig = self.plotly.Figure(
            data=[
                self.plotly.Bar(name="AI-AI", x=metrics, y=ai_ai_values),
                self.plotly.Bar(name="Human-AI", x=metrics, y=human_ai_values),
            ]
        )

        fig.update_layout(
            title="Conversation Metrics Comparison", barmode="group", yaxis_range=[0, 1]
        )

        return fig.to_html(full_html=False)

    def generate_timeline(
        self, assertions: Dict[str, Dict[str, AssertionEvidence]]
    ) -> str:
        """Generate timeline visualization of grounded assertions"""
        """
        Generate a timeline visualization of grounded assertions from both conversations.

        Creates a Plotly scatter plot showing assertions from both AI-AI and Human-AI
        conversations on a timeline, with each assertion represented as a point with text.

        Args:
            assertions: Dictionary mapping conversation types to their assertions

        Returns:
            str: HTML representation of the timeline visualization
        """
        ai_ai_assertions = list(assertions["ai-ai"].keys())
        human_ai_assertions = list(assertions["human-ai"].keys())

        fig = self.plotly.Figure(
            [
                self.plotly.Scatter(
                    x=list(range(len(ai_ai_assertions))),
                    y=[1] * len(ai_ai_assertions),
                    mode="markers+text",
                    name="AI-AI Assertions",
                    text=[
                        a[:30] + "..." if len(a) > 30 else a for a in ai_ai_assertions
                    ],
                    textposition="top center",
                ),
                self.plotly.Scatter(
                    x=list(range(len(human_ai_assertions))),
                    y=[0] * len(human_ai_assertions),
                    mode="markers+text",
                    name="Human-AI Assertions",
                    text=[
                        a[:30] + "..." if len(a) > 30 else a
                        for a in human_ai_assertions
                    ],
                    textposition="bottom center",
                ),
            ]
        )

        fig.update_layout(
            title="Conversation Timeline", showlegend=True, yaxis_visible=False
        )

        return fig.to_html(full_html=False)


def evaluate_conversations(
    ai_ai_convo: List[Dict[str, str]],
    human_ai_convo: List[Dict[str, str]],
    default_convo: List[Dict[str, str]],
    goal: str,
) -> ArbiterResult:
    """Compare and evaluate three conversation modes"""
    """
    Compare and evaluate three conversation modes: AI-AI, Human-AI, and default.

    This function performs comprehensive analysis of conversations including flow
    analysis, topic coherence, and grounding of assertions. It uses the Gemini API
    to evaluate the quality and effectiveness of different conversation modes.

    Args:
        ai_ai_convo: List of message dictionaries from AI-AI conversation
        human_ai_convo: List of message dictionaries from Human-AI conversation
        default_convo: List of message dictionaries from default conversation
        goal: The conversation goal or topic

    Returns:
        ArbiterResult: Comprehensive evaluation results
    """
    try:
        # Get Gemini API key from environment
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY environment variable is not set")
        
        # Initialize arbiter with error handling
        try:
            convmetrics = ConversationMetrics()
            arbiter = ConversationArbiter(api_key=gemini_api_key)
            logger.info("ConversationArbiter initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ConversationArbiter: {e}")
            # Create a fallback arbiter without proper initialization
            arbiter = None
    
        # Analyze conversation flows with enhanced error handling
        ai_ai_flow = None
        human_ai_flow = None
        default_flow = None
        flow_analysis_error = None
        
        if arbiter:
            try:
                if ai_ai_convo:  # Check if ai_ai_convo is not empty
                    ai_ai_flow = arbiter.analyze_conversation_flow(ai_ai_convo)
                    # Check if there was an error during analysis
                    if ai_ai_flow and "error" in ai_ai_flow:
                        flow_analysis_error = ai_ai_flow["error"]
                        logger.warning(f"Error in AI-AI flow analysis: {flow_analysis_error}")
                
                if human_ai_convo:  # Check if human_ai_convo is not empty
                    human_ai_flow = arbiter.analyze_conversation_flow(human_ai_convo)
                    # Check if there was an error during analysis
                    if human_ai_flow and "error" in human_ai_flow:
                        flow_analysis_error = human_ai_flow["error"]
                        logger.warning(f"Error in Human-AI flow analysis: {flow_analysis_error}")
                
                if default_convo:  # Check if default_convo is not empty
                    default_flow = arbiter.analyze_conversation_flow(default_convo)
                    # Check if there was an error during analysis
                    if default_flow and "error" in default_flow:
                        flow_analysis_error = default_flow["error"]
                        logger.warning(f"Error in default flow analysis: {flow_analysis_error}")
            except Exception as e:
                logger.error(f"Uncaught error during conversation flow analysis: {e}")
                flow_analysis_error = str(e)
        
        # If we found "Negative values in data" error, log it specially
        if flow_analysis_error and "Negative values in data" in flow_analysis_error:
            logger.error(f"Failed to generate metrics report: {flow_analysis_error}")
        
        # Ground assertions with Gemini API
        try:
            grounder = AssertionGrounder(api_key=gemini_api_key)
            logger.info(f"Using Gemini model: {grounder.model} for analysis")
            
            # Convert conversations to string form if needed
            formatted_ai_ai_convo = ai_ai_convo
            formatted_human_ai_convo = human_ai_convo
            formatted_default_convo = default_convo
            
            # Generate report with ground assertions
            result = grounder.ground_assertions(
                formatted_ai_ai_convo, formatted_human_ai_convo, formatted_default_convo, goal
            )
            
            # If flow analysis had errors, add a note to the HTML result
            if flow_analysis_error and isinstance(result, str) and "<div class=\"arbiter-report\">" in result:
                error_note = f"""
                <div class="section">
                    <h2>Flow Analysis Issues</h2>
                    <p class="error-message">Note: An error occurred during conversation flow analysis: {flow_analysis_error}</p>
                    <p>The analysis has proceeded with default metrics.</p>
                </div>
                """
                # Insert error note after the first section
                insertion_point = result.find("</div>", result.find("<div class=\"section\">"))
                if insertion_point > 0:
                    result = result[:insertion_point + 6] + error_note + result[insertion_point + 6:]
            
            return result
        except Exception as e:
            logger.error(f"Error in ground assertions: {e}")
            # Return an error HTML template instead of raising
            error_html = f"""
            <div class="arbiter-report">
                <div class="model-info">
                    <h2>Analysis Error</h2>
                    <p>Topic: {goal}</p>
                </div>
                <div class="section">
                    <h2>Error in Evaluation</h2>
                    <p>An error occurred during conversation evaluation: {str(e)}</p>
                </div>
                
                <!-- Empty sections to maintain template structure -->
                <div class="section">
                    <h2>Key Milestones</h2>
                    <p>No data available due to evaluation error</p>
                </div>
                
                <div class="section">
                    <h2>Conversation Scores</h2>
                    <p>No data available due to evaluation error</p>
                </div>
                
                <div class="section">
                    <h2>Participant Analysis</h2>
                    <p>No data available due to evaluation error</p>
                </div>
                
                <div class="section">
                    <h2>Comparative Analysis</h2>
                    <p>No data available due to evaluation error</p>
                </div>
            </div>
            """
            return error_html

    except Exception as e:
        logger.error(f"Unhandled error in conversation evaluation: {e}")
        # Return an error HTML template instead of raising
        error_html = f"""
        <div class="arbiter-report">
            <div class="model-info">
                <h2>Fatal Error</h2>
                <p>Topic: {goal}</p>
            </div>
            <div class="section">
                <h2>Unhandled Error</h2>
                <p>A fatal error occurred during conversation evaluation: {str(e)}</p>
            </div>
            
            <!-- Empty sections to maintain template structure -->
            <div class="section">
                <h2>Key Milestones</h2>
                <p>No data available due to fatal error</p>
            </div>
            
            <div class="section">
                <h2>Conversation Scores</h2>
                <p>No data available due to fatal error</p>
            </div>
            
            <div class="section">
                <h2>Participant Analysis</h2>
                <p>No data available due to fatal error</p>
            </div>
            
            <div class="section">
                <h2>Comparative Analysis</h2>
                <p>No data available due to fatal error</p>
            </div>
        </div>
        """
        return error_html
