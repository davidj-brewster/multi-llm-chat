"""Enhanced arbiter module combining functionality from v1 and v2"""
import json
import logging
import datetime
import spacy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from pydantic import BaseModel, Field, ConfigDict
from dataclasses import asdict
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
MODEL_CONFIG = ConfigDict(
    extra='allow',  # Allow additional fields
    arbitrary_types_allowed=True,
    protected_namespaces=('model_', ),
    extra_fields='allow',  # Allow additional fields
    json_schema_extra = {
        "additionalProperties": True
    }

)

# Keep the enhanced model classes from v2
class NLPAnalysis(BaseModel):
    model_config = MODEL_CONFIG
    """Advanced NLP analysis results"""
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    key_phrases: List[str] = Field(default_factory=list)
    argument_structure: Dict[str, List[str]] = Field(default_factory=dict)
    sentiment_scores: Dict[str, float] = Field(default_factory=dict)
    complexity_metrics: Dict[str, float] = Field(default_factory=dict)
    topic_clusters: Dict[str, List[str]] = Field(default_factory=dict)
    discourse_markers: Dict[str, int] = Field(default_factory=dict)
    reference_chains: List[List[str]] = Field(default_factory=list)

class ConversationMetrics(BaseModel):
    model_config = MODEL_CONFIG
    """Metrics for conversation quality assessment"""
    coherence: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    depth: float = Field(default=0.0, ge=0.0, le=1.0)
    engagement: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: float = Field(default=0.0, ge=0.0, le=1.0)
    knowledge: float = Field(default=0.0, ge=0.0, le=1.0)
    goal_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    strategy_effectiveness: float = Field(default=0.0, ge=0.0, le=1.0)

class ParticipantMetrics(BaseModel):
    model_config = MODEL_CONFIG
    """Metrics for individual participant performance"""
    response_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    knowledge_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning_depth: float = Field(default=0.0, ge=0.0, le=1.0)
    engagement_level: float = Field(default=0.0, ge=0.0, le=1.0)
    strategy_adherence: float = Field(default=0.0, ge=0.0, le=1.0)
    adaptation: float = Field(default=0.0, ge=0.0, le=1.0)

class AssertionEvidence(BaseModel):
    model_config = MODEL_CONFIG
    """Evidence supporting a grounded assertion"""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    sources: List[Dict[str, str]] = Field(default_factory=list)
    verification_method: str = "search"

class ArbiterResult(BaseModel):
    model_config = MODEL_CONFIG
    """Complete results of conversation arbitration"""
    winner: str
    conversation_metrics: Dict[str, ConversationMetrics]
    participant_metrics: Dict[str, Dict[str, ParticipantMetrics]]
    key_insights: List[str]
    improvement_suggestions: List[str]
    strategy_analysis: Dict[str, float]
    grounded_assertions: Dict[str, Dict[str, AssertionEvidence]]
    execution_timestamp: str
    conversation_ids: Dict[str, str]

# Add the Gemini model response schema
class AssessmentSchema(BaseModel):
    model_config = MODEL_CONFIG
    """Schema for Gemini model response validation"""
    conversation_quality: Dict[str, float] = Field(
        default_factory=dict,
        description="Quality metrics for the conversation"
    )
    participant_analysis: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Analysis metrics for each participant"
    )
    assertions: List[str] = Field(
        default_factory=list,
        description="Key assertions from the conversation"
    )
    key_insights: List[str] = Field(
        default_factory=list,
        description="Important insights from the analysis"
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for improvement"
    )

class AssertionGrounder:
    """Verifies and grounds assertions using search and NLP"""
    
    def __init__(self, search_client: Any):
        self.search_client = search_client
        try:
            self.nlp = spacy.load('en_core_web_trf')
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}")
            self.nlp = None

    def _calculate_confidence(self, 
                            sources: List[Dict[str, str]], 
                            assertion: str) -> float:
        """Calculate confidence score using improved metrics"""
        if not sources:
            return 0.0
            
        # Source scoring (from v1)
        source_score = min(len(sources) / 3.0, 1.0)
        
        # Authority scoring (from v1)
        authority_score = sum(0.2 if ".edu" in s["url"] or 
                                   ".gov" in s["url"] or 
                                   ".org" in s["url"] 
                            else 0.1 
                            for s in sources) / len(sources)
        
        # Enhanced consistency scoring (from v2)
        try:
            if self.nlp:
                assertion_doc = self.nlp(assertion)
                similarities = []
                for source in sources:
                    source_doc = self.nlp(source["excerpt"])
                    similarity = assertion_doc.similarity(source_doc)
                    similarities.append(similarity)
                consistency_score = sum(similarities) / len(similarities)
            else:
                # Fallback to basic SequenceMatcher (from v1)
                from difflib import SequenceMatcher
                consistency_scores = []
                for i, s1 in enumerate(sources):
                    for s2 in sources[i+1:]:
                        ratio = SequenceMatcher(None, 
                                              s1["excerpt"], 
                                              s2["excerpt"]).ratio()
                        consistency_scores.append(ratio)
                consistency_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5
        except Exception as e:
            logger.error(f"Error in consistency calculation: {e}")
            consistency_score = 0.5

        # Combined scoring with weights
        final_score = (
            source_score * 0.3 +
            authority_score * 0.3 +
            consistency_score * 0.4
        )
        
        return min(final_score, 1.0)

class ConversationArbiter:
    """Evaluates and compares conversations using Gemini model with enhanced analysis"""
    
    def __init__(self, api_key: str, 
                 model: str = "gemini-exp-1206",
                 search_client: Optional[Any] = None):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.grounder = AssertionGrounder(search_client) if search_client else None
        try:
            self.nlp = spacy.load('en_core_web_trf')
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}")
            self.nlp = None

    async def analyze_conversation_flow(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze conversation flow patterns and transitions"""
        try:
            # Use spaCy for linguistic analysis if available
            if self.nlp:
                docs = [self.nlp(msg["content"]) for msg in messages]
                
                # Analyze topic transitions
                topics = []
                for doc in docs:
                    # Extract main topics using noun chunks and named entities
                    topics.extend([chunk.text for chunk in doc.noun_chunks])
                    topics.extend([ent.text for ent in doc.ents])
                
                # Calculate topic coherence
                topic_shifts = 0
                for i in range(1, len(topics)):
                    if not any(self._text_similarity(topics[i], prev) > 0.3 
                             for prev in topics[max(0, i-3):i]):
                        topic_shifts += 1
                
                flow_metrics = {
                    "topic_coherence": 1.0 - (topic_shifts / len(messages)),
                    "topic_depth": len(set(topics)) / len(messages),
                    "topic_distribution": self._calculate_topic_distribution(topics)
                }
                
            else:
                # Fallback to basic analysis
                flow_metrics = self._basic_flow_analysis(messages)
            
            return flow_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing conversation flow: {e}")
            return {"topic_coherence": 0.5, "topic_depth": 0.5, "topic_distribution": {}}

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if self.nlp:
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            return doc1.similarity(doc2)
        else:
            # Fallback to basic string similarity
            return SequenceMatcher(None, text1, text2).ratio()

    def _calculate_topic_distribution(self, topics: List[str]) -> Dict[str, float]:
        """Calculate normalized topic frequencies"""
        counts = Counter(topics)
        total = sum(counts.values())
        return {topic: count/total for topic, count in counts.items()}

    def _compare_conversations(self, 
                             ai_ai_analysis: Dict,
                             human_ai_analysis: Dict) -> Tuple[str, List[str]]:
        """Compare analyses to determine winner"""
        
        # Calculate overall scores with grounding bonus
        def calc_score(analysis: Dict) -> float:
            base_score = sum([
                analysis["conversation_quality"]["coherence"],
                analysis["conversation_quality"]["depth"],
                analysis["conversation_quality"]["goal_progress"]
            ]) / 3
            
            # Bonus for grounded assertions
            grounding_score = len(analysis.get("grounded_assertions", {})) * 0.05
            
            return base_score + min(grounding_score, 0.2)  # Cap grounding bonus
        
        ai_ai_score = calc_score(ai_ai_analysis)
        human_ai_score = calc_score(human_ai_analysis)
        
        # Determine winner
        winner = "ai-ai" if ai_ai_score > human_ai_score else "human-ai"
        
        # Extract key differences
        differences = []
        metrics = ["coherence", "depth", "goal_progress", "grounded_assertions"]
        for metric in metrics:
            ai_ai_val = (ai_ai_analysis["conversation_quality"].get(metric, 0) 
                        if metric != "grounded_assertions"
                        else len(ai_ai_analysis.get("grounded_assertions", {})))
            human_ai_val = (human_ai_analysis["conversation_quality"].get(metric, 0)
                          if metric != "grounded_assertions" 
                          else len(human_ai_analysis.get("grounded_assertions", {})))
            
            diff = abs(ai_ai_val - human_ai_val)
            if diff > 0.1 or (metric == "grounded_assertions" and diff > 0):
                better = "AI-AI" if ai_ai_val > human_ai_val else "Human-AI"
                differences.append(
                    f"{better} performed better at {metric}: "
                    f"({ai_ai_val:.2f} vs {human_ai_val:.2f})"
                )
        
        return winner, differences
    
    def _create_empty_analysis(self) -> Dict:
        """Create empty analysis structure for error cases"""
        return {
            "conversation_quality": {
                "coherence": 0.0,
                "relevance": 0.0,
                "depth": 0.0,
                "engagement": 0.0,
                "reasoning": 0.0,
                "knowledge": 0.0,
                "goal_progress": 0.0,
                "strategy_effectiveness": 0.0,
                "grounded_assertions": 0,
                "unverified_claims": 0
            },
            "participant_analysis": {},
            "assertions": [],
            "key_insights": [
                "Analysis failed - insufficient data"
            ],
            "improvement_suggestions": [
                "Retry analysis with valid conversation data"
            ]
        }
    

    def _create_evaluation_prompt(self, conversation: List[Dict[str, str]], 
                                goal: str, mode: str) -> str:
        """Create detailed prompt for conversation evaluation"""
        return f"""
        Analyze this conversation and return a structured evaluation in JSON format.
        
        GOAL: {goal}
        MODE: {mode}
        
        REQUIRED OUTPUT FORMAT:
        {{
            "conversation_quality": {{
                "coherence": <float 0-1>,        // Message flow and transitions
                "relevance": <float 0-1>,        // Alignment with goal/topic
                "depth": <float 0-1>,            // Level of insight and analysis
                "engagement": <float 0-1>,       // Interaction quality
                "reasoning": <float 0-1>,        // Logic and analytical thinking
                "knowledge": <float 0-1>,        // Information accuracy and breadth
                "goal_progress": <float 0-1>,    // Movement toward objective
                "strategy_effectiveness": <float 0-1>,  // Effectiveness of approaches
                "grounded_assertions": <integer>, // Count of verifiable claims
                "unverified_claims": <integer>   // Count of unverified claims
            }},
            "participant_analysis": {{
                "participant_1": {{
                    "response_quality": <float 0-1>,    // Clarity and completeness
                    "knowledge_accuracy": <float 0-1>,  // Factual correctness
                    "reasoning_depth": <float 0-1>,     // Analytical thinking
                    "engagement_level": <float 0-1>,    // Interaction quality
                    "strategy_adherence": <float 0-1>,  // Following approaches
                    "adaptation": <float 0-1>,          // Adjusting to flow
                    "factual_accuracy": <float 0-1>,    // Claim correctness
                    "citation_quality": <float 0-1>     // Evidence usage
                }},
                "participant_2": {{
                    "response_quality": <float 0-1>,
                    "knowledge_accuracy": <float 0-1>,
                    "reasoning_depth": <float 0-1>,
                    "engagement_level": <float 0-1>,
                    "strategy_adherence": <float 0-1>,
                    "adaptation": <float 0-1>,
                    "factual_accuracy": <float 0-1>,
                    "citation_quality": <float 0-1>
                }}
            }},
            "assertions": [
                // List of factual claims that need verification
                // Each assertion should be a specific, verifiable statement
                "assertion 1",
                "assertion 2"
            ],
            "key_insights": [
                // List of notable patterns, techniques, and critical moments
                "insight 1",
                "insight 2"
            ],
            "improvement_suggestions": [
                // List of specific recommendations and alternative approaches
                "suggestion 1",
                "suggestion 2"
            ]
        }}
        
        EVALUATION GUIDELINES:
        
        1. Score all numeric metrics on a 0-1 scale where:
           - 0.0-0.2: Poor performance
           - 0.3-0.4: Below average
           - 0.5-0.6: Average
           - 0.7-0.8: Good
           - 0.9-1.0: Excellent
        
        2. When identifying assertions:
           - Focus on specific, verifiable factual claims
           - Include quantitative statements
           - Note technical claims requiring evidence
           - Highlight referenced research or studies
        
        3. For key insights:
           - Identify effective patterns and techniques
           - Note significant breakthroughs or realizations
           - Highlight missed opportunities
           - Mark critical turning points
        
        4. For improvement suggestions:
           - Provide specific, actionable recommendations
           - Suggest alternative approaches
           - Recommend strategy adjustments
           - Focus on enhancing goal achievement
        
        IMPORTANT:
        - Return ONLY valid JSON matching the specified schema
        - Ensure all numeric values are between 0 and 1
        - Include specific examples and quotes to support ratings
        - Ground factual claims in the conversation content
        """
    
    def _analyze_conversation(self, conversation: List[Dict[str, str]], 
                            goal: str, mode: str) -> Dict:
        """Analyze a single conversation with assertion grounding"""
        analysis = None
        prompt = self._create_evaluation_prompt(conversation, goal, mode)
        
        try:
            # Get initial analysis
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=AssessmentSchema,
                    temperature=0.1,
                    maxOutputTokens=4096,
                    candidateCount=1,
                    tools=[types.Tool(
                        google_search=types.GoogleSearchRetrieval(
                            dynamic_retrieval_config=types.DynamicRetrievalConfig(
                                dynamic_threshold=0.6))
                    )]
                )
            )
            
            try:
                analysis = response.parsed
            except Exception as parse_error:
                logger.error(f"Failed to parse Gemini response as JSON: {parse_error}")
                # Attempt to extract JSON from text response
                import re
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                analysis = json.loads(json_match.group(0)) if json_match else None
            
            # Ground assertions if grounder available
            if self.grounder:
                grounded_assertions = {}
                for assertion in analysis.get("assertions", []):
                    evidence = self.grounder.ground_assertion(assertion)
                    if evidence.confidence > 0.5:  # Only keep well-supported assertions
                        grounded_assertions[assertion] = evidence
                analysis["grounded_assertions"] = grounded_assertions
            
            if not analysis:
                raise ValueError("Failed to parse model response into valid analysis format")
            
            return analysis
                
        except Exception as e:
            logger.error(f"Error analyzing conversation: {e}")
            raise
            #return self._create_empty_analysis()
    
    def evaluate_conversations(self,
                             ai_ai_conversation: List[Dict[str, str]],
                             human_ai_conversation: List[Dict[str, str]],
                             goal: str) -> ArbiterResult:
        """Evaluate and compare two conversations"""
        
        # Generate unique IDs for conversations
        from uuid import uuid4
        conversation_ids = {
            "ai-ai": str(uuid4()),
            "human-ai": str(uuid4())
        }
        
        # Analyze each conversation
        ai_ai_analysis = self._analyze_conversation(
            ai_ai_conversation, goal, "ai-ai"
        )
        human_ai_analysis = self._analyze_conversation(
            human_ai_conversation, goal, "human-ai"
        )
        
        # Compare and determine winner
        winner, key_differences = self._compare_conversations(
            ai_ai_analysis, human_ai_analysis
        )
        
        # Extract strategy scores with fallback
        strategy_analysis = {
            "ai-ai": ai_ai_analysis["conversation_quality"].get("strategy", 0.0),
            "human-ai": human_ai_analysis["conversation_quality"].get("strategy", 0.0)
        }
        
        result = ArbiterResult(
            winner=winner,
            conversation_metrics={
                "ai-ai": ConversationMetrics(**ai_ai_analysis["conversation_quality"]),
                "human-ai": ConversationMetrics(**human_ai_analysis["conversation_quality"])
            },
            participant_metrics={
                "ai-ai": {
                    role: ParticipantMetrics(**metrics)
                    for role, metrics in ai_ai_analysis["participant_analysis"].items()
                },
                "human-ai": {
                    role: ParticipantMetrics(**metrics)
                    for role, metrics in human_ai_analysis["participant_analysis"].items()
                }
            },
            key_insights=key_differences + 
                        ai_ai_analysis["key_insights"] +
                        human_ai_analysis["key_insights"],
            improvement_suggestions=ai_ai_analysis["improvement_suggestions"] +
                                  human_ai_analysis["improvement_suggestions"],
            strategy_analysis=strategy_analysis,
            grounded_assertions={
                "ai-ai": ai_ai_analysis.get("grounded_assertions", {}),
                "human-ai": human_ai_analysis.get("grounded_assertions", {})
            },
            execution_timestamp=datetime.datetime.now().isoformat(),
            conversation_ids=conversation_ids
        )
        
        # Save metrics to JSON
        self._save_metrics(result)
        
        return result
    
    def _save_metrics(self, result: ArbiterResult) -> None:
        """Save metrics to JSON file for analytics"""
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        
        metrics_file = metrics_dir / "conversation_metrics.json"
        
        # Load existing metrics if any
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics_history = json.load(f)
        else:
            metrics_history = []
        
        # Add new metrics
        metrics_history.append({
            "timestamp": result.execution_timestamp,
            "conversation_ids": result.conversation_ids,
            "winner": result.winner,
            "metrics": {
                "ai-ai": asdict(result.conversation_metrics["ai-ai"]),
                "human-ai": asdict(result.conversation_metrics["human-ai"])
            },
            "participant_metrics": {
                mode: {
                    role: asdict(metrics)
                    for role, metrics in role_metrics.items()
                }
                for mode, role_metrics in result.participant_metrics.items()
            },
            "grounded_assertions": {
                mode: {
                    assertion: asdict(evidence)
                    for assertion, evidence in assertions.items()
                }
                for mode, assertions in result.grounded_assertions.items()
            }
        })
        
        # Save updated metrics
        with open(metrics_file, 'w') as f:
            json.dump(metrics_history, f, indent=2)
    
    def generate_report(self, result: ArbiterResult) -> str:
        """Generate detailed HTML report of arbiter results"""
        try:
            with open("templates/arbiter_report.html") as f:
                template = f.read()
        except Exception as e:
            logger.warning(f"Failed to load report template: {e}")
            template = None
            
        return template.format(report_content=self._generate_report_content(result))

    def _generate_report_content(self, result: ArbiterResult) -> str:
        """Generate the HTML content for the report"""
        return f"""
        <link rel="stylesheet" href="/static/css/arbiter_report.css">
        <div class="arbiter-report">
            <h2>Conversation Analysis Report</h2>
            
            <div class="winner-section">
                <h3>Winner: {result.winner.upper()}</h3>
                <p>This mode demonstrated better overall performance in achieving the conversation goals.</p>
            </div>
            
            <div class="metrics-section">
                <h3>Conversation Metrics</h3>
                <div class="metrics-comparison">
                    <div class="ai-ai-metrics">
                        <h4>AI-AI Conversation</h4>
                        <ul>
                            <li>Coherence: <span class="metric-value">{result.conversation_metrics["ai-ai"].coherence:.2f}</span></li>
                            <li>Depth: <span class="metric-value">{result.conversation_metrics["ai-ai"].depth:.2f}</span></li>
                            <li>Goal Progress: <span class="metric-value">{result.conversation_metrics["ai-ai"].goal_progress:.2f}</span></li>
                            <li>Grounded Assertions: <span class="metric-value">{len(result.grounded_assertions["ai-ai"])}</span></li>
                        </ul>
                    </div>
                    <div class="human-ai-metrics">
                        <h4>Human-AI Conversation</h4>
                        <ul>
                            <li>Coherence: <span class="metric-value">{result.conversation_metrics["human-ai"].coherence:.2f}</span></li>
                            <li>Depth: <span class="metric-value">{result.conversation_metrics["human-ai"].depth:.2f}</span></li>
                            <li>Goal Progress: <span class="metric-value">{result.conversation_metrics["human-ai"].goal_progress:.2f}</span></li>
                            <li>Grounded Assertions: <span class="metric-value">{len(result.grounded_assertions["human-ai"])}</span></li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="grounded-assertions-section">
                <h3>Grounded Assertions</h3>
                <div class="ai-ai-assertions">
                    <h4>AI-AI Conversation Assertions</h4>
                    {self._format_assertions(result.grounded_assertions["ai-ai"])}
                </div>
                <div class="human-ai-assertions">
                    <h4>Human-AI Conversation Assertions</h4>
                    {self._format_assertions(result.grounded_assertions["human-ai"])}
                </div>
            </div>
            
            <div class="insights-section">
                <h3>Key Insights</h3>
                <ul>
                    {
                        ''.join(f"<li>{insight}</li>" for insight in result.key_insights)
                    }
                </ul>
            </div>
            
            <div class="suggestions-section">
                <h3>Improvement Suggestions</h3>
                <ul>
                    {
                        ''.join(f"<li>{suggestion}</li>" for suggestion in result.improvement_suggestions)
                    }
                </ul>
            </div>
            
            <div class="strategy-section">
                <h3>Strategy Analysis</h3>
                <p>AI-AI Strategy Effectiveness: <span class="metric-value">{result.strategy_analysis["ai-ai"]:.2f}</span></p>
                <p>Human-AI Strategy Effectiveness: <span class="metric-value">{result.strategy_analysis["human-ai"]:.2f}</span></p>
            </div>
        </div>
        """
    
    def _format_assertions(self, assertions: Dict[str, AssertionEvidence]) -> str:
        """Format grounded assertions as HTML"""
        if not assertions:
            return "<p>No grounded assertions found.</p>"
            
        html = ["<div class='assertions-list'>"]
        for assertion, evidence in assertions.items():
            html.append(f"""
                <div class='assertion-item'>
                    <p class='assertion-text'>{assertion}</p>
                    <div class='evidence-section'>
                        <p class='confidence'>Confidence: {evidence.confidence:.2%}</p>
                        <div class='sources'>
                            <h5>Sources:</h5>
                            <ul>
                                {
                                    ''.join(f"<li><a href='{source['url']}'>{source['title']}</a><br/><small>{source['excerpt']}</small></li>"
                                           for source in evidence.sources)
                                }
                            </ul>
                        </div>
                    </div>
                </div>
            """)
        html.append("</div>")
        return '\n'.join(html)

    def _extract_reasoning_patterns(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Extract and analyze reasoning patterns in messages"""
        patterns = []
        reasoning_markers = {
            "causal": ["because", "therefore", "thus", "as a result"],
            "comparative": ["however", "whereas", "while", "in contrast"],
            "evidence": ["evidence shows", "research indicates", "according to"],
            "logical": ["if...then", "consequently", "it follows that"],
            "analytical": ["analyze", "consider", "examine", "evaluate"]
        }
        
        for msg in messages:
            content = msg["content"].lower()
            found_patterns = defaultdict(int)
            
            for pattern_type, markers in reasoning_markers.items():
                for marker in markers:
                    if marker in content:
                        found_patterns[pattern_type] += 1
            
            if found_patterns:
                patterns.append({
                    "message_index": messages.index(msg),
                    "patterns": dict(found_patterns)
                })
        
        return patterns

    def _assess_knowledge_integration(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """Assess how well knowledge is integrated and built upon"""
        knowledge_metrics = {
            "build_on_previous": 0.0,
            "fact_density": 0.0,
            "knowledge_depth": 0.0
        }
        
        try:
            # Calculate metrics
            total_facts = 0
            knowledge_connections = 0
            
            for i, msg in enumerate(messages):
                # Count potential fact statements
                sentences = msg["content"].split(". ")
                facts = len([s for s in sentences if any(marker in s.lower() 
                    for marker in ["is", "are", "was", "were", "research", "study", "evidence"])])
                total_facts += facts
                
                # Look for references to previous messages
                if i > 0:
                    prev_content = " ".join(m["content"] for m in messages[max(0, i-3):i])
                    references = sum(1 for word in msg["content"].split() 
                                  if word in prev_content)
                    knowledge_connections += references / len(msg["content"].split())
            
            # Calculate final metrics
            if len(messages) > 0:
                knowledge_metrics["fact_density"] = min(1.0, total_facts / len(messages))
                knowledge_metrics["build_on_previous"] = min(1.0, knowledge_connections / len(messages))
                knowledge_metrics["knowledge_depth"] = min(1.0, (knowledge_metrics["fact_density"] + 
                                                               knowledge_metrics["build_on_previous"]) / 2)
        
        except Exception as e:
            logger.error(f"Error in knowledge integration assessment: {e}")
        
        return knowledge_metrics

class VisualizationGenerator:
    """Generates visualizations for conversation analysis"""
    
    def __init__(self):
        try:
            import plotly.graph_objects as go
            self.plotly = go
        except ImportError:
            logger.warning("Plotly not available - visualizations will be limited")
            self.plotly = None
            
    def generate_metrics_chart(self, result: ArbiterResult) -> str:
        """Generate comparison chart of conversation metrics"""
        if not self.plotly:
            return ""
            
        # Extract metrics for comparison
        metrics = ["coherence", "depth", "engagement", "reasoning", "knowledge"]
        ai_ai_values = [getattr(result.conversation_metrics["ai-ai"], m) for m in metrics]
        human_ai_values = [getattr(result.conversation_metrics["human-ai"], m) for m in metrics]
        
        fig = self.plotly.Figure(data=[
            self.plotly.Bar(name="AI-AI", x=metrics, y=ai_ai_values),
            self.plotly.Bar(name="Human-AI", x=metrics, y=human_ai_values)
        ])
        
        fig.update_layout(
            title="Conversation Metrics Comparison",
            barmode="group",
            yaxis_range=[0, 1]
        )
        
        return fig.to_html(full_html=False)

    def generate_timeline(self, result: ArbiterResult) -> str:
        """Generate timeline visualization of conversation flow"""
        if not self.plotly:
            return ""
            
        # Create timeline data
        ai_ai_assertions = list(result.grounded_assertions["ai-ai"].keys())
        human_ai_assertions = list(result.grounded_assertions["human-ai"].keys())
        
        fig = self.plotly.Figure([
            self.plotly.Scatter(
                x=list(range(len(ai_ai_assertions))),
                y=[1] * len(ai_ai_assertions),
                mode="markers+text",
                name="AI-AI Assertions",
                text=ai_ai_assertions,
                textposition="top center"
            ),
            self.plotly.Scatter(
                x=list(range(len(human_ai_assertions))),
                y=[0] * len(human_ai_assertions),
                mode="markers+text",
                name="Human-AI Assertions", 
                text=human_ai_assertions,
                textposition="bottom center"
            )
        ])
        
        fig.update_layout(
            title="Conversation Timeline",
            showlegend=True,
            yaxis_visible=False
        )
        
        return fig.to_html(full_html=False)

class ReportGenerator:
    """Generates detailed HTML reports with visualizations"""
    
    def __init__(self):
        self.viz = VisualizationGenerator()
        
    def generate_report(self, result: ArbiterResult) -> str:
        """Generate complete HTML report with metrics and visualizations"""
        return f"""
        <div class="arbiter-report">
            <h1>Conversation Analysis Report</h1>
            
            <div class="summary-section">
                <h2>Summary</h2>
                <p>Winner: <strong>{result.winner}</strong></p>
                <p>Analysis completed: {result.execution_timestamp}</p>
            </div>
            
            <div class="visualization-section">
                <h2>Metrics Comparison</h2>
                {self.viz.generate_metrics_chart(result)}
                
                <h2>Conversation Timeline</h2>
                {self.viz.generate_timeline(result)}
            </div>
            
            <div class="detailed-metrics">
                <h2>Detailed Metrics</h2>
                {self._format_detailed_metrics(result)}
            </div>
            
            <div class="insights-section">
                <h2>Key Insights</h2>
                {self._format_insights(result)}
            </div>
            
            <div class="grounded-assertions">
                <h2>Grounded Assertions</h2>
                {self._format_assertions(result)}
            </div>
        </div>
        """
    
    def _format_detailed_metrics(self, result: ArbiterResult) -> str:
        """Format detailed metrics as HTML tables"""
        return f"""
        <div class="metrics-tables">
            <h3>AI-AI Conversation</h3>
            {self._metrics_table(result.conversation_metrics["ai-ai"])}
            
            <h3>Human-AI Conversation</h3>
            {self._metrics_table(result.conversation_metrics["human-ai"])}
        </div>
        """
    
    def _metrics_table(self, metrics: ConversationMetrics) -> str:
        """Create HTML table for conversation metrics"""
        rows = []
        for field, value in metrics:
            rows.append(f"""
            <tr>
                <td>{field.title()}</td>
                <td>{value:.2f}</td>
            </tr>
            """)
        
        return f"""
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Score</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
def evaluate_conversations(ai_ai_conversation: List[Dict[str, str]],
                         human_ai_conversation: List[Dict[str, str]],
                         goal: str,
                         gemini_api_key: str,
                         search_client: Optional[Any] = None) -> Tuple[str, str]:
    """
    Evaluate two conversations and return winner with report
    
    Args:
        ai_ai_conversation: List of messages from AI-AI conversation
        human_ai_conversation: List of messages from Human-AI conversation
        goal: Original conversation goal/topic
        gemini_api_key: API key for Gemini model
        search_client: Optional search client for grounding assertions
    
    Returns:
        Tuple of (winner, HTML report)
    """
    model_config = MODEL_CONFIG
    try:
        arbiter = ConversationArbiter(
            api_key=gemini_api_key,
            search_client=search_client
        )
        result = arbiter.evaluate_conversations(
            ai_ai_conversation=ai_ai_conversation,
            human_ai_conversation=human_ai_conversation,
            goal=goal
        )
        report = arbiter.generate_report(result)
        return result.winner, report
    except Exception as e:
        logger.error(f"Error in conversation evaluation: {e}")
        return "unknown", f"<p>Error evaluating conversations: {str(e)}</p>"