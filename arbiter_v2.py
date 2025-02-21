"""Enhanced arbiter module with improved validation and NLP capabilities"""
import json
import logging
import datetime
import spacy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, ConfigDict
from dataclasses import asdict
from google import genai
from google.genai import types
import spacy
logger = logging.getLogger(__name__)

class NLPAnalysis(BaseModel):
    """Advanced NLP analysis results"""
    entities: Dict[str, List[str]] = Field(default_factory=dict)  # Named entities by type
    key_phrases: List[str] = Field(default_factory=list)  # Important phrases
    argument_structure: Dict[str, List[str]] = Field(default_factory=dict)  # Subject-verb-object patterns
    sentiment_scores: Dict[str, float] = Field(default_factory=dict)  # Detailed sentiment analysis
    complexity_metrics: Dict[str, float] = Field(default_factory=dict)  # Syntactic complexity measures
    topic_clusters: Dict[str, List[str]] = Field(default_factory=dict)  # Related terms by topic
    discourse_markers: Dict[str, int] = Field(default_factory=dict)  # Rhetorical devices and transitions
    reference_chains: List[List[str]] = Field(default_factory=list)  # Coreference chains

class MessageAnalysis(BaseModel):
    """Detailed analysis of a single message"""
    content: str
    nlp_analysis: NLPAnalysis
    tokens: int
    sentences: int
    avg_sentence_length: float
    readability_score: float
    response_type: str  # e.g., "explanation", "question", "challenge", "agreement"
    discourse_level: str  # e.g., "surface", "analytical", "evaluative", "synthetic"

class ConversationSummary(BaseModel):
    """Structured summary of conversation context"""
    key_points: List[str] = Field(default_factory=list)
    main_topics: Dict[str, float] = Field(default_factory=dict)  # Topic -> importance score
    argument_chain: List[Dict[str, str]] = Field(default_factory=list)  # Logical flow of arguments
    unresolved_questions: List[str] = Field(default_factory=list)
    context_length: int = 0  # Original context length in tokens
    summary_length: int = 0  # Summarized length in tokens

class AssertionEvidence(BaseModel):
    """Evidence supporting an assertion"""
    assertion: str
    sources: List[Dict[str, str]]  # List of {url: str, excerpt: str}
    confidence: float
    verification_method: str

class ConversationMetrics(BaseModel):
    """Metrics for conversation evaluation"""
    coherence: float
    relevance: float
    depth: float
    engagement: float
    reasoning: float
    knowledge: float
    goal_progress: float
    strategy_effectiveness: float
    grounded_assertions: int = 0
    unverified_claims: int = 0

class ParticipantMetrics(BaseModel):
    """Metrics for individual participant evaluation"""
    response_quality: float
    knowledge_accuracy: float
    reasoning_depth: float
    engagement_level: float
    strategy_adherence: float
    adaptation: float
    factual_accuracy: float
    citation_quality: float
    message_analyses: List[MessageAnalysis] = Field(default_factory=list)

class AssessmentSchema(BaseModel):
    """Schema for conversation assessment results"""
    model_config = ConfigDict(extra='allow')  # Allow extra fields

    class ParticipantRating(BaseModel):
        model_config = ConfigDict(extra='allow')
        coherence: float
        engagement: float
        reasoning_depth: float
        response_relevance: float

    class ConversationQuality(BaseModel):
        model_config = ConfigDict(extra='allow')
        flow_coherence: float
        topic_depth: float
        knowledge_exchange: float
        goal_progress: float

    participant_ratings: Dict[str, Dict[str, float]]
    conversation_quality: Dict[str, float]
    assertions: List[str]
    key_insights: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)

class ArbiterResult(BaseModel):
    """Complete evaluation result"""
    winner: str
    conversation_metrics: Dict[str, ConversationMetrics]
    participant_metrics: Dict[str, Dict[str, ParticipantMetrics]]
    key_insights: List[str]
    improvement_suggestions: List[str]
    strategy_analysis: Dict[str, float]
    grounded_assertions: Dict[str, Dict[str, AssertionEvidence]]
    execution_timestamp: str
    conversation_ids: Dict[str, str]

class AssertionGrounder:
    """Verifies and grounds assertions using search and NLP"""
    
    def __init__(self, search_client: Any):
        self.search_client = search_client
        try:
            self.nlp = spacy.load('en_core_web_trf')
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}")
            self.nlp = None
        
    def _analyze_message_nlp(self, content: str) -> MessageAnalysis:
        """Perform detailed NLP analysis of a message"""
        if not self.nlp:
            return self._fallback_message_analysis(content)
            
        try:
            doc = self.nlp(content)
            
            # Named Entity Recognition
            entities = defaultdict(list)
            for ent in doc.ents:
                entities[ent.label_].append(ent.text)
            
            # Key phrase extraction using noun chunks and verb phrases
            key_phrases = [chunk.text for chunk in doc.noun_chunks]
            key_phrases.extend([
                token.text for token in doc 
                if token.pos_ == "VERB" and token.dep_ in ["ROOT", "xcomp"]
            ])
            
            # Argument structure analysis
            arg_structure = defaultdict(list)
            for token in doc:
                if token.dep_ == "nsubj":
                    # Find subject-verb-object patterns
                    subj = token.text
                    verb = token.head.text
                    obj = next((child.text for child in token.head.children 
                              if child.dep_ == "dobj"), "")
                    if obj:
                        arg_structure["svo"].append(f"{subj}-{verb}-{obj}")
            
            # Sentiment analysis using transformer features
            sentiment_scores = {
                "positive": float(doc._.trf_data.tensors[0].mean()),
                "negative": float(doc._.trf_data.tensors[0].std())
            }
            
            # Syntactic complexity
            complexity = {
                "tree_depth": max(token.head.i - token.i for token in doc),
                "clause_count": len([token for token in doc if token.dep_ == "ROOT"]),
                "subordinate_clauses": len([token for token in doc 
                                          if token.dep_ in ["advcl", "acl", "ccomp"]])
            }
            
            # Topic clustering using transformer embeddings
            topic_vectors = doc._.trf_data.tensors[0].reshape(-1, 768)  # Transformer dim
            from sklearn.cluster import KMeans
            n_clusters = min(3, len(topic_vectors))
            if n_clusters > 0:
                kmeans = KMeans(n_clusters=n_clusters)
                clusters = kmeans.fit_predict(topic_vectors)
                topic_clusters = defaultdict(list)
                for i, cluster in enumerate(clusters):
                    topic_clusters[f"topic_{cluster}"].append(doc[i].text)
            else:
                topic_clusters = {}
            
            # Discourse markers
            discourse_markers = Counter([
                token.text for token in doc
                if token.dep_ == "discourse" or token.text.lower() in {
                    "however", "therefore", "thus", "moreover", "furthermore",
                    "nevertheless", "consequently", "meanwhile", "subsequently"
                }
            ])
            
            # Coreference chains (if available)
            reference_chains = []
            if doc.has_annotation("coref"):
                for chain in doc._.coref_chains:
                    reference_chains.append([doc[i].text for i in chain])
            
            nlp_analysis = NLPAnalysis(
                entities=dict(entities),
                key_phrases=key_phrases,
                argument_structure=dict(arg_structure),
                sentiment_scores=sentiment_scores,
                complexity_metrics=complexity,
                topic_clusters=dict(topic_clusters),
                discourse_markers=dict(discourse_markers),
                reference_chains=reference_chains
            )
            
            return MessageAnalysis(
                content=content,
                nlp_analysis=nlp_analysis,
                tokens=len(doc),
                sentences=len(list(doc.sents)),
                avg_sentence_length=len(doc) / len(list(doc.sents)) if doc.sents else 0,
                readability_score=self._calculate_readability(doc),
                response_type=self._classify_response_type(doc),
                discourse_level=self._analyze_discourse_level(doc)
            )
            
        except Exception as e:
            logger.error(f"Error in NLP analysis: {e}")
            return self._fallback_message_analysis(content)
    
    def _fallback_message_analysis(self, content: str) -> MessageAnalysis:
        """Basic analysis when NLP is not available"""
        words = content.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        
        return MessageAnalysis(
            content=content,
            nlp_analysis=NLPAnalysis(),
            tokens=len(words),
            sentences=len(sentences),
            avg_sentence_length=len(words) / len(sentences) if sentences else 0,
            readability_score=0.0,
            response_type="unknown",
            discourse_level="unknown"
        )
    
    def _calculate_readability(self, doc) -> float:
        """Calculate readability score using transformer features"""
        try:
            # Use transformer attention patterns for readability
            attention = doc._.trf_data.attention[0]  # First layer attention
            coherence = float(attention.mean())  # Average attention score
            complexity = float(attention.std())  # Attention variation
            return (coherence + (1 - complexity)) / 2  # Normalize to [0,1]
        except:
            return 0.0
    
    def _classify_response_type(self, doc) -> str:
        """Classify response type using linguistic features"""
        # Question detection
        if any(token.tag_ == "." and token.text == "?" for token in doc):
            return "question"
        
        # Look for discourse markers
        markers = [token.text.lower() for token in doc if token.dep_ == "discourse"]
        if any(m in ["however", "but", "although"] for m in markers):
            return "challenge"
        if any(m in ["therefore", "thus", "hence"] for m in markers):
            return "explanation"
        if any(m in ["yes", "agree", "correct"] for m in markers):
            return "agreement"
            
        return "statement"
    
    def _analyze_discourse_level(self, doc) -> str:
        """Analyze level of discourse using linguistic features"""
        # Count analytical markers
        analytical = len([t for t in doc if t.text.lower() in {
            "analyze", "examine", "consider", "evaluate", "assess"
        }])
        
        # Count evaluative markers
        evaluative = len([t for t in doc if t.text.lower() in {
            "better", "worse", "best", "worst", "should", "must"
        }])
        
        # Count synthetic markers
        synthetic = len([t for t in doc if t.text.lower() in {
            "combine", "integrate", "synthesize", "merge", "unify"
        }])
        
        if synthetic > 0:
            return "synthetic"
        if evaluative > analytical:
            return "evaluative"
        if analytical > 0:
            return "analytical"
        return "surface"

    def summarize_conversation(self, messages: List[Dict[str, str]], 
                             max_length: int = 1000) -> ConversationSummary:
        """Create a condensed summary of conversation context"""
        if not self.nlp:
            return ConversationSummary()

        try:
            # Process all messages
            docs = [self.nlp(msg["content"]) for msg in messages]
            original_length = sum(len(doc) for doc in docs)

            # Extract key sentences using transformer attention
            key_sentences = []
            for doc in docs:
                # Get attention scores from transformer
                attention = doc._.trf_data.attention[0].mean(axis=0).mean(axis=0)
                
                # Select sentences with highest attention scores
                sentences = list(doc.sents)
                sentence_scores = []
                for sent in sentences:
                    # Average attention for tokens in sentence
                    score = attention[sent.start:sent.end].mean()
                    sentence_scores.append((sent, score))
                
                # Keep top sentences
                sentence_scores.sort(key=lambda x: x[1], reverse=True)
                key_sentences.extend([s[0].text for s in sentence_scores[:2]])

            # Extract main topics using noun chunks and named entities
            topic_scores = defaultdict(float)
            for doc in docs:
                # Get noun chunks
                for chunk in doc.noun_chunks:
                    topic_scores[chunk.root.text] += 1
                
                # Get named entities
                for ent in doc.ents:
                    topic_scores[ent.text] += 2  # Weight entities higher
                    
                # Get important verbs
                for token in doc:
                    if token.pos_ == "VERB" and token.dep_ == "ROOT":
                        topic_scores[token.text] += 0.5

            # Normalize topic scores
            total = sum(topic_scores.values()) or 1
            topic_scores = {k: v/total for k, v in topic_scores.items()}

            # Extract argument chain
            argument_chain = []
            current_claim = None
            for doc in docs:
                # Look for claim indicators
                for sent in doc.sents:
                    if any(token.dep_ == "mark" for token in sent):
                        # This might be a claim
                        if current_claim:
                            # Look for support/opposition
                            relation = "supports" if any(
                                token.text.lower() in {"therefore", "thus", "because"}
                                for token in sent
                            ) else "opposes" if any(
                                token.text.lower() in {"however", "but", "although"}
                                for token in sent
                            ) else "relates"
                            
                            argument_chain.append({
                                "claim": current_claim,
                                "relation": relation,
                                "response": sent.text
                            })
                        current_claim = sent.text

            # Extract unresolved questions
            questions = []
            for doc in docs:
                for sent in doc.sents:
                    if sent.text.strip().endswith("?"):
                        # Check if question was answered
                        question_tokens = set(token.text.lower() for token in sent)
                        answered = False
                        for later_doc in docs[docs.index(doc)+1:]:
                            answer_tokens = set(token.text.lower() for token in later_doc)
                            if len(question_tokens & answer_tokens) / len(question_tokens) > 0.3:
                                answered = True
                                break
                        if not answered:
                            questions.append(sent.text)

            # Create final summary
            summary = ConversationSummary(
                key_points=key_sentences,
                main_topics={k: v for k, v in sorted(
                    topic_scores.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]},  # Top 10 topics
                argument_chain=argument_chain,
                unresolved_questions=questions,
                context_length=original_length,
                summary_length=sum(len(s.split()) for s in key_sentences)
            )

            # Verify we're under max length
            while summary.summary_length > max_length and summary.key_points:
                summary.key_points.pop()
                summary.summary_length = sum(len(s.split()) for s in summary.key_points)

            return summary
        except Exception as e:
            logger.error(f"Error in conversation summarization: {e}")
            return ConversationSummary()

    def ground_assertion(self, assertion: str) -> AssertionEvidence:
        """Ground a single assertion with evidence"""
        try:
            # Search for supporting evidence
            results = self.search_client.search(assertion, max_results=3)
            
            sources = []
            for result in results:
                sources.append({
                    "url": result.url,
                    "excerpt": result.snippet,
                    "title": result.title
                })
            
            # Calculate confidence using NLP if available
            confidence = self._calculate_confidence(sources, assertion)
            
            return AssertionEvidence(
                assertion=assertion,
                sources=sources,
                confidence=confidence,
                verification_method="web_search_nlp" if self.nlp else "web_search"
            )
        except Exception as e:
            logger.error(f"Error grounding assertion: {e}")
            return AssertionEvidence(
                assertion=assertion,
                sources=[],
                confidence=0.0,
                verification_method="failed"
            )
    
    def _calculate_confidence(self, 
                          sources: List[Dict[str, str]], 
                          assertion: str) -> float:
        """Calculate confidence score using NLP and source analysis"""
        if not sources:
            return 0.0
            
        try:
            # Use spaCy for semantic similarity if available
            if self.nlp:
                assertion_doc = self.nlp(assertion)
                similarities = []
                for source in sources:
                    source_doc = self.nlp(source["excerpt"])
                    similarity = assertion_doc.similarity(source_doc)
                    similarities.append(similarity)
                semantic_score = sum(similarities) / len(similarities)
            else:
                semantic_score = 0.5  # Default if no NLP
                
            # Source authority scoring
            authority_score = sum(0.2 if ".edu" in s["url"] or 
                                     ".gov" in s["url"] or 
                                     ".org" in s["url"] 
                              else 0.1 
                              for s in sources) / len(sources)
            
            # Combine scores with weights
            final_score = (
                semantic_score * 0.6 +
                authority_score * 0.4
            )
            
            return min(final_score, 1.0)
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0

class ConversationArbiter:
    """Evaluates and compares conversations using Gemini model with grounded analysis"""
    
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
        
        # Combine analyses
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
                        ai_ai_analysis.get("key_insights", []) +
                        human_ai_analysis.get("key_insights", []),
            improvement_suggestions=ai_ai_analysis.get("improvement_suggestions", []) +
                                  human_ai_analysis.get("improvement_suggestions", []),
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

    def _analyze_conversation(self, conversation: List[Dict[str, str]], 
                            goal: str, mode: str) -> Dict:
        """Analyze a single conversation with assertion grounding"""
        analysis = None
        # Create empty conversation for the other mode to maintain structure
        empty_conversation = []
        
        if mode == "ai-ai":
            prompt = self._create_evaluation_prompt(conversation, empty_conversation, goal, mode)
        else:
            prompt = self._create_evaluation_prompt(empty_conversation, conversation, goal, mode)
        
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
            if self.grounder and analysis:
                grounded_assertions = {}
                for assertion in analysis.get("assertions", []):
                    evidence = self.grounder.ground_assertion(assertion)
                    if evidence.confidence > 0.5:  # Only keep well-supported assertions
                        grounded_assertions[assertion] = evidence
                analysis["grounded_assertions"] = grounded_assertions
            
            # Add detailed NLP analysis for each message
            for role, metrics in analysis["participant_analysis"].items():
                message_analyses = [
                    self._analyze_message_nlp(msg["content"])
                    for msg in conversation if msg["role"] == role
                ]
                metrics["message_analyses"] = message_analyses

            # Add conversation summary
            analysis["conversation_summary"] = self.summarize_conversation(conversation)
            
            if not analysis:
                raise ValueError("Failed to parse model response into valid analysis format")
            
            return analysis
                
        except Exception as e:
            logger.error(f"Error analyzing conversation: {e}")
            return self._create_empty_analysis()

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

    def _create_evaluation_prompt(self, 
                                ai_ai_conversation: List[Dict[str, str]],
                                human_ai_conversation: List[Dict[str, str]],
                               goal: str,
                                mode: str) -> str:
        """Create a structured evaluation prompt for Gemini model"""
        prompt = f"""
        Evaluate the following conversations and provide detailed analysis on how well the conversations achieve the goal of discussing "{goal}".:

        AI-AI CONVERSATION:
        {self._format_conversation(ai_ai_conversation, "ai-ai")}

        HUMAN-AI CONVERSATION:
        {self._format_conversation(human_ai_conversation, "human-ai")}

        Your output should ground any claims made in the conversations and provide quantitative insights on the quality of the conversation, including coherence, depth, engagement, reasoning, knowledge exchange, goal progress, and strategy effectiveness.
        Addititionally, provide a summary of the conversation context, key insights, and suggestions for improvement.
        Finally determine the winner of each conversation based on the quality of the discussion and provide a table detailed performance metrics for each actor in the conversation and how well they performed their role in terms of:
        - Response quality
        - Knowledge accuracy
        - Reasoning depth
        - Engagement level
        - Strategy adherence
        - Adaptation
        - Factual accuracy
        - Citation quality
        """
        return prompt
        
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
                return template.format(report_content=self._generate_report_content(result))
        except Exception as e:
            logger.warning(f"Failed to load report template: {e}")
            return self._generate_report_content(result)

    def _generate_report_content(self, result: ArbiterResult) -> str:
        """Generate the HTML content for the report"""
        return f"""
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

    def _format_conversation_summary(self, summary: ConversationSummary) -> str:
        """Format conversation summary as HTML"""
        if not summary:
            return ""
            
        return f"""
        <div class="conversation-summary">
            <h3>Conversation Summary</h3>
            <p class="summary-stats">
                Reduced from {summary.context_length} to {summary.summary_length} tokens 
                ({(summary.summary_length/summary.context_length*100):.1f}% of original)
            </p>
            
            <div class="key-points">
                <h4>Key Points</h4>
                <ul>
                    {''.join(f"<li>{point}</li>" for point in summary.key_points)}
                </ul>
            </div>
            
            <div class="main-topics">
                <h4>Main Topics</h4>
                <ul>
                    {''.join(f"<li>{topic}: {score:.2f}</li>" 
                            for topic, score in summary.main_topics.items())}
                </ul>
            </div>
            
            <div class="unresolved-questions">
                <h4>Unresolved Questions</h4>
                <ul>
                    {''.join(f"<li>{q}</li>" for q in summary.unresolved_questions)}
                </ul>
            </div>
        </div>
        """

    def _format_nlp_analysis(self, message_analyses: List[MessageAnalysis]) -> str:
        """Format NLP analysis results as HTML"""
        if not message_analyses:
            return "<p>No detailed analysis available.</p>"
            
        html = ["<div class='nlp-analysis'>"]
        
        for i, analysis in enumerate(message_analyses, 1):
            html.append(f"""
                <div class='message-analysis'>
                    <h4>Message {i}</h4>
                    <div class='metrics'>
                        <p>Tokens: {analysis.tokens}</p>
                        <p>Sentences: {analysis.sentences}</p>
                        <p>Avg Sentence Length: {analysis.avg_sentence_length:.1f}</p>
                        <p>Readability: {analysis.readability_score:.2f}</p>
                        <p>Response Type: {analysis.response_type}</p>
                        <p>Discourse Level: {analysis.discourse_level}</p>
                    </div>
                    
                    <div class='entities'>
                        <h5>Named Entities</h5>
                        {self._format_dict(analysis.nlp_analysis.entities)}
                    </div>
                    
                    <div class='topics'>
                        <h5>Topic Clusters</h5>
                        {self._format_dict(analysis.nlp_analysis.topic_clusters)}
                    </div>
                </div>
            """)
            
        html.append("</div>")
        return "\n".join(html)
    
    def _format_dict(self, d: Dict) -> str:
        """Format dictionary as HTML list"""
        if not d:
            return "<p>None found.</p>"
        return "<ul>" + "".join(
            f"<li><strong>{k}:</strong> {', '.join(v) if isinstance(v, list) else v}</li>"
            for k, v in d.items()
        ) + "</ul>"

    def _format_conversation(self, conversation: List[Dict[str, str]], mode: str) -> str:
        """Format conversation messages for evaluation"""
        if not conversation:
            return "No conversation data available."
        
        formatted = []
        for i, msg in enumerate(conversation, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Handle content that might be a dictionary
            if isinstance(content, dict):
                if "response" in content:
                    content = content["response"]
                elif "text" in content:
                    content = content["text"]
                else:
                    content = str(content)  # Fallback to string representation
            
            formatted.append(f"Message {i} - {role.upper()}: {content}")
            
        return "\n\n".join(formatted)

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