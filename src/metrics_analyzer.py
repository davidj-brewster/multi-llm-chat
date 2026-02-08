"""Metrics and conversation flow analysis for AI Battle"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict


@dataclass
class TopicCluster:
    """A cluster of related messages"""

    messages: List[str]
    keywords: List[str]
    coherence: float
    timeline: List[int]  # Message indices in this cluster


@dataclass
class MessageMetrics:
    """Metrics for a single message"""

    length: int
    thinking_sections: int
    references_previous: bool
    question_count: int
    assertion_count: int
    code_blocks: int
    sentiment_score: float
    complexity_score: float
    response_time: Optional[float]
    topics: List[str]  # Keywords representing message topics


@dataclass
class ConversationMetrics:
    """Metrics for entire conversation"""

    total_messages: int
    avg_message_length: float
    avg_thinking_sections: float
    topic_coherence: float
    turn_taking_balance: float
    question_answer_ratio: float
    assertion_density: float
    avg_complexity: float
    avg_response_time: Optional[float]
    topic_clusters: Dict[str, Dict[str, float]]  # Cluster name -> metrics
    topic_evolution: List[Dict[str, float]]  # Timeline of topic strengths


class TopicAnalyzer:
    """Analyzes topics and their evolution in conversations"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )

    def identify_topics(self, messages: List[str]) -> List[TopicCluster]:
        """Identify topic clusters in messages using TF-IDF and DBSCAN"""
        # Clean messages
        cleaned_messages = [self._clean_message(msg) for msg in messages]

        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(cleaned_messages)

        # Calculate similarity matrix
        similarities = cosine_similarity(tfidf_matrix)
        distances = 1 - np.abs(similarities)
        # Ensure all distances are valid (not negative)
        distances = np.abs(distances)  # Use absolute values to ensure non-negative distances

        # Cluster messages
        try:
            clustering = DBSCAN(
                eps=0.4,  # Maximum distance between samples
                min_samples=2,  # Minimum cluster size
                metric="precomputed",  # Use pre-computed similarities
            ).fit(distances)
        except ValueError as e:
            # If we still get an error despite using absolute values, fall back to a simpler approach
            print(f"DBSCAN clustering failed: {e}")
            # Create a dummy clustering result with all points as noise
            clustering = type('obj', (object,), {
                'labels_': np.array([-1] * len(distances))
            })

        # Extract clusters
        clusters = []
        labels = clustering.labels_

        for label in set(labels):
            if label == -1:  # Skip noise
                continue

            # Get messages in this cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_messages = [messages[i] for i in cluster_indices]

            # Get cluster keywords
            cluster_tfidf = tfidf_matrix[cluster_indices]
            feature_names = self.vectorizer.get_feature_names_out()

            # Sum TF-IDF scores for each term across cluster
            cluster_scores = cluster_tfidf.sum(axis=0).A1
            top_indices = cluster_scores.argsort()[-5:][::-1]  # Top 5 terms
            keywords = [feature_names[i] for i in top_indices]

            # Calculate cluster coherence
            cluster_similarities = similarities[cluster_indices][:, cluster_indices]
            coherence = cluster_similarities.mean()

            clusters.append(
                TopicCluster(
                    messages=cluster_messages,
                    keywords=keywords,
                    coherence=coherence,
                    timeline=list(cluster_indices),
                )
            )

        return clusters

    def _clean_message(self, message: str) -> str:
        """Clean message text for topic analysis.

        Removes code blocks, thinking tags, HTML, URLs, and special characters.

        Args:
            message: The message text to clean

        Returns:
            str: Cleaned message text suitable for topic analysis
        """
        # Remove code blocks
        text = re.sub(r"```.*?```", "", message, flags=re.DOTALL)

        # Remove thinking tags
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)

        # Remove HTML
        text = re.sub(r"<[^>]+>", "", text)

        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # Remove special characters
        text = re.sub(r"[^\w\s]", "", text)

        return text.lower()

    def get_message_topics(self, message: str, all_messages: List[str]) -> List[str]:
        """Get topics for a single message.

        Extract the top topics from a message by analyzing its TF-IDF vector.

        Args:
            message: The message text to analyze
            all_messages: List of all messages in the conversation for context

        Returns:
            List[str]: Top 3 terms representing the message topics based on
                      TF-IDF scores
        """
        message = self._clean_message(message)
        # Add message to corpus and get its vector
        all_messages.append(message)
        tfidf_matrix = self.vectorizer.fit_transform(all_messages)
        message_vector = tfidf_matrix[-1]  # Last row is our message

        # Get top terms
        feature_names = self.vectorizer.get_feature_names_out()
        scores = message_vector.toarray()[0]
        top_indices = scores.argsort()[-3:][::-1]  # Top 3 terms

        return [feature_names[i] for i in top_indices]


class MetricsAnalyzer:
    """Analyzes conversation metrics and flow"""

    def __init__(self):
        self.topic_analyzer = TopicAnalyzer()
        self.conversation_graph = nx.DiGraph()

    def analyze_message(
        self, message: Dict[str, str], all_messages: List[str]
    ) -> MessageMetrics:
        """Extract metrics from a single message.

        Analyzes message content to extract metrics like length, thinking sections,
        question count, code blocks, references to previous messages, assertion count,
        complexity score, sentiment score, and topics.

        Args:
            message: Dictionary containing message data with 'content' key
            all_messages: List of all previous message contents for context

        Returns:
            MessageMetrics: Object containing all extracted metrics
        """
        content = message["content"]

        # Basic metrics
        length = len(content)
        thinking_sections = len(
            re.findall(r"<thinking>.*?</thinking>", content, re.DOTALL)
        )

        # Question analysis
        questions = len(re.findall(r"\?", content))

        # Code block analysis
        code_blocks = len(re.findall(r"```.*?```", content, re.DOTALL))

        # Reference analysis
        references_previous = any(
            ref in content.lower()
            for ref in [
                "you mentioned",
                "as you said",
                "earlier",
                "previously",
                "your point about",
                "you noted",
                "you suggested",
            ]
        )

        # Assertion counting
        assertions = len(re.findall(r"(?<=[.!?])\s+(?=[A-Z])", content))

        # Complexity scoring
        complexity_indicators = {
            "technical_terms": [
                "algorithm",
                "implementation",
                "architecture",
                "framework",
                "pattern",
                "design",
                "system",
                "component",
                "interface",
                "module",
                "library",
                "package",
                "function",
                "method",
                "class",
                "object",
                "variable",
                "constant",
                "parameter",
                "argument",
                "property",
                "attribute",
                "operation",
                "procedure",
                "process",
                "service",
                "resource",
                "database",
                "query",
                "transaction",
                "connection",
                "authentication",
                "authorization",
                "encryption",
                "decryption",
                "compression",
                "decompression",
                "serialization",
                "deserialization",
                "validation",
                "verification",
                "transformation",
                "conversion",
                "generation",
                "parsing",
                "formatting",
                "encoding",
                "decoding",
                "routing",
                "mapping",
                "filtering",
                "sorting",
                "searching",
                "indexing",
                "caching",
                "logging",
                "monitoring",
                "reporting",
                "auditing",
                "debugging",
                "testing",
                "benchmarking",
                "profiling",
                "optimization",
                "refactoring",
                "migration",
                "integration",
                "automation",
                "orchestration",
                "synchronization",
                "replication",
                "scaling",
                "load balancing",
                "failover",
                "recovery",
                "backup",
                "restore",
                "deployment",
                "installation",
                "configuration",
                "customization",
                "personalization",
                "localization",
                "internationalization",
                "accessibility",
                "usability",
                "reliability",
                "availability",
                "scalability",
                "performance",
                "security",
                "privacy",
                "compliance",
                "regulation",
                "governance",
                "management",
                "administration",
                "monitoring",
                "reporting",
                "analytics",
                "insights",
                "predictions",
                "recommendations",
                "decisions",
                "actions",
                "notifications",
                "alerts",
                "reminders",
            ],
            "reasoning_markers": [
                "because",
                "therefore",
                "thus",
                "consequently",
                "however",
                "although",
                "despite",
                "while",
                "unless",
                "except",
                "until",
                "otherwise",
                "instead",
                "meanwhile",
                "furthermore",
                "moreover",
                "nevertheless",
                "nonetheless",
                "regardless",
                "indeed",
                "certainly",
                "surely",
                "absolutely",
                "definitely",
                "undoubtedly",
                "clearly",
                "obviously",
                "apparently",
                "evidently",
                "presumably",
                "arguably",
                "possibly",
                "potentially",
                "likely",
                "probably",
                "maybe",
                "perhaps",
                "possibly",
                "conceivably",
                "hypothetically",
                "theoretically",
                "ideally",
                "practically",
                "realistically",
                "effectively",
                "efficiently",
                "productively",
                "successfully",
                "profitably",
                "beneficially",
                "advantageously",
                "favorably",
                "constructively",
                "positively",
            ],
            "analysis_terms": [
                "analyze",
                "evaluate",
                "compare",
                "consider",
                "examine",
                "investigate",
                "assess",
                "review",
                "study",
                "inspect",
                "explore",
                "scrutinize",
                "interpret",
                "understand",
                "comprehend",
                "grasp",
                "appreciate",
                "recognize",
                "realize",
                "acknowledge",
                "identify",
                "detect",
                "diagnose",
                "determine",
                "decide",
                "conclude",
                "infer",
                "deduce",
                "predict",
                "forecast",
                "anticipate",
                "expect",
                "speculate",
                "hypothesize",
                "theorize",
                "postulate",
                "propose",
                "suggest",
                "recommend",
                "advise",
                "counsel",
                "guide",
                "instruct",
                "teach",
                "educate",
                "inform",
                "notify",
                "alert",
                "remind",
                "warn",
                "caution",
                "prepare",
                "plan",
                "design",
                "develop",
                "create",
                "build",
                "construct",
                "formulate",
                "establish",
            ],
        }

        complexity_score = sum(
            len([1 for term in terms if term in content.lower()]) / len(terms)
            for terms in complexity_indicators.values()
        ) / len(complexity_indicators)

        # Simple sentiment analysis
        positive_terms = {
            "good",
            "great",
            "excellent",
            "helpful",
            "interesting",
            "important",
            "useful",
            "valuable",
            "beneficial",
            "advantage",
            "correct",
            "accurate",
            "true",
            "believe",
            "agree",
            "clear",
            "understandable",
            "easy",
            "novel",
            "well grounded",
            "insightful",
            "remarkable",
            "fresh",
            "innovative",
            "creative",
            "original",
            "ingenious",
            "brilliant",
            "smart",
            "intelligent",
            "wise",
            "clever",
            "astute",
            "sensible",
            "practical",
            "realistic",
            "feasible",
            "viable",
            "effective",
            "efficient",
            "productive",
            "successful",
            "profitable",
            "advantageous",
            "favorable",
            "constructive",
            "positive",
            "upbeat",
            "optimistic",
            "encouraging",
            "hopeful",
            "inspiring",
            "motivating",
            "stimulating",
            "exciting",
            "thrilling",
            "enjoyable",
            "fun",
            "pleasant",
            "satisfying",
            "fulfilling",
            "rewarding",
            "gratifying",
            "pleasurable",
            "delightful",
            "wonderful",
            "marvelous",
            "fabulous",
            "fantastic",
            "terrific",
            "awesome",
            "amazing",
            "incredible",
            "extraordinary",
            "astonishing",
            "astounding",
            "stunning",
            "breathtaking",
            "awe-inspiring",
            "jaw-dropping",
            "mind-blowing",
            "heartwarming",
            "touching",
            "moving",
            "uplifting",
            "inspirational",
            "supportive",
            "reassuring",
            "comforting",
            "soothing",
            "calming",
            "relaxing",
            "peaceful",
            "tranquil",
            "serene",
            "harmonious",
            "balanced",
            "centered",
            "grounded",
            "stable",
            "secure",
            "safe",
            "protected",
            "sheltered",
            "shielded",
            "defended",
            "guarded",
            "fortified",
            "strengthened",
            "empowered",
            "enhanced",
            "improved",
            "upgraded",
            "optimized",
            "perfected",
            "polished",
        }
        negative_terms = {
            "dubious",
            "ungrounded",
            "bad",
            "poor",
            "wrong",
            "incorrect",
            "confusing",
            "problematic",
            "inaccurate",
            "challenge",
            "false",
            "unbelievable",
            "erroneous",
            "mistaken",
            "hardly",
            "disagree",
            "dislike",
            "hate",
            "difficult",
            "complicated",
            "complex",
            "tricky",
            "unclear",
            "ambiguous",
            "vague",
            "obscure",
            "uncertain",
            "doubtful",
            "incomprehensible",
            "inconsistent",
            "contradictory",
            "incoherent",
            "illogical",
            "irrational",
            "absurd",
            "nonsense",
            "ridiculous",
            "stupid",
            "foolish",
            "silly",
            "unintelligible",
            "unreasonable",
            "unconvincing",
            "unpersuasive",
            "unsubstantiated",
            "unfounded",
            "unjustified",
            "unwarranted",
            "unreliable",
            "untrustworthy",
            "unethical",
            "immoral",
            "unacceptable",
            "inappropriate",
            "offensive",
            "harmful",
            "dangerous",
            "risky",
            "threatening",
            "scary",
            "frightening",
            "disturbing",
            "upsetting",
            "worrying",
            "alarming",
            "shocking",
            "surprising",
            "unexpected",
            "unpredictable",
            "unforeseen",
            "unanticipated",
            "unwanted",
            "undesirable",
            "unpleasant",
            "uncomfortable",
            "painful",
            "hurtful",
            "damaging",
            "destructive",
            "disruptive",
            "disastrous",
            "catastrophic",
            "devastating",
            "ruinous",
        }

        words = content.lower().split()
        sentiment_score = (
            (sum(1.0 for w in words if w in positive_terms) + 1)
            / (sum(1.0 for w in words if w in negative_terms) + 1)
        ) / (len(words) + 1)

        # Topic analysis
        topics = self.topic_analyzer.get_message_topics(content, all_messages)

        return MessageMetrics(
            length=length,
            thinking_sections=thinking_sections,
            references_previous=references_previous,
            question_count=questions,
            assertion_count=assertions,
            code_blocks=code_blocks,
            sentiment_score=sentiment_score,
            complexity_score=complexity_score,
            response_time=message.get("response_time"),
            topics=topics,
        )

    def analyze_conversation_flow(
        self, conversation: List[Dict[str, str]]
    ) -> nx.DiGraph:
        """Analyze conversation flow and create a directed graph representation.

        Creates a graph where nodes represent messages and edges represent the flow
        between messages. Edge weights are based on relevance between messages.
        Cross-references between non-adjacent messages are also detected and added
        as edges.

        Args:
            conversation: List of message dictionaries with 'role' and 'content' keys

        Returns:
            nx.DiGraph: Directed graph representing the conversation flow
        """
        G = nx.DiGraph()

        # Add nodes for each message
        for i, msg in enumerate(conversation):
            G.add_node(
                i,
                role=msg["role"],
                content_preview=msg["content"][:100],
                metrics=self.analyze_message(
                    msg, [m["content"] for m in conversation[:i]]
                ),
            )

        # Add edges for message flow
        for i in range(len(conversation) - 1):
            msg1 = conversation[i]
            msg2 = conversation[i + 1]

            # Calculate edge weight based on relevance
            weight = self._calculate_relevance(msg1["content"], msg2["content"])

            G.add_edge(i, i + 1, weight=weight)

            # Add cross-references if found
            if i > 0:
                for j in range(i - 1):
                    if self._has_reference(msg2["content"], conversation[j]["content"]):
                        G.add_edge(i + 1, j, weight=0.5, type="reference")

        return G

    def _calculate_relevance(self, msg1: str, msg2: str) -> float:
        """Calculate relevance between two messages.

        Uses Jaccard similarity of key terms.

        Args:
            msg1: First message text
            msg2: Second message text

        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        # Extract key terms
        terms1 = set(self._extract_key_terms(msg1))
        terms2 = set(self._extract_key_terms(msg2))

        # Calculate Jaccard similarity
        if not terms1 or not terms2:
            return 0.0

        return len(terms1 & terms2) / len(terms1 | terms2)

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text.

        Removes code blocks and thinking tags, then extracts words longer than
        3 characters that are not common stopwords.

        Args:
            text: The text to extract terms from

        Returns:
            List[str]: List of key terms (words longer than 3 characters
                      that aren't stopwords)
        """
        # Remove code blocks
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

        # Remove thinking tags
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)

        # Extract words
        words = re.findall(r"\b\w+\b", text.lower())

        # Remove common words
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to"}
        return [w for w in words if w not in stopwords and len(w) > 3]

    def _has_reference(self, msg: str, previous_msg: str) -> bool:
        """Check if message references a previous message.

        Detects references through direct reference phrases or by mentioning
        key terms from the previous message.

        Args:
            msg: The message to check for references
            previous_msg: The previous message that might be referenced

        Returns:
            bool: True if the message references the previous message,
                 False otherwise
        """
        key_terms = self._extract_key_terms(previous_msg)
        msg_lower = msg.lower()

        # Direct reference check
        if any(
            ref in msg_lower
            for ref in [
                "you mentioned",
                "as you said",
                "earlier",
                "previously",
                "your point about",
                "you noted",
                "you suggested",
            ]
        ):
            return True

        # Content reference check
        referenced_terms = sum(1 for term in key_terms if term in msg_lower)
        return referenced_terms >= 3

    def analyze_conversation(
        self, conversation: List[Dict[str, str]]
    ) -> ConversationMetrics:
        """Analyze an entire conversation to extract comprehensive metrics.

        Identifies topic clusters, tracks topic evolution, analyzes individual
        messages, and calculates aggregate metrics for the conversation as a whole.

        Args:
            conversation: List of message dictionaries with 'role' and 'content' keys

        Returns:
            ConversationMetrics: Object containing comprehensive metrics for the
                                entire conversation
        """
        # Get message contents
        messages = [msg["content"] for msg in conversation]

        # Identify topics
        topic_clusters = self.topic_analyzer.identify_topics(messages)

        # Track topic evolution
        topic_evolution = []
        for i in range(len(messages)):
            timepoint = {}
            for cluster in topic_clusters:
                # Calculate topic strength at this point
                strength = sum(1 for idx in cluster.timeline if idx <= i) / (i + 1)
                timepoint[" ".join(cluster.keywords[:2])] = strength
            topic_evolution.append(timepoint)

        # Analyze individual messages
        message_metrics = [
            self.analyze_message(msg, messages[:i])
            for i, msg in enumerate(conversation)
        ]

        # Calculate aggregate metrics
        total_messages = len(message_metrics)
        avg_message_length = sum(m.length for m in message_metrics) / total_messages
        avg_thinking_sections = (
            sum(m.thinking_sections for m in message_metrics) / total_messages
        )

        # Calculate turn-taking balance
        role_counts = defaultdict(int)
        for msg in conversation:
            role_counts[msg["role"]] += 1
        max_role_msgs = max(role_counts.values())
        min_role_msgs = min(role_counts.values())
        turn_taking_balance = (
            min_role_msgs / max_role_msgs if max_role_msgs > 0 else 1.0
        )

        # Calculate topic coherence
        topic_coherence = 0.0
        edges = 0
        for i in range(len(conversation) - 1):
            coherence = self._calculate_relevance(
                conversation[i]["content"], conversation[i + 1]["content"]
            )
            topic_coherence += coherence
            edges += 1
        topic_coherence = topic_coherence / edges if edges > 0 else 0.0

        # Question-answer analysis
        questions = sum(m.question_count for m in message_metrics)
        assertions = sum(m.assertion_count for m in message_metrics)
        question_answer_ratio = questions / assertions if assertions > 0 else 0.0

        # Complexity analysis
        avg_complexity = (
            sum(m.complexity_score for m in message_metrics) / total_messages
        )

        # Response time analysis
        response_times = [
            m.response_time for m in message_metrics if m.response_time is not None
        ]
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else None
        )

        return ConversationMetrics(
            total_messages=total_messages,
            avg_message_length=avg_message_length,
            avg_thinking_sections=avg_thinking_sections,
            topic_coherence=topic_coherence,
            turn_taking_balance=turn_taking_balance,
            question_answer_ratio=question_answer_ratio,
            assertion_density=assertions / total_messages,
            avg_complexity=avg_complexity,
            avg_response_time=avg_response_time,
            topic_clusters={
                " ".join(cluster.keywords): {
                    "messages": len(cluster.messages),
                    "coherence": cluster.coherence,
                }
                for cluster in topic_clusters
            },
            topic_evolution=topic_evolution,
        )

    def generate_flow_visualization(self, graph: nx.DiGraph) -> Dict:
        """Generate visualization data for conversation flow graph.

        Converts the NetworkX graph into a format suitable for visualization,
        including node positions, role information, content previews, and
        edge weights.

        Args:
            graph: NetworkX DiGraph representing the conversation flow

        Returns:
            Dict: Visualization data with 'nodes' and 'edges' keys
        """
        # Node positions using spring layout
        pos = nx.spring_layout(graph)

        # Node data
        nodes = [
            {
                "id": n,
                "role": graph.nodes[n]["role"],
                "preview": graph.nodes[n]["content_preview"],
                "x": float(pos[n][0]),
                "y": float(pos[n][1]),
                "metrics": {
                    k: abs(float(v)) if isinstance(v, (int, float)) else v
                    for k, v in graph.nodes[n]["metrics"].__dict__.items()
                },
            }
            for n in graph.nodes()
        ]

        # Edge data
        edges = [
            {
                "source": u,
                "target": v,
                "weight": abs(float(d["weight"])),
                "type": d.get("type", "flow"),
            }
            for u, v, d in graph.edges(data=True)
        ]

        return {"nodes": nodes, "edges": edges}


def analyze_conversations(
    ai_ai_conversation: List[Dict[str, str]],
    human_ai_conversation: List[Dict[str, str]],
) -> Dict:
    """
    Analyze and compare two conversations (AI-AI and Human-AI).

    Performs comprehensive analysis of both conversations, including metrics
    calculation and flow visualization. The results can be used to compare
    the quality and characteristics of AI-AI versus Human-AI interactions.

    Args:
        ai_ai_conversation: List of message dictionaries from AI-AI conversation
        human_ai_conversation: List of message dictionaries from Human-AI conversation

    Returns:
        Dict: Comparison results with 'metrics' and 'flow' data for both conversations
    """
    analyzer = MetricsAnalyzer()

    # Analyze each conversation
    ai_ai_metrics = analyzer.analyze_conversation(ai_ai_conversation)
    human_ai_metrics = analyzer.analyze_conversation(human_ai_conversation)

    # Generate flow visualizations
    ai_ai_flow = analyzer.analyze_conversation_flow(ai_ai_conversation)
    human_ai_flow = analyzer.analyze_conversation_flow(human_ai_conversation)

    return {
        "metrics": {
            "ai-ai": ai_ai_metrics.__dict__,
            "human-ai": human_ai_metrics.__dict__,
        },
        "flow": {
            "ai-ai": analyzer.generate_flow_visualization(ai_ai_flow),
            "human-ai": analyzer.generate_flow_visualization(human_ai_flow),
        },
    }
