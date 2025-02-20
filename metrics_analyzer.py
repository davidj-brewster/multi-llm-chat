"""Metrics and conversation flow analysis for AI Battle"""
import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter, defaultdict

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
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def identify_topics(self, messages: List[str]) -> List[TopicCluster]:
        """Identify topic clusters in messages using TF-IDF and DBSCAN"""
        # Clean messages
        cleaned_messages = [self._clean_message(msg) for msg in messages]
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(cleaned_messages)
        
        # Calculate similarity matrix
        similarities = cosine_similarity(tfidf_matrix)
        
        # Cluster messages
        clustering = DBSCAN(
            eps=0.3,  # Maximum distance between samples
            min_samples=2,  # Minimum cluster size
            metric='precomputed'  # Use pre-computed similarities
        ).fit(1 - similarities)  # Convert similarities to distances
        
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
            
            clusters.append(TopicCluster(
                messages=cluster_messages,
                keywords=keywords,
                coherence=coherence,
                timeline=list(cluster_indices)
            ))
        
        return clusters
    
    def _clean_message(self, message: str) -> str:
        """Clean message text for topic analysis"""
        # Remove code blocks
        text = re.sub(r'```.*?```', '', message, flags=re.DOTALL)
        
        # Remove thinking tags
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
        
        # Remove HTML
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.lower()
    
    def get_message_topics(self, message: str, all_messages: List[str]) -> List[str]:
        """Get topics for a single message"""
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
        
    def analyze_message(self, message: Dict[str, str], 
                       all_messages: List[str]) -> MessageMetrics:
        """Extract metrics from a single message"""
        content = message["content"]
        
        # Basic metrics
        length = len(content)
        thinking_sections = len(re.findall(r'<thinking>.*?</thinking>', content, re.DOTALL))
        
        # Question analysis
        questions = len(re.findall(r'\?', content))
        
        # Code block analysis
        code_blocks = len(re.findall(r'```.*?```', content, re.DOTALL))
        
        # Reference analysis
        references_previous = any(ref in content.lower() for ref in [
            "you mentioned", "as you said", "earlier", "previously",
            "your point about", "you noted", "you suggested"
        ])
        
        # Assertion counting
        assertions = len(re.findall(r'(?<=[.!?])\s+(?=[A-Z])', content))
        
        # Complexity scoring
        complexity_indicators = {
            "technical_terms": [
                "algorithm", "implementation", "architecture", "framework",
                "pattern", "design", "system", "component", "interface"
            ],
            "reasoning_markers": [
                "because", "therefore", "thus", "consequently",
                "however", "although", "despite", "while"
            ],
            "analysis_terms": [
                "analyze", "evaluate", "compare", "consider",
                "examine", "investigate", "assess"
            ]
        }
        
        complexity_score = sum(
            len([1 for term in terms if term in content.lower()]) / len(terms)
            for terms in complexity_indicators.values()
        ) / len(complexity_indicators)
        
        # Simple sentiment analysis
        positive_terms = {"good", "great", "excellent", "helpful", "interesting", "important"}
        negative_terms = {"bad", "poor", "wrong", "incorrect", "confusing", "problematic"}
        
        words = content.lower().split()
        sentiment_score = (
            sum(1 for w in words if w in positive_terms) -
            sum(1 for w in words if w in negative_terms)
        ) / len(words)
        
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
            topics=topics
        )
    
    def analyze_conversation_flow(self, 
                                conversation: List[Dict[str, str]]) -> nx.DiGraph:
        """Analyze conversation flow and create graph"""
        G = nx.DiGraph()
        
        # Add nodes for each message
        for i, msg in enumerate(conversation):
            G.add_node(i, 
                      role=msg["role"],
                      content_preview=msg["content"][:100],
                      metrics=self.analyze_message(msg, 
                                                 [m["content"] for m in conversation[:i]]))
        
        # Add edges for message flow
        for i in range(len(conversation)-1):
            msg1 = conversation[i]
            msg2 = conversation[i+1]
            
            # Calculate edge weight based on relevance
            weight = self._calculate_relevance(msg1["content"], msg2["content"])
            
            G.add_edge(i, i+1, weight=weight)
            
            # Add cross-references if found
            if i > 0:
                for j in range(i-1):
                    if self._has_reference(msg2["content"], conversation[j]["content"]):
                        G.add_edge(i+1, j, weight=0.5, type="reference")
        
        return G
    
    def _calculate_relevance(self, msg1: str, msg2: str) -> float:
        """Calculate relevance between two messages"""
        # Extract key terms
        terms1 = set(self._extract_key_terms(msg1))
        terms2 = set(self._extract_key_terms(msg2))
        
        # Calculate Jaccard similarity
        if not terms1 or not terms2:
            return 0.0
            
        return len(terms1 & terms2) / len(terms1 | terms2)
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # Remove thinking tags
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common words
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to"}
        return [w for w in words if w not in stopwords and len(w) > 3]
    
    def _has_reference(self, msg: str, previous_msg: str) -> bool:
        """Check if message references a previous message"""
        key_terms = self._extract_key_terms(previous_msg)
        msg_lower = msg.lower()
        
        # Direct reference check
        if any(ref in msg_lower for ref in [
            "you mentioned", "as you said", "earlier", "previously",
            "your point about", "you noted", "you suggested"
        ]):
            return True
            
        # Content reference check
        referenced_terms = sum(1 for term in key_terms if term in msg_lower)
        return referenced_terms >= 3
    
    def analyze_conversation(self, 
                           conversation: List[Dict[str, str]]) -> ConversationMetrics:
        """Analyze entire conversation"""
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
                timepoint[' '.join(cluster.keywords[:2])] = strength
            topic_evolution.append(timepoint)
        
        # Analyze individual messages
        message_metrics = [
            self.analyze_message(msg, messages[:i])
            for i, msg in enumerate(conversation)
        ]
        
        # Calculate aggregate metrics
        total_messages = len(message_metrics)
        avg_message_length = sum(m.length for m in message_metrics) / total_messages
        avg_thinking_sections = sum(m.thinking_sections for m in message_metrics) / total_messages
        
        # Calculate turn-taking balance
        role_counts = defaultdict(int)
        for msg in conversation:
            role_counts[msg["role"]] += 1
        max_role_msgs = max(role_counts.values())
        min_role_msgs = min(role_counts.values())
        turn_taking_balance = min_role_msgs / max_role_msgs if max_role_msgs > 0 else 1.0
        
        # Calculate topic coherence
        topic_coherence = 0.0
        edges = 0
        for i in range(len(conversation)-1):
            coherence = self._calculate_relevance(
                conversation[i]["content"],
                conversation[i+1]["content"]
            )
            topic_coherence += coherence
            edges += 1
        topic_coherence = topic_coherence / edges if edges > 0 else 0.0
        
        # Question-answer analysis
        questions = sum(m.question_count for m in message_metrics)
        assertions = sum(m.assertion_count for m in message_metrics)
        question_answer_ratio = questions / assertions if assertions > 0 else 0.0
        
        # Complexity analysis
        avg_complexity = sum(m.complexity_score for m in message_metrics) / total_messages
        
        # Response time analysis
        response_times = [m.response_time for m in message_metrics if m.response_time is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else None
        
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
                ' '.join(cluster.keywords): {
                    'messages': len(cluster.messages),
                    'coherence': cluster.coherence
                }
                for cluster in topic_clusters
            },
            topic_evolution=topic_evolution
        )
    
    def generate_flow_visualization(self, graph: nx.DiGraph) -> Dict:
        """Generate visualization data for conversation flow"""
        # Node positions using spring layout
        pos = nx.spring_layout(graph)
        
        # Node data
        nodes = [{
            "id": n,
            "role": graph.nodes[n]["role"],
            "preview": graph.nodes[n]["content_preview"],
            "x": float(pos[n][0]),
            "y": float(pos[n][1]),
            "metrics": {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in graph.nodes[n]["metrics"].__dict__.items()
            }
        } for n in graph.nodes()]
        
        # Edge data
        edges = [{
            "source": u,
            "target": v,
            "weight": float(d["weight"]),
            "type": d.get("type", "flow")
        } for u, v, d in graph.edges(data=True)]
        
        return {
            "nodes": nodes,
            "edges": edges
        }

def analyze_conversations(ai_ai_conversation: List[Dict[str, str]],
                        human_ai_conversation: List[Dict[str, str]]) -> Dict:
    """Analyze and compare two conversations"""
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
            "human-ai": human_ai_metrics.__dict__
        },
        "flow": {
            "ai-ai": analyzer.generate_flow_visualization(ai_ai_flow),
            "human-ai": analyzer.generate_flow_visualization(human_ai_flow)
        }
    }