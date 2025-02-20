import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#import spacy
import logging
from collections import Counter
import re

logger = logging.getLogger(__name__)

@dataclass
class ContextVector:
    """Multi-dimensional context analysis of conversation state"""
    semantic_coherence: float = 0.0  # How well responses relate to previous context
    topic_evolution: Dict[str, float] = field(default_factory=dict)  # Topic drift/focus tracking
    response_patterns: Dict[str, float] = field(default_factory=dict)  # Patterns in response styles
    engagement_metrics: Dict[str, float] = field(default_factory=dict)  # Interaction quality metrics
    cognitive_load: float = 0.0  # Complexity of current discussion
    knowledge_depth: float = 0.0  # Depth of domain understanding shown
    reasoning_patterns: Dict[str, float] = field(default_factory=dict)  # Types of reasoning used
    uncertainty_markers: Dict[str, float] = field(default_factory=dict)  # Confidence indicators

class ContextAnalyzer:
    """Analyzes conversation context across multiple dimensions"""
    
    def __init__(self, mode: str = "l"):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.nlp = None
        self.mode = mode
        try:
            import spacy
            self.nlp = spacy.load('en_core_web_sm')
        except (ImportError, OSError):
            logger.warning("spaCy not available, falling back to basic analysis")
            
        self.reasoning_patterns = {
            'deductive': r'therefore|thus|hence|consequently|as a result|it follows that|by definition',
            'inductive': r'generally|typically|usually|tends to|often|in most cases|frequently|commonly|regularly',
            'abductive': r'best explanation|most likely|probably because',
            'analogical': r'similar to|like|analogous|comparable',
            'causal': r'because|since|due to|results in'
        }

        self.ai_ai_patterns = {
            'formal_logic': r'axiom|theorem|proof|implies|given that|let|assume',
            'systematic': r'systematically|methodically|formally|structurally',
            'technical': r'implementation|specification|framework|architecture',
            'precision': r'precisely|specifically|explicitly|definitively',
            'integration': r'integrate|combine|synthesize|unify|merge'
        }
        
    def analyze(self, conversation_history: List[Dict[str, str]]) -> ContextVector:
        """Analyze conversation context across multiple dimensions"""
        
        # Extract just the content from conversation history
        contents = [msg['content'] for msg in conversation_history]
        
        return ContextVector(
            semantic_coherence=self._analyze_semantic_coherence(contents),
            topic_evolution=self._analyze_topic_drift(contents),
            response_patterns=self._analyze_response_patterns(conversation_history),
            engagement_metrics=self._calculate_engagement_metrics(conversation_history),
            cognitive_load=self._estimate_cognitive_load(contents),
            knowledge_depth=self._assess_knowledge_depth(contents),
            reasoning_patterns=self._analyze_reasoning_patterns(contents),
            uncertainty_markers=self._detect_uncertainty(contents)
        )
        
    def _analyze_semantic_coherence(self, contents: List[str]) -> float:
        """Measure how well responses relate to previous context"""
        if len(contents) < 3:
            return 1.0
            
        # Create TF-IDF matrix
        try:
            tfidf_matrix = self.vectorizer.fit_transform(contents)
            # Calculate average cosine similarity between consecutive responses
            similarities = []
            for i in range(len(contents)-1):
                similarity = cosine_similarity(
                    tfidf_matrix[i:i+1], 
                    tfidf_matrix[i+1:i+2]
                )[0][0]
                similarities.append(similarity)
            return np.mean(similarities)/2
        except Exception as e:
            logger.error(f"Error calculating semantic coherence: {e}")
            return 0.0
            
    def _analyze_topic_drift(self, contents: List[str]) -> Dict[str, float]:
        """Track evolution of topics over conversation"""
        topics = {}
        try:
            if self.nlp:
                # Use spaCy for sophisticated topic extraction
                for content in contents:
                    doc = self.nlp(content)
                    for chunk in doc.noun_chunks:
                        topic = chunk.root.text.lower()
                        topics[topic] = topics.get(topic, 0) + 1
            else:
                # Fallback: Use simple word frequency for nouns
                for content in contents:
                    # Split into words and filter for likely nouns (words longer than 3 chars)
                    words = [w.lower() for w in content.split() if len(w) > 3]
                    for word in words:
                        if not any(c.isdigit() for c in word):  # Skip numbers
                            topics[word] = topics.get(word, 0) + 1
            
            # Normalize counts to frequencies
            total = sum(topics.values()) or 1  # Avoid division by zero
            return {k: v/total for k, v in topics.items()}
        except Exception as e:
            logger.error(f"Error analyzing topic drift: {e}")
            return {}
            
    def _analyze_response_patterns(self, history: List[Dict[str, str]]) -> Dict[str, float]:
        """Analyze patterns in response styles"""
        patterns = {
            'question_frequency': 0.0,
            'elaboration_frequency': 0.0,
            'challenge_frequency': 0.0,
            'agreement_frequency': 0.0
        }
        
        if not history:
            return patterns
            
        try:
            for msg in history:
                content = msg['content'].lower()
                # Question patterns
                patterns['question_frequency'] += content.count('?')
                # Elaboration patterns
                elaboration_words = ['furthermore', 'moreover', 'additionally', 'in addition']
                patterns['elaboration_frequency'] += sum(content.count(word) for word in elaboration_words)
                # Challenge patterns
                challenge_words = ['however', 'but', 'although', 'disagree', 'incorrect']
                patterns['challenge_frequency'] += sum(content.count(word) for word in challenge_words)
                # Agreement patterns
                agreement_words = ['agree', 'yes', 'indeed', 'exactly', 'correct']
                patterns['agreement_frequency'] += sum(content.count(word) for word in agreement_words)
            
            # Normalize by message count (safe since we checked for empty history)
            msg_count = len(history)
            logger.info("Response patterns: {{k: v/msg_count for k, v in patterns.items()}}")
            return {k: v/msg_count for k, v in patterns.items()}
        except Exception as e:
            logger.error(f"Error analyzing response patterns: {e}")
            return patterns
            
    def _calculate_engagement_metrics(self, history: List[Dict[str, str]]) -> Dict[str, float]:
        """Calculate metrics for interaction quality"""
        metrics = {
            'avg_response_length': 0.0,
            'turn_taking_balance': 0.0,
            'response_time_consistency': 0.0
        }
        
        if not history:
            return metrics
            
        try:
            # Average response length
            lengths = [len(msg['content'].split()) for msg in history]
            metrics['avg_response_length'] = float(np.mean(lengths)) if lengths else 0.0
            
            # Turn-taking balance (ratio of human:AI responses)
            roles = [msg['role'] for msg in history]
            human_turns = sum(1 for role in roles if role == 'user')
            ai_turns = sum(1 for role in roles if role == 'assistant')
            
            # Calculate balance ensuring no division by zero
            if ai_turns > 0:
                metrics['turn_taking_balance'] = human_turns / ai_turns
            elif human_turns > 0:
                metrics['turn_taking_balance'] = float('inf')  # All human turns, no AI turns
            else:
                metrics['turn_taking_balance'] = 0.0  # No turns at all
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating engagement metrics: {e}")
            return metrics
            
    def _estimate_cognitive_load(self, contents: List[str]) -> float:
        """Estimate complexity of current discussion"""
        try:
            total_complexity = 0
            for content in contents[-3:]:  # Look at recent messages
                if self.nlp:
                    # Use spaCy for sophisticated analysis
                    doc = self.nlp(content)
                    
                    # Average sentence length
                    sent_lengths = [len([token for token in sent]) for sent in doc.sents]
                    avg_sent_length = np.mean(sent_lengths) if sent_lengths else 0
                    
                    # Vocabulary complexity (ratio of unique words)
                    tokens = [token.text.lower() for token in doc if not token.is_punct]
                    vocabulary_complexity = len(set(tokens)) / len(tokens) if tokens else 0
                else:
                    # Fallback to basic text analysis
                    # Estimate sentences by punctuation
                    sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
                    words = content.lower().split()
                    
                    # Average sentence length
                    avg_sent_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
                    
                    # Vocabulary complexity
                    vocabulary_complexity = len(set(words)) / len(words) if words else 0
                
                # Additional complexity indicators (works with or without spaCy)
                technical_indicators = len(re.findall(
                    r'\b(algorithm|function|parameter|variable|concept|theory|framework)\b',
                    content.lower()
                )) / 100.0  # Normalize technical terms
                
                # Combine metrics
                message_complexity = (
                    avg_sent_length * 0.3 +
                    vocabulary_complexity * 0.4 +
                    technical_indicators * 0.3
                )
                total_complexity += message_complexity
            
            return min(1.0, total_complexity / (3 * 2))  # Normalize to [0,1]
        except Exception as e:
            logger.error(f"Error estimating cognitive load: {e}")
            return 0.0
            
    def _assess_knowledge_depth(self, contents: List[str]) -> float:
        """Assess depth of domain understanding shown"""
        try:
            depth_score = 0
            for content in contents[-3:]:  # Focus on recent messages
                if self.nlp:
                    # Use spaCy for sophisticated analysis
                    doc = self.nlp(content)
                    technical_terms = len([token for token in doc
                                        if token.pos_ in ['NOUN', 'PROPN']])
                    term_density = technical_terms / len(doc)
                else:
                    # Fallback to basic text analysis
                    # Look for likely technical terms (capitalized words and known technical terms)
                    words = content.split()
                    technical_terms = len([w for w in words if (
                        w[0].isupper() or  # Capitalized words
                        w.lower() in {  # Common technical terms
                            'algorithm', 'function', 'method', 'theory',
                            'concept', 'framework', 'system', 'process',
                            'analysis', 'structure', 'pattern', 'model'
                        }
                    )])
                    term_density = technical_terms / len(words) if words else 0
                
                # Common analysis regardless of spaCy availability
                # Explanation patterns
                explanations = len(re.findall(
                    r'because|therefore|explains|means that|in other words',
                    content.lower()
                ))
                
                # Reference to concepts
                concept_references = len(re.findall(
                    r'concept|principle|theory|idea|approach|technique',
                    content.lower()
                ))
                
                # Interconnection markers
                interconnections = len(re.findall(
                    r'related to|connected with|linked to|associated with|depends on',
                    content.lower()
                ))
                
                # Combine metrics
                message_depth = (
                    term_density * 0.4 +
                    (explanations / 10) * 0.3 +  # Normalize explanations
                    (concept_references / 5) * 0.2 +  # Normalize concept references
                    (interconnections / 5) * 0.1  # Normalize interconnections
                )
                depth_score += message_depth
            
            return min(1.0, depth_score / 3)  # Normalize to [0,1]
        except Exception as e:
            logger.error(f"Error assessing knowledge depth: {e}")
            return 0.0
            
    def _analyze_reasoning_patterns(self, contents: List[str]) -> Dict[str, float]:
        """Analyze types of reasoning used"""
        pattern_counts = {pattern: 0.0 for pattern in self.reasoning_patterns}
        if self.mode == "ai-ai":
            pattern_counts.update({pattern: 0.0 for pattern in self.ai_ai_patterns})
        
        try:
            for content in contents:
                for pattern, regex in self.reasoning_patterns.items():
                    matches = len(re.findall(regex, content.lower()))
                    pattern_counts[pattern] += matches

                if self.mode == "ai-ai":
                    for pattern, regex in self.ai_ai_patterns.items():
                        matches = len(re.findall(regex, content.lower()))
                        pattern_counts[pattern] += matches
                    
            # Normalize counts
            total = sum(pattern_counts.values()) or 1  # Avoid division by zero
            normalized = {k: v/total for k, v in pattern_counts.items()}
            
            # For AI-AI mode, boost the weight of formal patterns
            if self.mode == "ai-ai":
                normalized = {k: v * 1.5 if k in self.ai_ai_patterns else v for k, v in normalized.items()}
            return normalized
        except Exception as e:
            logger.error(f"Error analyzing reasoning patterns: {e}")
            return pattern_counts
            
    def _detect_uncertainty(self, contents: List[str]) -> Dict[str, float]:
        """Detect markers of uncertainty or confidence"""
        markers = {
            'socratic': 0.0,
            'uncertainty': 0.0,
            'confidence': 0.0,
            'qualification': 0.0
        }
        
        try:
            socratic_patterns = r'what|why|how|where|when|who|which|whom|think|would it|could it|should it|may|might|can|could|would|should|will|is it|are they|are we|is there|are there|do you|does it|did it|have you|has it|had it|will it|won\'t it|can it|could it|should it|would it|isn\'t it|aren\'t they|isn\'t there|aren\'t there|don\'t you|doesn\'t it|didn\'t it|haven\'t you|hasn\'t it|hadn\'t it|won\'t it|can\'t it|couldn\'t it|shouldn\'t it|wouldn\'t it'
            uncertainty_patterns = r'maybe|might|could|unsure|potentially|theoretically|probably'
            confidence_patterns = r'definitely|certainly|clearly|obviously|undoubtedly|even if|regardless|always'
            qualification_patterns = r'possibly|however|though|except|unless|only if'
            
            for content in contents[-3:]:  # Focus on recent messages
                markers['socratic'] += len(re.findall(socratic_patterns, 
                                                       content.lower()))
                markers['uncertainty'] += len(re.findall(uncertainty_patterns, 
                                                       content.lower()))
                markers['confidence'] += len(re.findall(confidence_patterns, 
                                                      content.lower()))
                markers['qualification'] += len(re.findall(qualification_patterns,
                                                         content.lower()))
                
            # Normalize by message count
            logger.info("_detect_uncertainty: " + ''.join(contents) + f": {markers}")
            return {k: v/4 for k, v in markers.items()}
        except Exception as e:
            logger.error(f"Error detecting uncertainty: {e}")
            return markers

# Example usage
if __name__ == "__main__":
    # Test conversation
    conversation = [
        {"role": "user", "content": "What are the key principles of machine learning?"},
        {"role": "assistant", "content": "Machine learning principles include: 1) Data quality is crucial 2) Avoid overfitting 3) Feature selection matters"},
        {"role": "user", "content": "Can you elaborate on overfitting?"},
        {"role": "assistant", "content": "Overfitting occurs when a model learns noise in training data, performing well on training but poorly on new data."}
    ]
    
    analyzer = ContextAnalyzer()
    context = analyzer.analyze(conversation)
    print("Context Analysis Results:")
    print(f"Semantic Coherence: {context.semantic_coherence:.2f}")
    print(f"Topic Evolution: {dict(context.topic_evolution)}")
    print(f"Response Patterns: {dict(context.response_patterns)}")
    print(f"Cognitive Load: {context.cognitive_load:.2f}")
    print(f"Knowledge Depth: {context.knowledge_depth:.2f}")
    print(f"Reasoning Patterns: {dict(context.reasoning_patterns)}")
    print(f"Uncertainty Markers: {dict(context.uncertainty_markers)}")