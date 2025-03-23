import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
import traceback

from shared_resources import (SpacyModelSingleton, VectorizerSingleton,
                          MemoryManager)

class ContextAnalysisError(Exception):
    """Base exception for context analysis errors."""
    pass

class SemanticAnalysisError(ContextAnalysisError):
    """Raised when there's an error analyzing semantic coherence."""
    pass

class TopicAnalysisError(ContextAnalysisError):
    """Raised when there's an error analyzing topic drift."""
    pass

class PatternAnalysisError(ContextAnalysisError):
    """Raised when there's an error analyzing response patterns."""
    pass

class EngagementAnalysisError(ContextAnalysisError):
    """Raised when there's an error calculating engagement metrics."""
    pass

class CognitiveLoadAnalysisError(ContextAnalysisError):
    """Raised when there's an error estimating cognitive load."""
    pass

class KnowledgeDepthAnalysisError(ContextAnalysisError):
    """Raised when there's an error assessing knowledge depth."""
    pass

class ReasoningPatternAnalysisError(ContextAnalysisError):
    """Raised when there's an error analyzing reasoning patterns."""
    pass

class UncertaintyAnalysisError(ContextAnalysisError):
    """Raised when there's an error detecting uncertainty markers."""
    pass

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
        self.vectorizer = VectorizerSingleton.get_instance()
        self.nlp = SpacyModelSingleton.get_instance()
        self.mode = mode
        
        # Log memory usage after initialization
        if logger.isEnabledFor(logging.DEBUG):
            logger.warning("spaCy not available, falling back to basic analysis")
            logger.debug(MemoryManager.get_memory_usage())
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
        
        try:
            # Validate input
            if not isinstance(conversation_history, list):
                raise ValueError(f"Expected list for conversation_history, got {type(conversation_history)}")
            
            # Extract content from messages
            try:
                contents = [msg['content'] for msg in conversation_history]
            except (KeyError, TypeError) as e:
                logger.error(f"Invalid message format in conversation history: {e}")
                # Create a fallback with empty strings for invalid messages
                contents = []
                for msg in conversation_history:
                    try:
                        contents.append(msg.get('content', ''))
                    except (AttributeError, TypeError):
                        contents.append('')
                
            # Create context vector with error handling for each component
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
        except ValueError as e:
            logger.error(f"Invalid input for context analysis: {e}")
            # Return a default context vector with zeros
            return ContextVector()
        except Exception as e:
            logger.error(f"Unexpected error in context analysis: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            # Return a default context vector with zeros
            return ContextVector()
            
        
    def _analyze_semantic_coherence(self, contents: List[str]) -> float:
        """Measure how well responses relate to previous context"""
        if len(contents) < 3:
            return 1.0
            
        # Create TF-IDF matrix
        try:
            try:
                # Ensure contents are strings
                valid_contents = [str(content) for content in contents[-5:]]
                
                # Check if we have enough content to analyze
                if not valid_contents or all(not content.strip() for content in valid_contents):
                    logger.warning("No valid content for semantic coherence analysis")
                    return 0.0
                
                tfidf_matrix = self.vectorizer.fit_transform(valid_contents)  # Only analyze recent messages
                
                # Calculate average cosine similarity between consecutive responses
                similarities = []
                for i in range(len(valid_contents)-1):
                    similarity = cosine_similarity(
                        tfidf_matrix[i:i+1], 
                        tfidf_matrix[i+1:i+2]
                    )[0][0]
                    similarities.append(similarity)
                return np.mean(similarities)/2 if similarities else 0.0
            except (ValueError, TypeError) as e:
                logger.error(f"Error in TF-IDF processing: {e}")
                raise SemanticAnalysisError(f"Error in TF-IDF processing: {e}")
        except Exception as e:
            logger.error(f"Error calculating semantic coherence: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            raise SemanticAnalysisError(f"Error calculating semantic coherence: {e}")
            
    def _analyze_topic_drift(self, contents: List[str]) -> Dict[str, float]:
        """Track evolution of topics over conversation"""
        topics = {}
        try:
            if self.nlp:
                # Use spaCy for topic extraction (limited to recent messages)
                try:
                    for content in contents:
                        doc = self.nlp(str(content))
                        for chunk in doc.noun_chunks:
                            topic = chunk.root.text.lower()
                            topics[topic] = topics.get(topic, 0) + 1
                except (AttributeError, TypeError) as e:
                    logger.error(f"Error in spaCy processing: {e}")
                    raise TopicAnalysisError(f"Error in spaCy processing: {e}")
            else:
                # Fallback: Use simple word frequency for nouns
                try:
                    for content in contents:
                        # Split into words and filter for likely nouns (words longer than 3 chars)
                        words = [w.lower() for w in str(content).split() if len(w) > 3]
                        for word in words:
                            if not any(c.isdigit() for c in word):  # Skip numbers
                                topics[word] = topics.get(word, 0) + 1
                except (AttributeError, TypeError) as e:
                    logger.error(f"Error in basic text processing: {e}")
                    raise TopicAnalysisError(f"Error in basic text processing: {e}")
            
            # Normalize counts to frequencies
            try:
                total = sum(topics.values()) or 1  # Avoid division by zero
                return {k: v/total for k, v in topics.items()}
            except (TypeError, ZeroDivisionError) as e:
                logger.error(f"Error normalizing topic frequencies: {e}")
                raise TopicAnalysisError(f"Error normalizing topic frequencies: {e}")
        except Exception as e:
            logger.error(f"Error analyzing topic drift: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            raise TopicAnalysisError(f"Error analyzing topic drift: {e}")
            
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
            try:
                for msg in history:
                    try:
                        content = str(msg.get('content', '')).lower()
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
                    except (AttributeError, TypeError) as e:
                        logger.warning(f"Skipping invalid message in response pattern analysis: {e}")
                        continue
                
                # Normalize by message count (safe since we checked for empty history)
                try:
                    msg_count = len(history)
                    logger.debug("Response patterns: {{k: v/msg_count for k, v in patterns.items()}}")
                    return {k: v/msg_count for k, v in patterns.items()}
                except ZeroDivisionError:
                    logger.error("Division by zero in response pattern normalization (empty history)")
                    return patterns
            except (KeyError, AttributeError, TypeError) as e:
                logger.error(f"Error processing messages for response patterns: {e}")
                raise PatternAnalysisError(f"Error processing messages for response patterns: {e}")
        except Exception as e:
            logger.error(f"Error analyzing response patterns: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            raise PatternAnalysisError(f"Error analyzing response patterns: {e}")
            
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
            try:
                # Average response length
                lengths = []
                for msg in history:
                    try:
                        content = str(msg.get('content', ''))
                        lengths.append(len(content.split()))
                    except (AttributeError, TypeError) as e:
                        logger.warning(f"Skipping invalid message in engagement metrics: {e}")
                        continue
                
                metrics['avg_response_length'] = float(np.mean(lengths)) if lengths else 0.0
                
                # Turn-taking balance (ratio of human:AI responses)
                try:
                    roles = [msg.get('role', '') for msg in history]
                    human_turns = sum(1 for role in roles if role == 'user')
                    ai_turns = sum(1 for role in roles if role == 'assistant')
                    
                    # Calculate balance ensuring no division by zero
                    if ai_turns > 0:
                        metrics['turn_taking_balance'] = human_turns / ai_turns
                    elif human_turns > 0:
                        metrics['turn_taking_balance'] = float('inf')  # All human turns, no AI turns
                    else:
                        metrics['turn_taking_balance'] = 0.0  # No turns at all
                except (AttributeError, TypeError) as e:
                    logger.error(f"Error calculating turn-taking balance: {e}")
                    metrics['turn_taking_balance'] = 0.0
                
                return metrics
            except (KeyError, AttributeError, TypeError) as e:
                logger.error(f"Error processing messages for engagement metrics: {e}")
                raise EngagementAnalysisError(f"Error processing messages for engagement metrics: {e}")
        except Exception as e:
            logger.error(f"Error calculating engagement metrics: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            raise EngagementAnalysisError(f"Error calculating engagement metrics: {e}")
            
    def _estimate_cognitive_load(self, contents: List[str]) -> float:
        """Estimate complexity of current discussion"""
        try:
            total_complexity = 0
            for content in contents[-2:]:  # Reduced from -3 to -2 for memory efficiency
                if self.nlp:
                    try:
                        # Use spaCy for sophisticated analysis
                        doc = self.nlp(str(content))
                        
                        # Average sentence length
                        sent_lengths = [len([token for token in sent]) for sent in doc.sents]
                        avg_sent_length = np.mean(sent_lengths) if sent_lengths else 0
                        
                        # Vocabulary complexity (ratio of unique words)
                        tokens = [token.text.lower() for token in doc if not token.is_punct]
                        vocabulary_complexity = len(set(tokens)) / len(tokens) if tokens else 0
                    except (AttributeError, TypeError, ZeroDivisionError) as e:
                        logger.error(f"Error in spaCy cognitive load analysis: {e}")
                        # Fallback to basic analysis
                        avg_sent_length = 0
                        vocabulary_complexity = 0
                else:
                    try:
                        # Fallback to basic text analysis
                        content_str = str(content)
                        # Estimate sentences by punctuation
                        sentences = [s.strip() for s in re.split(r'[.!?]+', content_str) if s.strip()]
                        words = content_str.lower().split()
                        
                        # Average sentence length
                        avg_sent_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
                        
                        # Vocabulary complexity
                        vocabulary_complexity = len(set(words)) / len(words) if words else 0
                    except (AttributeError, TypeError, ZeroDivisionError) as e:
                        logger.error(f"Error in basic cognitive load analysis: {e}")
                        avg_sent_length = 0
                        vocabulary_complexity = 0
                
                # Additional complexity indicators (works with or without spaCy)
                try:
                    technical_indicators = len(re.findall(
                        r'\b(algorithm|function|parameter|variable|concept|theory|framework)\b',
                        str(content).lower()
                    )) / 100.0  # Normalize technical terms
                except (AttributeError, TypeError) as e:
                    logger.error(f"Error in technical indicators analysis: {e}")
                    technical_indicators = 0
                
                # Combine metrics
                try:
                    message_complexity = (
                        avg_sent_length * 0.3 +
                        vocabulary_complexity * 0.4 +
                        technical_indicators * 0.3
                    )
                    total_complexity += message_complexity
                except (TypeError, ValueError) as e:
                    logger.error(f"Error combining complexity metrics: {e}")
                    # Skip this message
            
            return min(1.0, total_complexity / (3 * 2))  # Normalize to [0,1]
        except Exception as e:
            logger.error(f"Error estimating cognitive load: {e}")
            logger.debug(f"Stack trace for cognitive load error: {traceback.format_exc()}")
            return 0.0  # Return default value for this metric as it's not critical
            
    def _assess_knowledge_depth(self, contents: List[str]) -> float:
        """Assess depth of domain understanding shown"""
        try:
            depth_score = 0
            for content in contents[-2:]:  # Reduced from -3 to -2 for memory efficiency
                if self.nlp:
                    try:
                        # Use spaCy for sophisticated analysis
                        doc = self.nlp(str(content))
                        technical_terms = len([token for token in doc
                                            if token.pos_ in ['NOUN', 'PROPN']])
                        term_density = technical_terms / len(doc) if len(doc) > 0 else 0
                    except (AttributeError, TypeError, ZeroDivisionError) as e:
                        logger.error(f"Error in spaCy knowledge depth analysis: {e}")
                        # Fallback to basic analysis
                        term_density = 0
                else:
                    try:
                        # Fallback to basic text analysis
                        content_str = str(content)
                        # Look for likely technical terms (capitalized words and known technical terms)
                        words = content_str.split()
                        technical_terms = len([w for w in words if (
                            w and w[0].isupper() or  # Capitalized words
                            w.lower() in {  # Common technical terms
                                'algorithm', 'function', 'method', 'theory',
                                'concept', 'framework', 'system', 'process',
                                'analysis', 'structure', 'pattern', 'model'
                            }
                        )])
                        term_density = technical_terms / len(words) if words else 0
                    except (AttributeError, TypeError, IndexError, ZeroDivisionError) as e:
                        logger.error(f"Error in basic knowledge depth analysis: {e}")
                        term_density = 0
                
                # Common analysis regardless of spaCy availability
                try:
                    content_str = str(content).lower()
                    # Explanation patterns
                    explanations = len(re.findall(
                        r'because|therefore|explains|means that|in other words',
                        content_str
                    ))
                    
                    # Reference to concepts
                    concept_references = len(re.findall(
                        r'concept|principle|theory|idea|approach|technique',
                        content_str
                    ))
                    
                    # Interconnection markers
                    interconnections = len(re.findall(
                        r'related to|connected with|linked to|associated with|depends on',
                        content_str
                    ))
                except (AttributeError, TypeError) as e:
                    logger.error(f"Error in pattern analysis for knowledge depth: {e}")
                    explanations = 0
                    concept_references = 0
                    interconnections = 0
                
                # Combine metrics
                try:
                    message_depth = (
                        term_density * 0.4 +
                        (explanations / 10) * 0.3 +  # Normalize explanations
                        (concept_references / 5) * 0.2 +  # Normalize concept references
                        (interconnections / 5) * 0.1  # Normalize interconnections
                    )
                    depth_score += message_depth
                except (TypeError, ValueError) as e:
                    logger.error(f"Error combining knowledge depth metrics: {e}")
                    # Skip this message
            
            return min(1.0, depth_score / 3)  # Normalize to [0,1]
        except Exception as e:
            logger.error(f"Error assessing knowledge depth: {e}")
            logger.debug(f"Stack trace for knowledge depth error: {traceback.format_exc()}")
            return 0.0  # Return default value for this metric as it's not critical
            
    def _analyze_reasoning_patterns(self, contents: List[str]) -> Dict[str, float]:
        """Analyze types of reasoning used"""
        pattern_counts = {pattern: 0.0 for pattern in self.reasoning_patterns}
        if self.mode == "ai-ai":
            pattern_counts.update({pattern: 0.0 for pattern in self.ai_ai_patterns})
        
        try:
            try:
                for content in contents:
                    content_str = str(content or "").lower()
                    for pattern, regex in self.reasoning_patterns.items():
                        try:
                            matches = len(re.findall(regex, content_str))
                            pattern_counts[pattern] += matches
                        except (re.error, TypeError) as e:
                            logger.error(f"Error in regex pattern '{pattern}': {e}")
                            continue

                    if self.mode == "ai-ai":
                        for pattern, regex in self.ai_ai_patterns.items():
                            try:
                                matches = len(re.findall(regex, content_str))
                                pattern_counts[pattern] += matches
                            except (re.error, TypeError) as e:
                                logger.error(f"Error in AI-AI regex pattern '{pattern}': {e}")
                                continue
                        
                # Normalize counts
                total = sum(pattern_counts.values()) or 1  # Avoid division by zero
                normalized = {k: v/total for k, v in pattern_counts.items()}
                return normalized  # Return the normalized counts
            except Exception as e:
                logger.info(f"Error processing reasoning patterns: {e}")
                return pattern_counts
        except Exception as e:
            logger.error(f"Error analyzing reasoning patterns: {e}", exc_info=True)
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
            socratic_patterns = r'or did|interested|intrigued|conclusions|interpret|analysis|reason|suggest|think|believe|perspective|propose|consider|counter|question'
            uncertainty_patterns = r'maybe|might|could|unsure|potentially|theoretically|probably|questionably|questionable|debatably|supposed to|allegedly|according to some|would have you believe|more to it|unclear|doubtful|vague|sceptical'
            confidence_patterns = r'confident|obvious|absolutely|clearly|definitely|certainly|undoubtedly|even if|regardless|always|very|never|always|impossible|inevitable|doubtless|inevitable'
            qualification_patterns = r'maintaining|status-quo|conflicting|possibly|however|though|except|unless|only if|perhaps|one day|in the future|in the long term'
            
            for content in contents[-3:]:  # Focus on recent messages
                try:
                    # Ensure content is a string
                    content_str = str(content or "").lower()
                    
                    markers['socratic'] += len(re.findall(socratic_patterns, content_str))
                    markers['uncertainty'] += len(re.findall(uncertainty_patterns, content_str))
                    markers['confidence'] += len(re.findall(confidence_patterns, content_str))
                    markers['qualification'] += len(re.findall(qualification_patterns, content_str))
                except (TypeError, AttributeError) as e:
                    logger.warning(f"Error processing content for uncertainty detection: {e}")
                    continue
                except re.error as e:
                    logger.error(f"Regex error in uncertainty detection: {e}")
                    continue
                
            # Normalize by message count
            logger.debug("_detect_uncertainty: " + ''.join(contents) + f": {markers}")
            return {k: v/4 for k, v in markers.items()}
        except Exception as e:
            logger.error(f"Error detecting uncertainty: {e}")
            return markers

