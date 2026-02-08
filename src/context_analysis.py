"""
Context Analysis Engine for Multi-LLM Conversations.

Analyzes conversation history to produce actionable signals for the adaptive
instruction system. Measures human-ness, detects conversational pathologies,
and provides per-participant breakdowns.

Uses spaCy (when available) + scikit-learn for NLP analysis.
All signals have functional fallbacks when spaCy is absent.
"""

import math
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from shared_resources import SpacyModelSingleton, VectorizerSingleton

logger = logging.getLogger(__name__)


class ContextAnalysisError(Exception):
    """Raised when context analysis fails."""


@dataclass
class ContextVector:
    """Multi-dimensional context analysis of conversation state.

    Signals are grouped into:
    - Conversation state (phase, coherence)
    - Readability & naturalness (Flesch, Fog, vocabulary, sentence variety)
    - Pathology detection (repetition, agreement saturation, formulaic)
    - Topic analysis (evolution, coherence)
    - Per-participant breakdown
    - Interaction dynamics (engagement, response patterns, epistemic stance, reasoning)
    """

    # Conversation state
    conversation_phase: float = 0.0
    semantic_coherence: float = 1.0

    # Readability & naturalness
    flesch_reading_ease: float = 50.0
    gunning_fog_index: float = 12.0
    vocabulary_richness: float = 0.5
    sentence_variety: float = 0.5

    # Pathology signals
    repetition_score: float = 0.0
    agreement_saturation: float = 0.0
    formulaic_score: float = 0.0

    # Topic analysis
    topic_coherence: float = 0.0
    topic_evolution: Dict[str, float] = field(default_factory=dict)

    # Per-participant breakdown
    participant_signals: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Interaction dynamics
    engagement_metrics: Dict[str, float] = field(default_factory=dict)
    response_patterns: Dict[str, float] = field(default_factory=dict)
    epistemic_stance: Dict[str, float] = field(default_factory=dict)
    reasoning_patterns: Dict[str, float] = field(default_factory=dict)

    domain_info: str = ""

    # Backward-compat properties
    @property
    def cognitive_load(self) -> float:
        """Map Gunning Fog to 0-1 scale: Fog 8=0.0, Fog 20=1.0."""
        return min(1.0, max(0.0, (self.gunning_fog_index - 8) / 12))

    @property
    def knowledge_depth(self) -> float:
        """Derived from vocabulary richness + topic coherence."""
        return min(1.0, (self.vocabulary_richness + self.topic_coherence) / 2)

    @property
    def uncertainty_markers(self) -> Dict[str, float]:
        """Return epistemic stance signals as uncertainty markers."""
        return self.epistemic_stance


class ContextAnalyzer:
    """Analyzes conversation context using NLP to produce actionable signals.

    Uses spaCy (when available) for sentence segmentation, POS tagging, lemmatization,
    noun chunk extraction, and NER. Falls back to regex-based analysis when spaCy is absent.
    Uses scikit-learn TF-IDF for topic and similarity analysis (always available).
    """

    def __init__(self, mode: str = "ai-ai"):
        self.mode = mode
        try:
            self.nlp: Any = SpacyModelSingleton.get_instance()
        except (OSError, Exception) as e:
            logger.warning(f"spaCy unavailable ({e}), using fallback analysis only")
            self.nlp = None
        self.vectorizer = VectorizerSingleton.get_instance()
        if self.nlp:
            logger.debug("ContextAnalyzer initialized with spaCy NLP pipeline")
        else:
            logger.debug("ContextAnalyzer initialized without spaCy (using fallbacks)")

    def analyze(self, conversation_history: List[Dict[str, str]]) -> ContextVector:
        """Analyze conversation history and return a ContextVector.

        Args:
            conversation_history: List of messages with 'role' and 'content' keys.

        Returns:
            ContextVector with all computed signals.
        """
        if not conversation_history:
            return ContextVector()

        contents = [str(msg.get("content", "")) for msg in conversation_history]

        return ContextVector(
            conversation_phase=self._safe(self._compute_conversation_phase, conversation_history, default=0.0),
            semantic_coherence=self._safe(self._compute_semantic_coherence, contents, default=1.0),
            flesch_reading_ease=self._safe(self._compute_flesch, contents, default=50.0),
            gunning_fog_index=self._safe(self._compute_gunning_fog, contents, default=12.0),
            vocabulary_richness=self._safe(self._compute_vocabulary_richness, contents, default=0.5),
            sentence_variety=self._safe(self._compute_sentence_variety, contents, default=0.5),
            repetition_score=self._safe(self._compute_repetition, conversation_history, default=0.0),
            agreement_saturation=self._safe(self._compute_agreement_saturation, conversation_history, default=0.0),
            formulaic_score=self._safe(self._compute_formulaic_score, conversation_history, default=0.0),
            topic_coherence=self._safe(self._compute_topic_coherence, contents, default=0.0),
            topic_evolution=self._safe(self._compute_topic_evolution, contents, default={}),
            participant_signals=self._safe(self._compute_participant_signals, conversation_history, default={}),
            engagement_metrics=self._safe(self._compute_engagement, conversation_history, default={}),
            response_patterns=self._safe(self._compute_response_patterns, conversation_history, default={}),
            epistemic_stance=self._safe(self._compute_epistemic_stance, contents, default={}),
            reasoning_patterns=self._safe(self._compute_reasoning_patterns, contents, default={}),
        )

    def _safe(self, fn, *args, default=None):
        """Call fn with args, returning default on any exception."""
        try:
            return fn(*args)
        except Exception as e:
            logger.warning(f"{fn.__name__} failed: {e}")
            return default

    # ─── Text Utilities ───────────────────────────────────────────────

    def _get_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy or regex fallback."""
        if self.nlp:
            doc = self.nlp(text[:10000])  # pylint: disable=not-callable
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def _get_words(self, text: str) -> List[str]:
        """Extract words from text, optionally lemmatized with spaCy."""
        if self.nlp:
            doc = self.nlp(text[:10000])  # pylint: disable=not-callable
            return [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
        return [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', text)]

    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count via vowel cluster counting."""
        word = word.lower().rstrip('e')
        if not word:
            return 1
        count = len(re.findall(r'[aeiouy]+', word))
        return max(1, count)

    def _get_messages_by_role(self, history: List[Dict]) -> Dict[str, List[str]]:
        """Group message contents by role."""
        by_role: Dict[str, List[str]] = {}
        for msg in history:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))
            if content.strip():
                by_role.setdefault(role, []).append(content)
        return by_role

    # ─── Conversation State ───────────────────────────────────────────

    def _compute_conversation_phase(self, history: List[Dict]) -> float:
        """Conversation maturity: 0.0=opening, 1.0=mature.

        Uses logistic curve over turn count and total word count.
        Anchored at 6 turns / 500 words = 0.5 (mid-conversation).
        """
        n_turns = len(history)
        total_words = sum(len(str(m.get("content", "")).split()) for m in history)

        turn_phase = 1.0 / (1.0 + math.exp(-0.5 * (n_turns - 6)))
        word_phase = 1.0 / (1.0 + math.exp(-0.005 * (total_words - 500)))

        return max(turn_phase, word_phase)

    def _compute_semantic_coherence(self, contents: List[str]) -> float:
        """Consecutive-message TF-IDF cosine similarity.

        Returns mean cosine similarity between adjacent messages.
        No /2 division -- raw cosine values, properly scaled 0-1.
        """
        if len(contents) < 3:
            return 1.0

        recent = [c for c in contents[-6:] if c.strip()]
        if len(recent) < 2:
            return 1.0

        tfidf_matrix = self.vectorizer.fit_transform(recent)
        similarities = []
        for i in range(len(recent) - 1):
            sim = cosine_similarity(tfidf_matrix[i:i + 1], tfidf_matrix[i + 1:i + 2])[0][0]
            similarities.append(float(sim))

        return float(np.mean(similarities)) if similarities else 1.0

    # ─── Readability & Naturalness ────────────────────────────────────

    def _compute_flesch(self, contents: List[str]) -> float:
        """Flesch Reading Ease for recent messages.

        Scale: 0-30=very difficult, 60-70=standard conversation, 90-100=very easy.
        """
        recent = [c for c in contents[-3:] if c.strip()]
        if not recent:
            return 50.0

        combined = " ".join(recent)
        sentences = self._get_sentences(combined)
        words = [w for w in re.findall(r'\b[a-zA-Z]+\b', combined)]

        if not sentences or not words:
            return 50.0

        n_sentences = max(len(sentences), 1)
        n_words = len(words)
        n_syllables = sum(self._count_syllables(w) for w in words)

        score = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / max(n_words, 1))
        return float(np.clip(score, 0.0, 100.0))

    def _compute_gunning_fog(self, contents: List[str]) -> float:
        """Gunning Fog Index for recent messages.

        Scale: grade level (8=8th grade, 12=college, 17+=academic).
        Complex words = 3+ syllables excluding common suffixes.
        """
        recent = [c for c in contents[-3:] if c.strip()]
        if not recent:
            return 12.0

        combined = " ".join(recent)
        sentences = self._get_sentences(combined)
        words = [w for w in re.findall(r'\b[a-zA-Z]+\b', combined)]

        if not sentences or not words:
            return 12.0

        n_sentences = max(len(sentences), 1)
        n_words = len(words)

        # Complex words: 3+ syllables, excluding common suffixes
        suffix_pattern = re.compile(r'(ing|ed|es|ly|ment|ness|tion|sion)$', re.I)
        complex_count = 0
        for w in words:
            base = suffix_pattern.sub('', w)
            if self._count_syllables(w) >= 3 and self._count_syllables(base) >= 3:
                complex_count += 1

        fog = 0.4 * ((n_words / n_sentences) + 100 * (complex_count / max(n_words, 1)))
        return float(np.clip(fog, 0.0, 25.0))

    def _compute_vocabulary_richness(self, contents: List[str]) -> float:
        """Moving-Average Type-Token Ratio (MATTR) windowed for short text.

        spaCy: uses lemmatized forms. Fallback: surface forms.
        Scale: 0.3=repetitive, 0.8+=rich vocabulary.
        """
        recent = [c for c in contents[-3:] if c.strip()]
        if not recent:
            return 0.5

        combined = " ".join(recent)
        words = self._get_words(combined)

        if len(words) < 5:
            return 0.5

        # MATTR with window of 50 words
        window = min(50, len(words))
        if len(words) <= window:
            return len(set(words)) / len(words)

        ratios = []
        for i in range(len(words) - window + 1):
            chunk = words[i:i + window]
            ratios.append(len(set(chunk)) / window)

        return float(np.mean(ratios))

    def _compute_sentence_variety(self, contents: List[str]) -> float:
        """Coefficient of variation of sentence lengths.

        High variety (>0.5) = human-like mixed sentence lengths.
        Low variety (<0.15) = robotic uniform lengths.
        """
        recent = [c for c in contents[-3:] if c.strip()]
        if not recent:
            return 0.5

        combined = " ".join(recent)
        sentences = self._get_sentences(combined)

        if len(sentences) < 3:
            return 0.5

        lengths = [len(s.split()) for s in sentences]
        mean_len = np.mean(lengths)

        if mean_len == 0:
            return 0.0

        return float(np.std(lengths) / mean_len)

    # ─── Topic Analysis ───────────────────────────────────────────────

    def _compute_topic_evolution(self, contents: List[str]) -> Dict[str, float]:
        """Extract topics from conversation using noun chunks (spaCy) or TF-IDF top terms.

        Returns dict of topic -> normalized frequency.
        """
        recent = [c for c in contents[-8:] if c.strip()]
        if not recent:
            return {}

        combined = " ".join(recent)
        topics: Dict[str, int] = {}

        if self.nlp:
            doc = self.nlp(combined[:10000])  # pylint: disable=not-callable
            for chunk in doc.noun_chunks:
                key = chunk.root.lemma_.lower()
                if len(key) > 2:
                    topics[key] = topics.get(key, 0) + 1
            for ent in doc.ents:
                key = ent.text.lower()
                topics[key] = topics.get(key, 0) + 1
        else:
            # Fallback: TF-IDF top terms
            try:
                tfidf = self.vectorizer.fit_transform(recent)
                feature_names = self.vectorizer.get_feature_names_out()
                scores = np.asarray(tfidf.sum(axis=0)).flatten()
                top_indices = scores.argsort()[-20:][::-1]
                for idx in top_indices:
                    if scores[idx] > 0:
                        topics[feature_names[idx]] = int(scores[idx] * 100)
            except Exception:
                for word in re.findall(r'\b[a-zA-Z]{4,}\b', combined.lower()):
                    topics[word] = topics.get(word, 0) + 1

        if not topics:
            return {}

        total = sum(topics.values())
        return {k: v / total for k, v in sorted(topics.items(), key=lambda x: -x[1])[:15]}

    def _compute_topic_coherence(self, contents: List[str]) -> float:
        """How focused the conversation is on consistent topics.

        Uses average pairwise TF-IDF cosine similarity across messages.
        """
        recent = [c for c in contents[-8:] if c.strip()]
        if len(recent) < 2:
            return 0.5

        tfidf_matrix = self.vectorizer.fit_transform(recent)
        sim_matrix = cosine_similarity(tfidf_matrix)

        n = sim_matrix.shape[0]
        if n < 2:
            return 0.5

        total_sim = (sim_matrix.sum() - n) / (n * (n - 1))
        return float(np.clip(total_sim, 0.0, 1.0))

    # ─── Pathology Signals ────────────────────────────────────────────

    def _compute_repetition(self, history: List[Dict]) -> float:
        """Per-participant self-similarity detecting looping/rehashing.

        Returns max across participants (worst case).
        """
        by_role = self._get_messages_by_role(history)
        scores = []

        for _role, messages in by_role.items():
            if len(messages) < 2:
                continue
            recent = messages[-4:]
            score = self._participant_repetition(recent)
            scores.append(score)

        return max(scores) if scores else 0.0

    def _participant_repetition(self, messages: List[str]) -> float:
        """Compute repetition score for a single participant's messages."""
        if len(messages) < 2:
            return 0.0

        # TF-IDF cosine self-similarity between consecutive own messages
        try:
            if self.nlp:
                processed = []
                for msg in messages:
                    doc = self.nlp(msg[:5000])  # pylint: disable=not-callable
                    processed.append(" ".join(t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space))
            else:
                processed = messages

            tfidf = self.vectorizer.fit_transform(processed)
            similarities = []
            for i in range(len(processed) - 1):
                sim = cosine_similarity(tfidf[i:i + 1], tfidf[i + 1:i + 2])[0][0]
                similarities.append(float(sim))
            cosine_score = float(np.mean(similarities)) if similarities else 0.0
        except Exception:
            cosine_score = 0.0

        ngram_score = self._ngram_overlap(messages)

        return 0.6 * cosine_score + 0.4 * ngram_score

    def _ngram_overlap(self, messages: List[str], n: int = 3) -> float:
        """Character n-gram overlap between last message and earlier ones."""
        if len(messages) < 2:
            return 0.0

        def get_ngrams(text: str) -> set:
            text = text.lower()
            return {text[i:i + n] for i in range(len(text) - n + 1)} if len(text) >= n else set()

        last_ngrams = get_ngrams(messages[-1])
        if not last_ngrams:
            return 0.0

        earlier = " ".join(messages[:-1])
        earlier_ngrams = get_ngrams(earlier)
        if not earlier_ngrams:
            return 0.0

        overlap = len(last_ngrams & earlier_ngrams)
        return overlap / len(last_ngrams)

    def _compute_agreement_saturation(self, history: List[Dict]) -> float:
        """Detect premature consensus/deadlock between participants.

        Combines cross-participant similarity, declining length, and absence of contrastive markers.
        """
        by_role = self._get_messages_by_role(history)
        roles = [r for r in by_role if len(by_role[r]) >= 2]

        if len(roles) < 2:
            return 0.0

        role_a, role_b = roles[0], roles[1]
        recent_a = " ".join(by_role[role_a][-2:])
        recent_b = " ".join(by_role[role_b][-2:])

        try:
            tfidf = self.vectorizer.fit_transform([recent_a, recent_b])
            cross_sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        except Exception:
            cross_sim = 0.0

        # Length trend: are messages getting shorter?
        all_lengths = [len(str(m.get("content", "")).split()) for m in history if str(m.get("content", "")).strip()]
        if len(all_lengths) >= 4:
            recent_avg = np.mean(all_lengths[-3:])
            overall_avg = np.mean(all_lengths)
            length_decline = max(0.0, (overall_avg - recent_avg) / max(overall_avg, 1))
        else:
            length_decline = 0.0

        # Contrastive marker ratio
        contrastive = {"but", "however", "although", "yet", "though", "whereas", "nevertheless", "disagree", "conversely"}
        recent_text = " ".join(str(m.get("content", "")).lower() for m in history[-4:])
        words = recent_text.split()
        contrastive_count = sum(1 for w in words if w in contrastive)
        contrastive_ratio = contrastive_count / max(len(words), 1)
        agreement_from_contrast = 1.0 - min(1.0, contrastive_ratio * 30)

        return float(np.clip(
            0.4 * cross_sim + 0.3 * length_decline + 0.3 * agreement_from_contrast,
            0.0, 1.0
        ))

    def _compute_formulaic_score(self, history: List[Dict]) -> float:
        """Per-participant roboticness detection. Returns max across participants."""
        by_role = self._get_messages_by_role(history)
        scores = []

        for _role, messages in by_role.items():
            if len(messages) < 2:
                continue
            recent = messages[-3:]
            score = self._participant_formulaic(recent)
            scores.append(score)

        return max(scores) if scores else 0.0

    def _participant_formulaic(self, messages: List[str]) -> float:
        """Compute formulaic score for a single participant's messages."""
        if not messages:
            return 0.0

        combined = "\n".join(messages)
        lines = combined.split("\n")

        # List ratio
        list_pattern = re.compile(r'^\s*[-*\u2022]\s|^\s*\d+[.)]\s')
        list_lines = sum(1 for line in lines if list_pattern.match(line))
        list_ratio = list_lines / max(len(lines), 1)

        # Sentence-start diversity
        sentences = self._get_sentences(combined)
        if len(sentences) >= 3:
            starts = [" ".join(s.split()[:2]).lower() for s in sentences if s.split()]
            start_diversity = len(set(starts)) / len(starts) if starts else 1.0
        else:
            start_diversity = 1.0

        # Structure fingerprint similarity across messages
        if len(messages) >= 2:
            fingerprints = []
            for msg in messages:
                msg_lines = msg.split("\n")
                n_paragraphs = len([l for l in msg_lines if l.strip() and not list_pattern.match(l)])
                n_lists = sum(1 for l in msg_lines if list_pattern.match(l))
                n_headings = sum(1 for l in msg_lines if l.strip().startswith("#") or l.strip().endswith(":"))
                fingerprints.append([n_paragraphs, n_lists, n_headings])

            fp_array = np.array(fingerprints, dtype=float)
            norms = np.linalg.norm(fp_array, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            fp_normalized = fp_array / norms
            sims = []
            for i in range(len(fp_normalized) - 1):
                sim = float(np.dot(fp_normalized[i], fp_normalized[i + 1]))
                sims.append(sim)
            structure_sim = float(np.mean(sims)) if sims else 0.0
        else:
            structure_sim = 0.0

        return float(np.clip(
            0.3 * list_ratio + 0.4 * (1.0 - start_diversity) + 0.3 * structure_sim,
            0.0, 1.0
        ))

    # ─── Per-Participant Signals ──────────────────────────────────────

    def _compute_participant_signals(self, history: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Compute per-participant pathology and readability scores."""
        by_role = self._get_messages_by_role(history)
        signals = {}

        for role, messages in by_role.items():
            if not messages:
                continue

            recent = messages[-3:]

            signals[role] = {
                "repetition": self._participant_repetition(recent) if len(recent) >= 2 else 0.0,
                "formulaic": self._participant_formulaic(recent) if len(recent) >= 2 else 0.0,
                "flesch": self._safe(self._compute_flesch, recent, default=50.0),
                "fog": self._safe(self._compute_gunning_fog, recent, default=12.0),
                "vocabulary_richness": self._safe(self._compute_vocabulary_richness, recent, default=0.5),
                "sentence_variety": self._safe(self._compute_sentence_variety, recent, default=0.5),
            }

        return signals

    # ─── Interaction Dynamics ─────────────────────────────────────────

    def _compute_engagement(self, history: List[Dict]) -> Dict[str, float]:
        """Engagement metrics based on structural properties."""
        if not history:
            return {"avg_response_length": 0.0, "turn_taking_balance": 1.0, "length_trend": 0.0}

        lengths = [len(str(m.get("content", "")).split()) for m in history]
        roles = [m.get("role", "") for m in history]

        user_turns = sum(1 for r in roles if r in ("user", "human"))
        assistant_turns = sum(1 for r in roles if r == "assistant")
        total = user_turns + assistant_turns
        if total > 1 and max(user_turns, assistant_turns) > 0:
            balance = min(user_turns, assistant_turns) / max(user_turns, assistant_turns)
        else:
            balance = 1.0

        if len(lengths) >= 3:
            recent_avg = float(np.mean(lengths[-3:]))
            overall_avg = float(np.mean(lengths))
            length_trend = (recent_avg - overall_avg) / max(overall_avg, 1)
        else:
            length_trend = 0.0

        return {
            "avg_response_length": float(np.mean(lengths)),
            "turn_taking_balance": float(balance),
            "length_trend": float(np.clip(length_trend, -1.0, 1.0)),
        }

    def _compute_response_patterns(self, history: List[Dict]) -> Dict[str, float]:
        """Structural response patterns."""
        if not history:
            return {"question_ratio": 0.0, "avg_sentences_per_msg": 0.0, "list_usage": 0.0}

        question_marks = 0
        total_sentences = 0
        list_markers = 0

        for msg in history:
            content = str(msg.get("content", ""))
            question_marks += content.count("?")
            sentences = self._get_sentences(content)
            total_sentences += max(len(sentences), 1)
            list_markers += len(re.findall(r'(?m)^\s*[-*\u2022\d]+[.)]\s', content))

        n = len(history)
        return {
            "question_ratio": question_marks / max(total_sentences, 1),
            "avg_sentences_per_msg": total_sentences / n,
            "list_usage": list_markers / n,
        }

    def _compute_epistemic_stance(self, contents: List[str]) -> Dict[str, float]:
        """Epistemic stance from modal verbs (closed-class) and question ratio."""
        recent = [c for c in contents[-4:] if c.strip()]
        if not recent:
            return {"hedging_ratio": 0.0, "assertiveness": 0.5,
                    "socratic": 0.0, "uncertainty": 0.0, "confidence": 0.5, "qualification": 0.0}

        combined = " ".join(recent).lower()
        words = combined.split()
        n_words = max(len(words), 1)

        hedge_words = {"might", "could", "would", "may", "perhaps", "possibly", "maybe",
                       "potentially", "probably", "likely", "unlikely", "seems", "appears"}
        hedge_count = sum(1 for w in words if w in hedge_words)
        hedging_ratio = hedge_count / n_words

        questions = combined.count("?")
        sentences = self._get_sentences(combined)
        total_sent = max(len(sentences), 1)
        assertiveness = 1.0 - (questions / total_sent) if total_sent > 0 else 0.5

        return {
            "hedging_ratio": hedging_ratio,
            "assertiveness": assertiveness,
            "socratic": questions / total_sent,
            "uncertainty": min(1.0, hedging_ratio * 15),
            "confidence": assertiveness,
            "qualification": min(1.0, hedging_ratio * 10),
        }

    def _compute_reasoning_patterns(self, contents: List[str]) -> Dict[str, float]:
        """Reasoning indicators from discourse connectives and formal structure."""
        recent = [c for c in contents[-4:] if c.strip()]
        if not recent:
            return {"deductive": 0.0, "inductive": 0.0, "abductive": 0.0,
                    "analogical": 0.0, "causal": 0.0}

        combined = " ".join(recent).lower()
        words = combined.split()
        n_words = max(len(words), 1)

        connectives = {"therefore", "thus", "hence", "because", "since", "however",
                       "although", "whereas", "consequently", "furthermore", "moreover",
                       "nevertheless", "if", "then", "given", "assuming"}
        connective_count = sum(1 for w in words if w in connectives)
        connective_density = connective_count / n_words

        formal_markers = len(re.findall(r'```|^\s*\d+\.\s|=>|->|:=|==', combined, re.M))
        formal_density = min(1.0, formal_markers / max(len(recent), 1))

        base_level = min(1.0, connective_density * 20)

        result = {
            "deductive": base_level * 0.3,
            "inductive": base_level * 0.2,
            "abductive": base_level * 0.1,
            "analogical": base_level * 0.1,
            "causal": base_level * 0.3,
        }

        if self.mode == "ai-ai":
            result.update({
                "formal_logic": formal_density,
                "systematic": min(1.0, connective_density * 15),
                "technical": 0.5,
                "precision": base_level * 0.2,
                "integration": base_level * 0.2,
            })

        return result
