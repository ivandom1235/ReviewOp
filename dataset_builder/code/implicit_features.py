from __future__ import annotations

from collections import Counter
import math
import re
from functools import lru_cache
from typing import Any, Dict, List

from mappings import (
    CONTRASTIVE_CONJUNCTIONS,
    GENERIC_ASPECT_STOPWORDS,
    NEGATIVE_WORDS,
    POSITIVE_WORDS,
    TEXT_STOPWORDS,
)
from llm_utils import build_llm_client
from utils import normalize_whitespace, split_sentences, token_count, tokenize


NEGATION_WORDS = {"not", "never", "no", "without", "hardly", "barely", "scarcely"}
INTENSIFIERS = {"very", "so", "too", "extremely", "really", "quite", "highly", "super", "incredibly", "especially"}
DIMINISHERS = {"barely", "hardly", "slightly", "somewhat", "little", "mildly"}
CLAUSE_BREAKERS = {"but", "however", "although", "though", "while", "yet", "still"}
POSITIVE_SENTIMENT_WORDS = set(POSITIVE_WORDS) | {
    "excellent", "great", "amazing", "wonderful", "love", "loved", "fantastic", "perfect", "solid", "smooth", "helpful",
    "stunning", "impressive", "quality", "reliable", "superb", "brilliant", "delighted", "satisfied", "pleased", "outstanding",
    "friendly", "efficient", "recommend", "best", "top-notch", "worth", "enjoyed", "refreshing", "tasty", "delicious",
}
NEGATIVE_SENTIMENT_WORDS = set(NEGATIVE_WORDS) | {
    "awful", "horrible", "poor", "annoying", "dirty", "broken", "late", "slow", "unusable", "badly", "issues", "problem", "problems",
    "disappointed", "worst", "terrible", "waste", "useless", "broken", "expensive", "rude", "frustrating", "faulty", "failed",
    "crash", "defect", "laggy", "slow", "clunky", "unhelpful", "noisy", "overpriced", "dirty", "old", "tiny", "small",
}


@lru_cache(maxsize=1)
def _load_spacy():
    try:
        import spacy
    except Exception:
        return None

    for model_name in ("en_core_web_sm", "en_core_web_md"):
        try:
            return spacy.load(model_name)
        except Exception:
            continue
    try:
        return spacy.blank("en")
    except Exception:
        return None


def _normalize_aspect(text: str) -> str:
    text = normalize_whitespace(text).lower()
    text = re.sub(r"[^a-z0-9\s_-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _canonicalize_aspect(aspect: str, *, seed_vocab: set[str]) -> str:
    aspect = _normalize_aspect(aspect)
    tokens = aspect.split()
    if not tokens:
        return aspect
    if aspect in seed_vocab:
        return aspect
    for token in tokens:
        if token in seed_vocab:
            return token
    return aspect


def _is_valid_aspect(aspect: str, *, seed_vocab: set[str] | None = None) -> bool:
    return _aspect_rejection_reason(aspect, seed_vocab=seed_vocab) is None


def _aspect_rejection_reason(aspect: str, *, seed_vocab: set[str] | None = None) -> str | None:
    if not aspect or len(aspect) < 3:
        return "too_short"
    if aspect in GENERIC_ASPECT_STOPWORDS:
        return "generic_stopword"
    if aspect in TEXT_STOPWORDS:
        return "text_stopword"
    tokens = aspect.split()
    if not tokens:
        return "empty"
    seeds = seed_vocab if seed_vocab is not None else set()
    if any(token in GENERIC_ASPECT_STOPWORDS or token in TEXT_STOPWORDS for token in tokens):
        return "contains_stopword_token"
    if all(token in TEXT_STOPWORDS for token in tokens):
        return "all_stopwords"
    if len(tokens) == 1:
        if aspect in POSITIVE_WORDS or aspect in NEGATIVE_WORDS:
            return "polar_word"
        return None
    if len(tokens) > 3:
        return "too_long"
    return None


def _lemmatize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def infer_sentiment(text: str) -> str:
    tokens = tokenize(text)
    pos, neg, _, _ = _sentiment_counts(tokens)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _sentiment_counts(tokens: List[str]) -> tuple[float, float, int, int]:
    pos = 0.0
    neg = 0.0
    negation_hits = 0
    sentiment_hits = 0
    for idx, token in enumerate(tokens):
        base = 0.0
        if token in POSITIVE_SENTIMENT_WORDS:
            base = 1.0
            sentiment_hits += 1
        elif token in NEGATIVE_SENTIMENT_WORDS:
            base = -1.0
            sentiment_hits += 1
        if base == 0.0:
            continue
        window = tokens[max(0, idx - 3):idx]
        negation_count = sum(1 for prev in window if prev in NEGATION_WORDS)
        if negation_count % 2 != 0:
            negation_hits += 1
            base *= -1.0
        if any(prev in INTENSIFIERS for prev in window):
            base *= 1.25
        if any(prev in DIMINISHERS for prev in window):
            base *= 0.75
        if base > 0:
            pos += base
        else:
            neg += abs(base)
    return pos, neg, negation_hits, sentiment_hits


def _extract_phrases_with_spacy(text: str) -> List[str]:
    nlp = _load_spacy()
    if nlp is None:
        return []
    doc = nlp(text)
    phrases: List[str] = []
    try:
        for chunk in getattr(doc, "noun_chunks", []):
            phrase = _normalize_aspect(chunk.text)
            if phrase:
                phrases.append(phrase)
    except Exception:
        pass
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"}:
            lemma = _normalize_aspect(token.lemma_ or token.text)
            if lemma:
                phrases.append(lemma)
    return phrases


def _extract_phrases_fallback(text: str, *, seed_vocab: set[str]) -> List[str]:
    phrases: List[str] = []
    tokens = tokenize(text)
    for idx, token in enumerate(tokens):
        if token in TEXT_STOPWORDS or token in GENERIC_ASPECT_STOPWORDS:
            continue
        if len(token) < 3:
            continue
        if token in POSITIVE_WORDS or token in NEGATIVE_WORDS:
            continue
        lemma = _lemmatize_token(token)
        if not seed_vocab or lemma in seed_vocab:
            phrases.append(lemma)
        elif not seed_vocab:
            phrases.append(lemma)
        if idx + 1 < len(tokens):
            nxt = tokens[idx + 1]
            if nxt not in TEXT_STOPWORDS and nxt not in GENERIC_ASPECT_STOPWORDS and nxt not in POSITIVE_WORDS and nxt not in NEGATIVE_WORDS:
                phrase = f"{token} {nxt}"
                phrases.append(_normalize_aspect(phrase))
    return phrases


def _extract_candidate_phrases(text: str, *, seed_vocab: set[str]) -> List[str]:
    phrases = _extract_phrases_with_spacy(text) or _extract_phrases_fallback(text, seed_vocab=seed_vocab)
    cleaned: List[str] = []
    for phrase in phrases:
        phrase = _canonicalize_aspect(phrase, seed_vocab=seed_vocab)
        if not _is_valid_aspect(phrase, seed_vocab=seed_vocab):
            continue
        cleaned.append(phrase)
    return cleaned


def _candidate_priority(phrase: str, *, doc_count: int, total_docs: int, seed_vocab: set[str], sentiment_context_boost: float = 0.0) -> float:
    tokens = phrase.split()
    if not tokens:
        return 0.0
    seed_hits = sum(1 for token in tokens if token in seed_vocab)
    support = doc_count / max(1, total_docs)
    exact_bonus = 1.5 if phrase in seed_vocab else 0.0
    length_bonus = 0.35 if len(tokens) == 1 else 0.6 if len(tokens) == 2 else 0.3
    support_bonus = min(1.5, support * 6.0)
    seed_bonus = seed_hits * 1.8
    return exact_bonus + length_bonus + support_bonus + seed_bonus + sentiment_context_boost


def learn_aspect_seed_vocab(
    train_rows: List[Dict[str, Any]],
    *,
    text_column: str,
    vocab_size: int,
    seed_vocab: set[str] | None = None,
) -> Dict[str, Any]:
    learned: List[str] = discover_aspects(train_rows, text_column=text_column, vocab_size=vocab_size, seed_vocab=seed_vocab)
    support = Counter()
    total_docs = 0
    for row in train_rows:
        text = normalize_whitespace(row.get(text_column, ""))
        if not text:
            continue
        total_docs += 1
        tokens = set(tokenize(text))
        for aspect in learned:
            if aspect in tokens:
                support[aspect] += 1
    filtered = [aspect for aspect in learned if support[aspect] >= 2]
    if not filtered:
        filtered = learned[: max(1, min(vocab_size, len(learned)))]
    return {
        "learned_seed_vocab": filtered,
        "learned_seed_support": {aspect: support[aspect] for aspect in filtered},
        "learned_seed_total_docs": total_docs,
    }


def discover_aspects(
    train_rows: List[Dict[str, Any]],
    *,
    text_column: str,
    vocab_size: int,
    seed_vocab: set[str] | None = None,
) -> List[str]:
    seed_vocab = set(seed_vocab or set())
    doc_count: Counter[str] = Counter()
    term_score: Counter[str] = Counter()
    total_docs = 0

    for row in train_rows:
        text = normalize_whitespace(row.get(text_column, ""))
        if not text:
            continue
        total_docs += 1
        phrases = _extract_candidate_phrases(text, seed_vocab=seed_vocab)
        if not phrases:
            continue
        seen = set()
        for phrase in phrases:
            if phrase not in seen:
                doc_count[phrase] += 1
                seen.add(phrase)
        for phrase in phrases:
            tf = phrases.count(phrase)
            idf = math.log((1 + total_docs) / (1 + doc_count[phrase])) + 1.0
            term_score[phrase] += tf * idf

    boosted = Counter()
    for phrase, score in term_score.items():
        priority = _candidate_priority(phrase, doc_count=doc_count.get(phrase, 0), total_docs=total_docs, seed_vocab=seed_vocab)
        boosted[phrase] = score + priority

    for aspect in seed_vocab:
        boosted[aspect] += 4
    ranked = [
        phrase
        for phrase, _ in boosted.most_common()
        if _is_valid_aspect(phrase, seed_vocab=seed_vocab) and (phrase in seed_vocab or doc_count[phrase] >= 2)
    ]

    canonicalized: List[str] = []
    seen = set()
    for phrase in ranked:
        root = _canonicalize_aspect(phrase, seed_vocab=seed_vocab)
        if not root or root in seen:
            continue
        if len(root.split()) > 1:
            continue
        seen.add(root)
        canonicalized.append(root)
        if len(canonicalized) >= vocab_size:
            break

    if not canonicalized:
        canonicalized = [aspect for aspect in list(seed_vocab)[:vocab_size]]
    return canonicalized


def _split_contrastive(sentence: str) -> List[str]:
    parts = [sentence]
    for conj in CONTRASTIVE_CONJUNCTIONS:
        next_parts: List[str] = []
        for part in parts:
            if f" {conj} " in part.lower():
                next_parts.extend([p.strip(" ,;:-") for p in re.split(rf"\s+{re.escape(conj)}\s+", part, flags=re.IGNORECASE)])
            else:
                next_parts.append(part)
        parts = next_parts
    cleaned = [part.strip(" ,;:-") for part in parts if token_count(part) >= 3]
    return cleaned or [sentence]


def _symptom_map_hits(clause_tokens: List[str], symptom_map: dict[str, str], aspect: str) -> float:
    hits = 0.0
    for symptom, target in symptom_map.items():
        if target == aspect and symptom in clause_tokens:
            hits += 1.0
    return hits


def _explicit_aspect_match(clause_tokens: List[str], seed_vocab: set[str]) -> str | None:
    for aspect in seed_vocab:
        if aspect in clause_tokens:
            return aspect
    return None


def _aspect_lexical_score(
    text: str,
    aspect: str,
    *,
    seed_vocab: set[str],
    symptom_map: dict[str, str],
) -> float:
    if not _is_valid_aspect(aspect, seed_vocab=seed_vocab):
        return 0.0
    tokens = tokenize(text.lower())
    aspect_tokens = aspect.split()
    if aspect in tokens:
        return -5.0
    score = 0.0
    score += _symptom_map_hits(tokens, symptom_map, aspect) * 2.5
    if len(aspect_tokens) == 1 and aspect in seed_vocab:
        score += 0.75
    if len(aspect_tokens) > 1:
        score += 0.35 * len(set(tokens).intersection(aspect_tokens))
    return score


def _aspect_window_support(clause_tokens: List[str], aspect: str) -> float:
    aspect_tokens = aspect.split()
    if not aspect_tokens:
        return 0.0
    joined = " ".join(clause_tokens)
    if aspect in joined:
        return 1.5
    token_set = set(clause_tokens)
    hits = len(token_set.intersection(aspect_tokens))
    if hits == 0:
        return 0.0
    return 0.8 + 0.4 * hits


def _sentiment_evidence(clause_tokens: List[str]) -> tuple[float, int, int]:
    pos, neg, negation_hits, sentiment_hits = _sentiment_counts(clause_tokens)
    evidence = abs(pos - neg)
    return evidence, negation_hits, sentiment_hits


def _softmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def _sentence_confidence(text: str, top_prob: float, margin: float, evidence_count: int, sentiment_evidence: float) -> float:
    length_boost = min(0.05, token_count(text) * 0.002)
    evidence_boost = min(0.12, evidence_count * 0.03)
    margin_boost = min(0.10, margin * 0.45)
    sentiment_boost = min(0.08, sentiment_evidence * 0.04)
    confidence = 0.30 + 0.34 * top_prob + length_boost + evidence_boost + margin_boost + sentiment_boost
    return round(min(0.99, confidence), 4)


def _score_clause(
    clause: str,
    candidate_aspects: List[str],
    *,
    seed_vocab: set[str],
    llm_client: Any | None,
    confidence_threshold: float,
    symptom_map: dict[str, str],
) -> tuple[str | None, str, float, bool]:
    if not candidate_aspects:
        return None, infer_sentiment(clause), 0.0, False

    clause_tokens = tokenize(clause)
    explicit_aspect = _explicit_aspect_match(clause_tokens, seed_vocab)
    if explicit_aspect is not None:
        return None, infer_sentiment(clause), 0.0, False

    sentiment_evidence, negation_hits, sentiment_hits = _sentiment_evidence(clause_tokens)
    has_symptom = False
    for symptom in symptom_map:
        if symptom in clause_tokens:
            has_symptom = True
            break

    if sentiment_hits == 0 and not has_symptom:
        return None, infer_sentiment(clause), 0.0, False

    scores = []
    for aspect in candidate_aspects:
        lexical = _aspect_lexical_score(clause, aspect, seed_vocab=seed_vocab, symptom_map=symptom_map)
        window_support = _aspect_window_support(clause_tokens, aspect)
        bonus = 0.0
        if aspect in seed_vocab:
            bonus += 0.2
        scores.append(lexical + window_support + bonus)
    probs = _softmax(scores)
    ranked = sorted(zip(candidate_aspects, scores, probs), key=lambda item: item[1], reverse=True)
    if not ranked:
        return None, infer_sentiment(clause), 0.0, False

    top_aspect, top_score, top_prob = ranked[0]
    second_prob = ranked[1][2] if len(ranked) > 1 else 0.0
    sentiment = infer_sentiment(clause)
    evidence_count = len([x for x in scores if x > 0])
    confidence = _sentence_confidence(clause, top_prob, top_prob - second_prob, evidence_count, sentiment_evidence)
    if sentiment_evidence < 0.4 and not has_symptom:
        confidence = round(max(0.0, confidence - 0.20), 4)
    if top_score < 1.0:
        confidence = round(max(0.0, confidence - 0.15), 4)
    if negation_hits:
        confidence = round(min(0.99, confidence + 0.05), 4)
    if len(clause_tokens) <= 4 and not has_symptom:
        confidence = round(max(0.0, confidence - 0.12), 4)

    plausible = (top_score >= 1.0 or top_prob >= 0.55) and (sentiment_evidence >= 0.35 or has_symptom)
    if has_symptom:
        plausible = plausible or top_score >= 0.8

    should_call_llm = llm_client is not None and plausible and confidence < confidence_threshold
    if should_call_llm:
        result = llm_client.infer(sentence=clause, candidate_aspects=candidate_aspects)
        if result is not None and result.aspect:
            llm_conf = max(confidence, float(result.confidence))
            return result.aspect, result.sentiment, llm_conf, bool(result.is_novel_aspect)

    if confidence >= confidence_threshold and plausible:
        return top_aspect, sentiment, confidence, False
    if plausible:
        return top_aspect, sentiment, confidence, False
    return None, sentiment, confidence, False


def collect_implicit_diagnostics(
    train_rows: List[Dict[str, Any]],
    *,
    text_column: str,
    candidate_aspects: List[str],
    seed_vocab: set[str],
    confidence_threshold: float,
    learned_seed_vocab: List[str] | None = None,
    symptom_map: dict[str, str] | None = None,
) -> Dict[str, Any]:
    symptom_map = symptom_map or {}
    rejection_reasons: Counter[str] = Counter()
    rejected_examples: List[Dict[str, str]] = []
    aspect_counts: Counter[str] = Counter()
    confusion_counts: Counter[str] = Counter()
    fallback_count = 0
    scored_clauses = 0
    negation_hits = 0
    sentiment_hit_count = 0
    explicit_leakage_count = 0
    accepted_examples: List[Dict[str, str]] = []

    for row in train_rows[:1000]:
        text = normalize_whitespace(row.get(text_column, ""))
        if not text:
            continue
        for sentence in split_sentences(text):
            for clause in _split_contrastive(sentence):
                scored_clauses += 1
                clause_candidate_aspects = candidate_aspects or []
                clause_tokens = tokenize(clause)
                if _explicit_aspect_match(clause_tokens, seed_vocab):
                    explicit_leakage_count += 1
                aspect, _, confidence, _ = _score_clause(
                    clause,
                    clause_candidate_aspects,
                    seed_vocab=seed_vocab,
                    llm_client=None,
                    confidence_threshold=confidence_threshold,
                    symptom_map=symptom_map,
                )
                _, clause_negations, clause_sentiment_hits = _sentiment_evidence(clause_tokens)
                negation_hits += clause_negations
                sentiment_hit_count += clause_sentiment_hits
                if aspect:
                    aspect_counts[aspect] += 1
                    if len(accepted_examples) < 20:
                        accepted_examples.append({"clause": clause, "aspect": aspect, "confidence": f"{confidence:.3f}"})
                else:
                    fallback_count += 1
                    if len(clause_tokens) <= 4:
                        confusion_counts["short_clause_rejected"] += 1
                    if clause_sentiment_hits == 0 and _explicit_aspect_match(clause_tokens, seed_vocab) is None:
                        confusion_counts["low_signal_rejected"] += 1
                    if len(rejected_examples) < 20:
                        rejected_examples.append({"clause": clause, "reason": "no_plausible_aspect"})

        for phrase in _extract_candidate_phrases(text, seed_vocab=seed_vocab):
            reason = _aspect_rejection_reason(phrase, seed_vocab=seed_vocab)
            if reason is not None:
                rejection_reasons[reason] += 1
                if len(rejected_examples) < 20:
                    rejected_examples.append({"phrase": phrase, "reason": reason})

    return {
        "top_implicit_aspects": aspect_counts.most_common(20),
        "learned_seed_vocab": list(learned_seed_vocab or candidate_aspects[: min(len(candidate_aspects), 50)]),
        "sentiment_lexicon_coverage": {
            "sentiment_hit_count": sentiment_hit_count,
            "negation_hit_count": negation_hits,
            "candidate_aspect_count": len(candidate_aspects),
        },
        "candidate_rejection_reasons": rejection_reasons.most_common(),
        "confusion_patterns": confusion_counts.most_common(),
        "explicit_leakage_count": explicit_leakage_count,
        "accepted_clause_count": sum(aspect_counts.values()),
        "rejected_clause_count": fallback_count,
        "false_positive_samples": accepted_examples,
        "false_negative_samples": rejected_examples,
        "accepted_examples": accepted_examples,
        "rejected_examples": rejected_examples,
        "fallback_clause_count_sample": fallback_count,
        "scored_clause_count_sample": scored_clauses,
    }


def build_implicit_row(
    row: Dict[str, Any],
    *,
    text_column: str,
    candidate_aspects: List[str],
    seed_vocab: set[str],
    confidence_threshold: float,
    llm_enabled: bool = False,
    llm_settings: Any | None = None,
    symptom_map: dict[str, str] = None,
) -> Dict[str, Any]:
    symptom_map = symptom_map or {}
    text = normalize_whitespace(row.get(text_column, ""))
    sentences = split_sentences(text)
    llm_client = build_llm_client(llm_settings, enabled=llm_enabled) if llm_settings is not None else None
    aspects: List[str] = []
    aspect_sentiments: Dict[str, str] = {}
    aspect_confidence: Dict[str, float] = {}
    novel_aspects: List[str] = []
    confidences: List[float] = []
    tier = 1
    if not sentences:
        tier = 3

    for sentence in (sentences or [text]):
        for clause in _split_contrastive(sentence):
            clause_tokens = tokenize(clause)
            explicit_aspect = _explicit_aspect_match(clause_tokens, seed_vocab)
            if explicit_aspect:
                if explicit_aspect not in aspects:
                    aspects.append(explicit_aspect)
                    aspect_sentiments[explicit_aspect] = infer_sentiment(clause)
                    aspect_confidence[explicit_aspect] = 0.99
                confidences.append(0.99)
                continue

            aspect, sentiment, confidence, is_novel = _score_clause(
                clause,
                candidate_aspects,
                seed_vocab=seed_vocab,
                llm_client=llm_client,
                confidence_threshold=confidence_threshold,
                symptom_map=symptom_map,
            )
            if not aspect:
                tier = 3
                continue

            if confidence < confidence_threshold:
                tier = 3
            elif confidence < 0.85 and tier < 3:
                tier = 2

            if aspect not in aspects:
                aspects.append(aspect)
            aspect_sentiments[aspect] = sentiment
            aspect_confidence[aspect] = confidence
            confidences.append(confidence)
            if is_novel and aspect not in candidate_aspects:
                novel_aspects.append(aspect)

    if not aspects:
        tier = 3

    dominant_sentiment = infer_sentiment(text)
    avg_confidence = round(sum(confidences) / max(1, len(confidences)), 4)
    return {
        "id": row.get("id"),
        "split": row.get("split"),
        "source_text": text,
        "implicit": {
            "aspects": aspects,
            "dominant_sentiment": dominant_sentiment,
            "aspect_sentiments": aspect_sentiments,
            "aspect_confidence": aspect_confidence,
            "avg_confidence": avg_confidence,
            "extraction_tier": tier,
            "novel_aspects": novel_aspects,
            "sentence_count_processed": len(sentences) or 1,
        },
    }
