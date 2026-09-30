"""Feature engineering layer.

TF-IDF | n-grams | sentiment | readability | entities | POS | stylometry |
clickbait — computed as a flat numeric vector consumed by the ML model layer.
"""
from __future__ import annotations

import re

from app.core.nlp import content_words, get_nlp, lexicon_sentiment, tokenize

CLICKBAIT_PAT = re.compile(
    r"\b(you won'?t believe|shocking|jaw[- ]dropping|insane|unbelievable|"
    r"what happened next|this one weird|doctors hate|miracle|secret(ly)?|"
    r"exposed|banned|they don'?t want you to know|wake up|sheep|"
    r"the truth about|what they.?re hiding|must see|gone viral|"
    r"break(ing)?:?|Exclusive:?|bombshell|game[- ]changer)\b",
    re.IGNORECASE,
)
HEDGE_PAT = re.compile(
    r"\b(approximately|roughly|about|estimated|according to|"
    r"preliminary|suggests|indicates|likely|approximately)\b",
    re.IGNORECASE,
)
FIRST_PERSON_PAT = re.compile(r"\b(i|me|my|mine|we|our|us)\b", re.IGNORECASE)
SECOND_PERSON_PAT = re.compile(r"\b(you|your|yours)\b", re.IGNORECASE)
_ADVERB_SUFFIX = re.compile(r"ly$", re.IGNORECASE)

ALLCAPS_WORD = re.compile(r"\b[A-Z]{3,}\b")


def _syllables(word: str) -> int:
    """Cheap syllable estimate for readability without external corpora."""
    word = word.lower().strip(".,!?;:")
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease computed locally (textstat-compatible formula)."""
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = tokenize(text)
    if not sentences or not words:
        return 50.0
    syllables = sum(_syllables(w) for w in words)
    words_per_sent = len(words) / len(sentences)
    syll_per_word = syllables / len(words)
    return 206.835 - 1.015 * words_per_sent - 84.6 * syll_per_word


def build_features(text: str) -> dict[str, float]:
    """Return the numeric feature vector used by the ML model layer."""
    words = tokenize(text)
    content = content_words(text)
    n_words = max(len(words), 1)

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    n_sent = max(len(sentences), 1)

    exclam = text.count("!")
    question = text.count("?")
    caps_words = ALLCAPS_WORD.findall(text)

    # POS / stylometry via spaCy when available
    adverb_ratio = 0.0
    entity_count = 0.0
    noun_ratio = 0.0
    nlp = get_nlp()
    if nlp.available and text.strip():
        try:
            doc = nlp(text[:4000])
            pos = [t.pos_ for t in doc if not t.is_punct]
            if pos:
                adverb_ratio = sum(1 for p in pos if p == "ADV") / len(pos)
                noun_ratio = sum(1 for p in pos if p in {"NOUN", "PROPN"}) / len(pos)
            entity_count = len(doc.ents)
        except Exception:  # noqa: BLE001
            pass

    clickbait_hits = CLICKBAIT_PAT.findall(text)
    hedge_hits = HEDGE_PAT.findall(text)

    feats = {
        # volumes
        "word_count": float(len(words)),
        "sentence_count": float(len(sentences)),
        "avg_sentence_len": len(words) / n_sent,
        "content_ratio": len(content) / n_words,
        # punctuation / caps stylometry
        "exclamation_count": float(exclam),
        "exclamation_ratio": exclam / n_words,
        "question_ratio": question / n_words,
        "caps_word_count": float(len(caps_words)),
        "caps_ratio": len(caps_words) / n_words,
        # sensationalism
        "clickbait_count": float(len(clickbait_hits)),
        "clickbait_density": len(clickbait_hits) / n_sent,
        # sentiment / hedging / style
        "sentiment": lexicon_sentiment(text),
        "sentiment_extremity": abs(lexicon_sentiment(text)),
        "hedge_count": float(len(hedge_hits)),
        "hedge_ratio": len(hedge_hits) / n_words,
        "first_person_ratio": len(FIRST_PERSON_PAT.findall(text)) / n_words,
        "second_person_ratio": len(SECOND_PERSON_PAT.findall(text)) / n_words,
        "adverb_ratio": adverb_ratio,
        "noun_ratio": noun_ratio,
        "unique_word_ratio": len(set(words)) / n_words,
        "entity_count": entity_count,
        # readability
        "flesch_reading_ease": flesch_reading_ease(text),
    }
    return feats


def features_to_vector(feats: dict[str, float]) -> list[float]:
    """Convert a feature dict into a fixed-order vector for ML models."""
    return [float(feats.get(name, 0.0)) for name in FEATURE_NAMES]


FEATURE_NAMES: list[str] = [
    "word_count", "sentence_count", "avg_sentence_len", "content_ratio",
    "exclamation_count", "exclamation_ratio", "question_ratio",
    "caps_word_count", "caps_ratio", "clickbait_count", "clickbait_density",
    "sentiment", "sentiment_extremity", "hedge_count", "hedge_ratio",
    "first_person_ratio", "second_person_ratio", "adverb_ratio", "noun_ratio",
    "unique_word_ratio", "entity_count", "flesch_reading_ease",
]
