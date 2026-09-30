"""METADATA signal: completeness and hygiene of the article's metadata."""
from __future__ import annotations

from app.core.features import build_features
from app.core.nlp import get_nlp
from app.schemas import MetadataSignal


def analyze_metadata(
    text: str,
    title: str = "",
    author: str = "",
    published: str = "",
) -> MetadataSignal:
    feats = build_features(text) if text.strip() else {}
    reasons: list[str] = []

    has_author = bool(author.strip())
    has_date = bool(published.strip())
    has_title = bool(title.strip())

    if has_author:
        reasons.append("Author byline present.")
    else:
        reasons.append("No author byline — anonymity is a weak warning sign.")
    if has_date:
        reasons.append("Publication date present.")
    else:
        reasons.append("No publication date.")
    if not has_title:
        reasons.append("No title supplied.")
    if feats.get("word_count", 0) < 60:
        reasons.append("Very short body text.")

    score = 0.5
    score += 0.15 if has_author else -0.1
    score += 0.1 if has_date else -0.1
    score += 0.05 if has_title else -0.05
    if feats.get("word_count", 0) >= 250:
        score += 0.05

    # Language detection is approximate: spaCy model is English; if entity
    # extraction finds nothing at all on long text, flag possible non-English.
    language = "en"
    nlp = get_nlp()
    if nlp.available and feats.get("word_count", 0) > 80:
        try:
            doc = nlp(text[:2000])
            if not doc.ents:
                language = "unknown"
                reasons.append("No entities recognized — possible non-English text.")
        except Exception:  # noqa: BLE001
            pass

    return MetadataSignal(
        score=round(max(0.0, min(1.0, score)), 3),
        confidence=0.4,
        word_count=int(feats.get("word_count", 0)),
        sentence_count=int(feats.get("sentence_count", 0)),
        has_author=has_author,
        has_date=has_date,
        has_title=has_title,
        language=language,
        reasons=reasons,
    )
