"""CONTENT signal: text-only analysis (features + ML ensemble)."""
from __future__ import annotations

from app.core.features import build_features
from app.core.ml import predict_ensemble
from app.schemas import NLPSignal


def analyze_text(text: str) -> tuple[NLPSignal, dict[str, float]]:
    """Return (NLPSignal, feature_dict). Score: 1 = credible-looking, 0 = fake-prone."""
    feats = build_features(text)

    reasons: list[str] = []
    score = 0.5
    ml = predict_ensemble(text)

    if ml.available:
        # ML verdict drives the base score; heuristics nudge it.
        score = 1.0 - ml.fake_probability
        if ml.fake_probability >= 0.6:
            reasons.append(
                f"Classifier flags fake-like writing style (p={ml.fake_probability:.2f})."
            )
        elif ml.fake_probability <= 0.4:
            reasons.append(
                f"Classifier sees credible-style writing (p_fake={ml.fake_probability:.2f})."
            )
    if feats["clickbait_count"] >= 2:
        score -= 0.08
        reasons.append(f"Clickbait phrases x{int(feats['clickbait_count'])}.")
    if feats["exclamation_count"] >= 3:
        score -= 0.05
        reasons.append("Heavy exclamation use.")
    if feats["caps_ratio"] > 0.08:
        score -= 0.05
        reasons.append("Excessive ALL-CAPS words.")
    if feats["hedge_count"] >= 2:
        score += 0.05
        reasons.append("Uses hedged/attribution language (typical of careful reporting).")
    if feats["word_count"] < 40:
        reasons.append("Very short text limits stylistic signal.")

    score = max(0.0, min(1.0, score))

    signal = NLPSignal(
        score=round(score, 3),
        confidence=round(ml.confidence if ml.available else 0.4, 3),
        label=("credible" if score >= 0.6 else "fake-prone" if score <= 0.4 else "neutral"),
        important_words=ml.important_words,
        reasons=reasons,
        readability=round(feats["flesch_reading_ease"], 1),
        clickbait_score=round(min(1.0, feats["clickbait_density"]), 3),
        sentiment=round(feats["sentiment"], 3),
        exclamation_ratio=round(feats["exclamation_ratio"], 4),
        caps_ratio=round(feats["caps_ratio"], 4),
    )
    return signal, feats
