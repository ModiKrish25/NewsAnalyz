"""ML MODEL LAYER.

Multiple classifiers vote on text features; the ensemble output is ONE of the
three signals feeding the decision engine — never the final word on its own.

Models (each stored separately in models/):
  - Logistic Regression + TF-IDF (also provides important words)
  - Multinomial Naive Bayes + TF-IDF
  - Random Forest on engineered features
  - Linear SVM + TF-IDF (calibrated to probabilities)
  - XGBoost on engineered features (optional, if xgboost installed)
  - DistilBERT zero/fine-tuned (optional, if transformers + torch installed)

If artifacts are missing the layer reports `available=False` and the pipeline
continues on heuristics + evidence, which is the intended degradation path.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib

from app.config import settings
from app.core.features import FEATURE_NAMES, build_features, features_to_vector
from app.schemas import EnsembleDetail, MLSignal

logger = logging.getLogger(__name__)
_MODELS_DIR = Path(settings.models_dir)


def _load(name: str):
    path = _MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to load %s: %s", path, exc)
        return None


class _Bundle:
    """Container for all loaded classic models."""

    def __init__(self) -> None:
        self.tfidf = _load("tfidf")
        self.logreg = _load("logreg")
        self.nb = _load("naive_bayes")
        self.svm = _load("linear_svc")
        self.rf = _load("random_forest")
        self.xgb = _load("xgboost")

    @property
    def any_available(self) -> bool:
        return any(m is not None for m in
                   (self.logreg, self.nb, self.svm, self.rf, self.xgb))


_bundle: _Bundle | None = None


def get_bundle() -> _Bundle:
    global _bundle
    if _bundle is None:
        _bundle = _Bundle()
    return _bundle


def reset_bundle() -> None:
    """Force reload after (re)training."""
    global _bundle
    _bundle = None


# ── Optional transformer (auto-detected) ─────────────────────────────────
_distilbert = {"pipe": None, "tried": False}


def _get_distilbert():
    if _distilbert["tried"]:
        return _distilbert["pipe"]
    _distilbert["tried"] = True
    try:
        from transformers import pipeline  # heavy import, only if installed

        _distilbert["pipe"] = pipeline(
            "text-classification",
            model="mrm8488/bert-tiny-finetuned-fake-news",
            truncation=True,
        )
        logger.info("DistilBERT fake-news pipeline loaded")
    except Exception as exc:  # noqa: BLE001
        logger.info("transformers unavailable or model download failed: %s", exc)
    return _distilbert["pipe"]


# ── Inference ────────────────────────────────────────────────────────────
def _proba_or_default(model, X, fallback: float = 0.5) -> float:
    """Get P(fake) from a model that exposes predict_proba or decision_function."""
    if model is None:
        return fallback
    try:
        if hasattr(model, "predict_proba"):
            classes = list(getattr(model, "classes_", [0, 1]))
            probs = model.predict_proba(X)[0]
            return float(probs[classes.index(1)]) if 1 in classes else float(probs[-1])
        # LinearSVC: squash decision value with a logistic
        import math

        margin = float(model.decision_function(X)[0])
        return 1.0 / (1.0 + math.exp(-margin))
    except Exception as exc:  # noqa: BLE001
        logger.warning("model predict failed: %s", exc)
        return fallback


def predict_ensemble(text: str, title: str = "") -> MLSignal:
    """Run all available models; aggregate into one MLSignal."""
    bundle = get_bundle()
    details: list[EnsembleDetail] = []
    important_words: list[str] = []

    if not text.strip():
        return MLSignal(available=False, models=details, confidence=0.0)

    X_text = bundle.tfidf.transform([text]) if bundle.tfidf is not None else None
    feats = build_features(text)
    X_feats = [features_to_vector(feats)]

    if bundle.logreg is not None and X_text is not None:
        p = _proba_or_default(bundle.logreg, X_text)
        details.append(EnsembleDetail(model="LogisticRegression+TFIDF", fake_probability=round(p, 4)))
        important_words = _doc_important_words(bundle.logreg, bundle.tfidf, X_text)
    if bundle.nb is not None and X_text is not None:
        p = _proba_or_default(bundle.nb, X_text)
        details.append(EnsembleDetail(model="NaiveBayes+TFIDF", fake_probability=round(p, 4)))
    if bundle.svm is not None and X_text is not None:
        p = _proba_or_default(bundle.svm, X_text)
        details.append(EnsembleDetail(model="LinearSVM+TFIDF", fake_probability=round(p, 4)))
    if bundle.rf is not None:
        p = _proba_or_default(bundle.rf, X_feats)
        details.append(EnsembleDetail(model="RandomForest+Features", fake_probability=round(p, 4)))
    if bundle.xgb is not None:
        p = _proba_or_default(bundle.xgb, X_feats)
        details.append(EnsembleDetail(model="XGBoost+Features", fake_probability=round(p, 4)))

    bert = _get_distilbert()
    if bert is not None:
        try:
            out = bert(text[:1000])[0]
            label = str(out.get("label", "")).lower()
            score = float(out.get("score", 0.5))
            p_fake = score if "fake" in label else 1.0 - score
            details.append(EnsembleDetail(model="DistilBERT", fake_probability=round(p_fake, 4)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("distilbert inference failed: %s", exc)

    if not details:
        return MLSignal(
            available=False,
            models=details,
            important_words=important_words,
            confidence=0.0,
        )

    # Agreement-weighted mean: models closer to the consensus get more say,
    # which dampens a single wildly-diverging classifier.
    probs = [d.fake_probability for d in details]
    mean_p = sum(probs) / len(probs)
    weights = [1.0 / (0.05 + abs(p - mean_p)) for p in probs]
    fake_p = sum(p * w for p, w in zip(probs, weights)) / sum(weights)
    spread = max(probs) - min(probs)
    confidence = max(0.2, 1.0 - spread) * min(1.0, len(details) / 3.0)

    return MLSignal(
        fake_probability=round(fake_p, 4),
        credible_probability=round(1.0 - fake_p, 4),
        confidence=round(confidence, 3),
        models=details,
        important_words=important_words,
        available=True,
    )


def _doc_important_words(logreg, tfidf, X, top_k: int = 8) -> list[str]:
    """Most fake-leaning words for THIS document via LR coefficients."""
    try:
        import numpy as np

        coefs = logreg.coef_[0]
        row = X.toarray()[0]
        contrib = row * coefs
        idx = np.argsort(contrib)[::-1][:top_k]
        names = tfidf.get_feature_names_out()
        return [str(names[i]) for i in idx if row[i] > 0]
    except Exception:  # noqa: BLE001
        return []


def top_important_words(text: str, top_k: int = 8) -> list[str]:
    """Explainable 'important words': LR-coefficient * tf-idf magnitude."""
    bundle = get_bundle()
    if bundle.logreg is None or bundle.tfidf is None:
        from app.core.nlp import content_words

        return content_words(text)[:top_k]
    X = bundle.tfidf.transform([text])
    return _doc_important_words(bundle.logreg, bundle.tfidf, X, top_k)


def models_trained() -> bool:
    return get_bundle().any_available
