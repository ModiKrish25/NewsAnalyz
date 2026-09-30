"""Train baseline models so the ML layer works out of the box.

Uses a bundled seed dataset (data/dataset_seed.csv) plus lightly-augmented
synthetic variations. Real projects should swap in LIAR / FakeNewsNet /
ISOT etc.; the script interface stays identical.

Usage:  python scripts/train_baseline.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import settings
from app.core.features import FEATURE_NAMES, build_features, features_to_vector

SEED_PATH = ROOT / "data" / "dataset_seed.csv"


def load_training_frame() -> pd.DataFrame:
    df = pd.read_csv(SEED_PATH)
    # Light synthetic augmentation: shuffle clauses to expand the seed set.
    rng = random.Random(42)
    rows = []
    for _, r in df.iterrows():
        rows.append({"text": r["text"], "label": int(r["label"])})
        parts = [p.strip() for p in str(r["text"]).split(".") if len(p.strip()) > 20]
        if len(parts) >= 2:
            for _ in range(3):
                combo = ". ".join(rng.sample(parts, k=min(len(parts), 2))) + "."
                rows.append({"text": combo, "label": int(r["label"])})
    return pd.DataFrame(rows)


def main() -> None:
    df = load_training_frame()
    y = df["label"].values
    texts = df["text"].tolist()
    print(f"Training on {len(df)} samples ({y.sum()} fake / {len(y) - y.sum()} real)")

    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True)
    X_text = tfidf.fit_transform(texts)

    X_feats = [features_to_vector(build_features(t)) for t in texts]

    logreg = LogisticRegression(max_iter=2000, C=1.0).fit(X_text, y)
    nb = MultinomialNB(alpha=0.3).fit(X_text, y)
    svm = CalibratedClassifierCV(LinearSVC(C=1.0), cv=3).fit(X_text, y)
    rf = RandomForestClassifier(n_estimators=250, random_state=42).fit(X_feats, y)

    bundle_dir = Path(settings.models_dir)
    joblib.dump(tfidf, bundle_dir / "tfidf.joblib")
    joblib.dump(logreg, bundle_dir / "logreg.joblib")
    joblib.dump(nb, bundle_dir / "naive_bayes.joblib")
    joblib.dump(svm, bundle_dir / "linear_svc.joblib")
    joblib.dump(rf, bundle_dir / "random_forest.joblib")

    # XGBoost (optional)
    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            eval_metric="logloss", random_state=42,
        ).fit(X_feats, y)
        joblib.dump(xgb, bundle_dir / "xgboost.joblib")
        print("XGBoost: trained")
    except Exception as exc:  # noqa: BLE001
        print(f"XGBoost skipped: {exc}")

    # Quick self-check
    from sklearn.metrics import classification_report
    for name, model, X in [
        ("logreg", logreg, X_text),
        ("naive_bayes", nb, X_text),
        ("linear_svc", svm, X_text),
        ("random_forest", rf, X_feats),
    ]:
        preds = model.predict(X)
        rep = classification_report(y, preds, output_dict=True, zero_division=0)
        print(f"  {name}: train-accuracy={rep['accuracy']:.3f}")

    print(f"Models saved to {bundle_dir}")


if __name__ == "__main__":
    main()
