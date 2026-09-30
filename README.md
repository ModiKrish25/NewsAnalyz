# TruthLens — Explainable Misinformation Detection

> **Evidence-first, not guess-first.** Instead of "the model thinks this sentence looks fake,"
> every claim is checked against retrieved evidence, and the final verdict fuses three
> independent signals: text, source, and evidence.

## Architecture

```
User / Admin
   │
   ▼
Streamlit Web App ──────── Admin Dashboard (analytics/logs)
   │
   ▼
FastAPI Backend
   │
   ├── Text Analysis (NLP)        ─┐
   ├── Source Analysis             ├─→ Feature Engineering → ML Model Layer → Ensemble
   └── Metadata Analysis          ─┘
                │
                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                    EVIDENCE ENGINE (core)                    │
   │   Article/Video → Claims → Evidence Retrieval →              │
   │   Evidence Classification → Per-claim Assessment             │
   └──────────────────────────────────────────────────────────────┘
                │
                ▼
        Decision Engine (SUPPORTS / CONFLICTS / UNKNOWN)
                │
                ▼
      Explainable Report → Result Page
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. (optional) news evidence: get a free key at newsapi.org
cp .env.example .env    # set NEWSAPI_KEY

# 3. Run the API
uvicorn app.main:app --reload

# 4. In a second terminal — the web app
streamlit run app/frontend/app.py

# 5. Admin dashboard
streamlit run app/frontend/admin.py
```

- API docs: http://localhost:8000/docs
- Web app:    http://localhost:8501

## The Evidence Engine

For every extracted claim:

```
CLAIM "XYZ organization announced ABC."
        ↓
  Search relevant evidence
        ↓
  ┌─────┼─────┐
  ▼     ▼     ▼
Src A  Src B  Src C
  │     │     │
SUPPORTS CONTRADICTS UNRELATED
  └─────┼─────┘
        ▼
  Evidence Model → Claim Assessment
```

Final scoring uses several signals — never one model:

```
        CONTENT
           │
  ┌────────┼────────┐
  ▼        ▼        ▼
 NLP    SOURCE   EVIDENCE
score    score    score
  └────────┼────────┘
           ▼
     Decision Engine
           │
   ┌───────┼────────┐
   ▼       ▼        ▼
SUPPORTS CONFLICTS UNKNOWN
```

## Project layout

```
app/
  main.py               FastAPI entrypoint
  config.py             Settings (.env)
  schemas.py            Pydantic models
  core/
    nlp.py              spaCy pipeline, entity extraction
    features.py         TF-IDF, n-grams, sentiment, readability, stylometry, clickbait
    ml.py               model layer (LR/NB/RF/SVM/XGB) + ensemble
    decision_engine.py  multi-signal fusion → SUPPORTS/CONFLICTS/UNKNOWN
    logging.py          JSONL audit log
    storage.py          SQLite history (SQLModel)
  evidence/
    claims.py           claim extraction (rule + NER based)
    retrieval.py        Wikipedia / NewsAPI / DuckDuckGo providers
    classification.py   Supports / Contradicts / Unrelated classifier
    engine.py           per-claim assessment loop
  analysis/
    text_analysis.py    content signal
    source_analysis.py  domain credibility signal
    metadata_analysis.py metadata signal
  pipeline.py           end-to-end orchestrator
  report.py             explainable report builder
  frontend/
    app.py              Streamlit UI
    admin.py            admin analytics dashboard
data/
models/                 trained model artifacts (.joblib)
tests/                  pytest suite
scripts/
  train_baseline.py     train on synthetic + heuristics (no dataset needed)
  run_demo.py           end-to-end demo without API keys
```

## Verdicts

| Verdict | Meaning |
|---|---|
| `SUPPORTS` | Evidence backs the claims |
| `CONFLICTS` | Evidence contradicts the claims |
| `UNKNOWN` | Not enough/neutral evidence — reported, not guessed |

## Notes

- Without a `NEWSAPI_KEY` the engine falls back to Wikipedia + DuckDuckGo evidence.
- `transformers`/`torch` are heavy; install only if you want DistilBERT in the ensemble
  (the code auto-detects availability).
- All analyses are logged to `data/logs/audit.jsonl` for the admin dashboard.
