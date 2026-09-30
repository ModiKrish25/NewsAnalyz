"""FastAPI backend for TruthLens."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import settings
from app.core.ml import models_trained
from app.core.nlp import get_nlp
from app.core.storage import list_analyses, load_report, verdict_stats
from app.evidence.claims import extract_claims
from app.pipeline import run_pipeline
from app.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title=settings.app_name,
    description=(
        "Explainable misinformation detection built around an **Evidence Engine**: "
        "Article → Claims → Evidence Retrieval → Evidence Classification → "
        "Claim Assessment → Decision Engine → Report."
    ),
    version=__version__,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        spacy=get_nlp().available,
        models_trained=models_trained(),
        newsapi=bool(settings.newsapi_key),
        version=__version__,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Run the full Evidence Engine pipeline on one submission."""
    report = run_pipeline(req)
    return AnalyzeResponse(report=report)


@app.post("/claims")
def claims_preview(req: AnalyzeRequest) -> dict:
    """Extract claims only (step 1 of the Evidence Engine) — useful for debugging."""
    found = extract_claims(req.text, max_claims=req.max_claims)
    return {
        "claims": [
            {
                "text": c.text,
                "entities": c.entities,
                "verifiability": c.verifiability,
                "checkable_score": round(c.score, 2),
            }
            for c in found
        ]
    }


@app.get("/history")
def history(limit: int = 50) -> dict:
    return {"analyses": list_analyses(limit=limit)}


@app.get("/history/{request_id}")
def history_detail(request_id: str) -> dict:
    report = load_report(request_id)
    return {"report": report.model_dump() if report else None}


@app.get("/stats")
def stats() -> dict:
    return verdict_stats()
