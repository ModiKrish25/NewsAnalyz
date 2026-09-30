"""End-to-end orchestrator.

Article/Video
    → Claims → Evidence Retrieval → Evidence Classification → Claim Assessments
    → NLP + SOURCE + METADATA signals → Decision Engine → Explainable Report
"""
from __future__ import annotations

import logging
import time

from app.analysis.metadata_analysis import analyze_metadata
from app.analysis.source_analysis import analyze_source
from app.analysis.text_analysis import analyze_text
from app.config import settings
from app.core.decision_engine import decide, fuse_scores
from app.core.logging_utils import audit
from app.core.ml import predict_ensemble, top_important_words
from app.core.storage import save_analysis
from app.evidence.claims import extract_claims
from app.evidence.engine import assess_claims
from app.evidence.retrieval import EvidenceRetriever
from app.schemas import (
    AnalyzeRequest,
    EvidenceStance,
    MLSignal,
    Report,
    Verdict,
)

logger = logging.getLogger(__name__)


def _aggregate_evidence(claims) -> tuple[float, float, int, int, int, int]:
    """Claim-level scores → article-level evidence aggregate.

    Returns (evidence_score -1..+1, coverage 0..1, n_support, n_contradict, n_unrelated, n_items).
    """
    if not claims:
        return 0.0, 0.0, 0, 0, 0, 0

    weights = [max(c.confidence, 0.15) for c in claims]
    scores = [c.evidence_score for c in claims]
    fused = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    n_sup = sum(1 for c in claims if c.stance == EvidenceStance.SUPPORTS)
    n_con = sum(1 for c in claims if c.stance == EvidenceStance.CONTRADICTS)
    n_unrel = sum(1 for c in claims if c.stance == EvidenceStance.UNRELATED)
    n_items = sum(len(c.evidence) for c in claims)

    relevant = sum(
        len([e for e in c.evidence if e.stance != EvidenceStance.UNRELATED]) for c in claims
    )
    total = max(n_items, 1)
    coverage = min(1.0, relevant / total)
    return fused, coverage, n_sup, n_con, n_unrel, n_items


def run_pipeline(req: AnalyzeRequest) -> Report:
    """Analyze one submission through the full Evidence Engine pipeline."""
    start = time.perf_counter()
    text = (req.text or "").strip()
    if not text and req.url:
        text = _fetch_article_text(req.url)

    # ── 1. Evidence Engine: Article → Claims ──────────────────────────
    claims = extract_claims(text, max_claims=req.max_claims)

    # ── 2-3. Retrieval + Classification + per-claim assessment ────────
    retriever = EvidenceRetriever()
    assessments = assess_claims(claims, retriever)

    ev_score, ev_coverage, n_sup, n_con, n_unrel, n_items = _aggregate_evidence(assessments)

    # ── Signals ───────────────────────────────────────────────────────
    nlp_signal, feats = analyze_text(text)
    source_signal = analyze_source(req.url or "")
    metadata_signal = analyze_metadata(text, req.title, req.author, req.published)
    try:
        ml_signal = predict_ensemble(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ML layer failed: %s", exc)
        ml_signal = MLSignal(available=False)

    # ── Decision Engine ───────────────────────────────────────────────
    fused, breakdown = fuse_scores(
        nlp=nlp_signal,
        source=source_signal,
        evidence_score=ev_score,
        evidence_coverage=ev_coverage,
        n_evidence=n_items,
    )
    verdict, confidence, breakdown = decide(fused, assessments, ml_signal, metadata_signal, breakdown)

    # ── Explainable report ────────────────────────────────────────────
    warning_indicators = _warning_indicators(nlp_signal, metadata_signal, assessments)
    reasons = _top_reasons(breakdown, assessments, verdict)

    report = Report(
        request_id=f"req-{int(time.time() * 1000)}",
        verdict=verdict,
        confidence=confidence,
        content_score=nlp_signal.score,
        source_score=source_signal.score,
        evidence_score=round((ev_score + 1) / 2, 4),  # -1..1 → 0..1 for display
        decision=breakdown,
        nlp=nlp_signal,
        source=source_signal,
        metadata=metadata_signal,
        ml=ml_signal,
        claims=assessments,
        warning_indicators=warning_indicators,
        reasons=reasons,
        important_words=top_important_words(text),
        source_info=source_signal.source,
        processing_ms=int((time.perf_counter() - start) * 1000),
    )

    save_analysis(report)
    audit(report)
    return report


def _warning_indicators(nlp_signal, metadata_signal, assessments) -> list[str]:
    warns: list[str] = []
    if nlp_signal.clickbait_score >= 0.5:
        warns.append("High clickbait density")
    if nlp_signal.exclamation_ratio > 0.02:
        warns.append("Excessive exclamation marks")
    if nlp_signal.caps_ratio > 0.08:
        warns.append("Heavy ALL-CAPS usage")
    if not metadata_signal.has_author:
        warns.append("No author byline")
    if not metadata_signal.has_date:
        warns.append("No publication date")
    contradicted = [c for c in assessments if c.stance == EvidenceStance.CONTRADICTS]
    for c in contradicted:
        warns.append(f"Evidence contradicts claim {c.claim_id}: “{c.text[:70]}…”")
    return warns


def _top_reasons(breakdown, assessments, verdict: Verdict) -> list[str]:
    reasons: list[str] = []
    for c in sorted(breakdown.contributions, key=lambda x: -abs(x.weighted)):
        direction = "supports credibility" if c.weighted >= 0 else "hurts credibility"
        reasons.append(f"{c.signal.value} signal {direction} (weighted {c.weighted:+.3f}).")
    if assessments:
        best = max(assessments, key=lambda c: (abs(c.evidence_score), c.confidence))
        if best.evidence:
            top_ev = best.evidence[0]
            reasons.append(
                f"Strongest evidence for claim {best.claim_id}: “{top_ev.title or top_ev.source_name}” — {top_ev.stance_rationale}"
            )
    return reasons


def _fetch_article_text(url: str) -> str:
    """Best-effort article extraction when only a URL is provided."""
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            return trafilatura.extract(downloaded) or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("trafilatura extraction failed for %s: %s", url, exc)
    return ""
