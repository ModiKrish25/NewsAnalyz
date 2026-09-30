"""Step 4 of the Evidence Engine: per-claim assessment.

CLAIM → Search evidence → classify each item → aggregate into a claim-level
evidence score in [-1, +1] (-1 = fully contradicted, +1 = fully supported).
"""
from __future__ import annotations

import logging

from app.config import settings
from app.evidence.classification import classify_all
from app.evidence.retrieval import EvidenceRetriever
from app.schemas import ClaimAssessment, EvidenceStance

logger = logging.getLogger(__name__)


def _build_query(claim_text: str, entities: list[str]) -> str:
    """Evidence search query: named entities + remaining key terms."""
    from app.core.nlp import content_words

    ent_lower = {e.lower() for e in entities}
    key_terms = [w for w in content_words(claim_text) if w not in ent_lower]
    query_parts = entities[:3] + key_terms[:6]
    return " ".join(query_parts) if query_parts else claim_text


def assess_claim(
    claim_id: int,
    claim_text: str,
    entities: list[str],
    retriever: EvidenceRetriever,
) -> ClaimAssessment:
    """Run one claim through retrieval + classification + aggregation."""
    query = _build_query(claim_text, entities)
    raw_items = retriever.retrieve(query, k=settings.evidence_per_claim)
    items = classify_all(claim_text, entities, raw_items)

    # Most relevant first so the report highlights the best evidence
    items.sort(key=lambda it: (it.relevance * it.stance_confidence), reverse=True)

    support_w = sum(
        it.stance_confidence * (0.5 + 0.5 * it.relevance)
        for it in items if it.stance == EvidenceStance.SUPPORTS
    )
    contradict_w = sum(
        it.stance_confidence * (0.5 + 0.5 * it.relevance)
        for it in items if it.stance == EvidenceStance.CONTRADICTS
    )
    n_unrelated = sum(1 for it in items if it.stance == EvidenceStance.UNRELATED)

    n_sup = sum(1 for it in items if it.stance == EvidenceStance.SUPPORTS)
    n_con = sum(1 for it in items if it.stance == EvidenceStance.CONTRADICTS)

    total_w = support_w + contradict_w
    if total_w > 0:
        evidence_score = (support_w - contradict_w) / total_w  # -1 .. +1
    else:
        evidence_score = 0.0

    # Coverage: how many retrieved items were actually topically relevant
    relevant = n_sup + n_con
    coverage = relevant / max(len(items), 1) if items else 0.0

    if relevant < settings.min_evidence_for_verdict:
        stance = EvidenceStance.UNRELATED
        confidence = 0.2
        rationale = (
            f"Insufficient evidence retrieved ({relevant} relevant of {len(items)})."
        )
    elif evidence_score > 0.15:
        stance = EvidenceStance.SUPPORTS
        confidence = min(0.95, 0.4 + 0.4 * abs(evidence_score) + 0.3 * coverage)
        rationale = f"{n_sup} source(s) support, {n_con} contradict the claim."
    elif evidence_score < -0.15:
        stance = EvidenceStance.CONTRADICTS
        confidence = min(0.95, 0.4 + 0.4 * abs(evidence_score) + 0.3 * coverage)
        rationale = f"{n_con} source(s) contradict, {n_sup} support the claim."
    else:
        stance = EvidenceStance.UNRELATED
        confidence = 0.3
        rationale = "Evidence is mixed or neutral; no clear stance."

    return ClaimAssessment(
        claim_id=claim_id,
        text=claim_text,
        entities=entities,
        evidence=items,
        n_supports=n_sup,
        n_contradicts=n_con,
        n_unrelated=n_unrelated,
        evidence_score=round(evidence_score, 4),
        stance=stance,
        confidence=round(confidence, 3),
        rationale=rationale,
    )


def assess_claims(
    claims: list,
    retriever: EvidenceRetriever | None = None,
) -> list[ClaimAssessment]:
    """Assess every extracted claim; retrieval failures degrade gracefully."""
    retriever = retriever or EvidenceRetriever()
    results: list[ClaimAssessment] = []
    for i, claim in enumerate(claims):
        try:
            results.append(assess_claim(i + 1, claim.text, claim.entities, retriever))
        except Exception as exc:  # noqa: BLE001 — one claim must not kill the report
            logger.warning("claim assessment failed for %r: %s", claim.text[:60], exc)
            results.append(
                ClaimAssessment(
                    claim_id=i + 1,
                    text=claim.text,
                    entities=claim.entities,
                    rationale=f"Evidence lookup failed: {exc}",
                )
            )
    return results
