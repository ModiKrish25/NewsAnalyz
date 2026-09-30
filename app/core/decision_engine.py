"""DECISION ENGINE.

Fuses the three independent signals (NLP / SOURCE / EVIDENCE) into a final
verdict: SUPPORTS / CONFLICTS / UNKNOWN. The evidence signal dominates by
default (weights are configurable via .env), and every rule that fires is
recorded in `rule_trail` so the report is fully explainable.
"""
from __future__ import annotations

from app.config import settings
from app.schemas import (
    ClaimAssessment,
    DecisionBreakdown,
    EvidenceStance,
    MetadataSignal,
    MLSignal,
    NLPSignal,
    SignalContribution,
    SignalKind,
    SourceSignal,
    Verdict,
)


def fuse_scores(
    nlp: NLPSignal,
    source: SourceSignal,
    evidence_score: float,          # -1..+1 (claim-level aggregate)
    evidence_coverage: float,       # 0..1 share of relevant evidence
    n_evidence: int,
) -> tuple[float, DecisionBreakdown]:
    """Confidence-weighted fusion. Returns (fused_score -1..+1, breakdown).

    The evidence signal is the only one whose *weight* scales with its own
    confidence (base 0.5 → up to 1.0): abundant, relevant evidence dominates
    the verdict, while thin evidence shrinks toward zero influence so that
    style/source signals alone can never manufacture strong support.
    """
    weights = {
        SignalKind.NLP: settings.nlp_weight,
        SignalKind.SOURCE: settings.source_weight,
        SignalKind.EVIDENCE: settings.evidence_weight,
    }

    # Evidence confidence scales with how much relevant evidence we actually got
    ev_conf = min(1.0, evidence_coverage + (0.15 if n_evidence >= 3 else 0.0))
    nlp_conf = nlp.confidence
    src_conf = source.confidence

    eff_ev_weight = weights[SignalKind.EVIDENCE] * (1.0 + ev_conf)

    contributions = [
        SignalContribution(
            signal=SignalKind.EVIDENCE,
            score=round(evidence_score, 4),
            weight=round(eff_ev_weight, 3),
            weighted=round(evidence_score * eff_ev_weight * ev_conf, 4),
            confidence=round(ev_conf, 3),
            reasons=[f"{n_evidence} evidence items, coverage {evidence_coverage:.0%}"],
        ),
        SignalContribution(
            signal=SignalKind.NLP,
            score=round(nlp.score * 2 - 1, 4),  # 0..1 → -1..+1
            weight=weights[SignalKind.NLP],
            weighted=round((nlp.score * 2 - 1) * weights[SignalKind.NLP] * nlp_conf, 4),
            confidence=nlp_conf,
            reasons=nlp.reasons[:3],
        ),
        SignalContribution(
            signal=SignalKind.SOURCE,
            score=round(source.score * 2 - 1, 4),
            weight=weights[SignalKind.SOURCE],
            weighted=round((source.score * 2 - 1) * weights[SignalKind.SOURCE] * src_conf, 4),
            confidence=src_conf,
            reasons=source.reasons[:3],
        ),
    ]

    # Weights sum to ~1 by construction (0.5..1.0 + 0.3 + 0.2), so the fused
    # score stays on the same -1..+1 scale without renormalization.
    fused = sum(c.weighted for c in contributions)

    return fused, DecisionBreakdown(
        contributions=contributions,
        fused_score=round(fused, 4),
    )


def decide(
    fused: float,
    claims: list[ClaimAssessment],
    ml: MLSignal,
    metadata: MetadataSignal,
    breakdown: DecisionBreakdown,
) -> tuple[Verdict, float, DecisionBreakdown]:
    """Apply verdict rules on top of the fused score. Returns (verdict, confidence, breakdown)."""
    rules: list[str] = list(breakdown.rule_trail)

    n_support = sum(1 for c in claims if c.stance == EvidenceStance.SUPPORTS)
    n_contradict = sum(1 for c in claims if c.stance == EvidenceStance.CONTRADICTS)
    n_unknown = sum(1 for c in claims if c.stance == EvidenceStance.UNRELATED)

    margin = abs(fused)
    verdict = Verdict.UNKNOWN
    confidence = 0.3

    band = settings.uncertain_band
    if fused >= max(settings.support_threshold, band):
        verdict = Verdict.SUPPORTS
        confidence = min(0.95, 0.45 + 0.5 * margin)
        rules.append(f"Fused score {fused:+.2f} >= support threshold {settings.support_threshold:+.2f}.")
    elif fused <= settings.conflict_threshold:
        verdict = Verdict.CONFLICTS
        confidence = min(0.95, 0.45 + 0.5 * margin)
        rules.append(f"Fused score {fused:+.2f} <= conflict threshold {settings.conflict_threshold:+.2f}.")
    else:
        rules.append(f"Fused score {fused:+.2f} inside uncertain band [{settings.conflict_threshold:+.2f}, {settings.support_threshold:+.2f}].")

    # ── Safeguards (can flip a verdict toward UNKNOWN) ────────────────
    if claims and n_unknown == len(claims):
        verdict = Verdict.UNKNOWN
        confidence = min(confidence, 0.35)
        rules.append("No claim received a resolvable evidence stance → UNKNOWN.")

    if claims and n_contradict > 0 and n_contradict >= n_support:
        # Contradicted claims on balance → at best UNCERTAIN, never SUPPORTS
        if verdict == Verdict.SUPPORTS:
            verdict = Verdict.UNKNOWN
            confidence = min(confidence, 0.45)
            rules.append(f"{n_contradict} claim(s) contradicted by evidence → downgraded to UNKNOWN.")

    if n_contradict > 0 and n_contradict > len(claims) / 2.0:
        verdict = Verdict.CONFLICTS
        confidence = max(confidence, 0.55)
        rules.append(f"Majority of claims ({n_contradict}/{len(claims)}) contradicted by evidence → CONFLICTS.")

    # Model disagreement lowers confidence but does not flip the verdict
    if ml.available and ml.models:
        probs = [m.fake_probability for m in ml.models]
        if max(probs) - min(probs) > 0.5:
            confidence *= 0.8
            rules.append("ML models disagree strongly; confidence reduced.")

    breakdown.rule_trail = rules
    return verdict, round(confidence, 3), breakdown
