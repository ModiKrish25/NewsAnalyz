"""Tests for the decision engine (signal fusion + verdict rules)."""
from __future__ import annotations

from app.core.decision_engine import decide, fuse_scores
from app.schemas import (
    ClaimAssessment,
    EvidenceStance,
    MetadataSignal,
    MLSignal,
    NLPSignal,
    SourceSignal,
    SourceInfo,
    Verdict,
)


def _nlp(score=0.5, conf=0.5):
    return NLPSignal(score=score, confidence=conf)


def _src(score=0.5, conf=0.5):
    return SourceSignal(score=score, confidence=conf, source=SourceInfo(domain="x.com"))


def _claim(stance: EvidenceStance, score: float, conf: float = 0.8) -> ClaimAssessment:
    return ClaimAssessment(
        claim_id=1,
        text="t",
        stance=stance,
        evidence_score=score,
        confidence=conf,
    )


class TestFuseScores:
    def test_evidence_dominates_when_strong(self):
        fused, breakdown = fuse_scores(
            nlp=_nlp(0.2, 0.9),
            source=_src(0.2, 0.8),
            evidence_score=1.0,
            evidence_coverage=1.0,
            n_evidence=5,
        )
        assert fused > 0.3
        assert len(breakdown.contributions) == 3

    def test_no_evidence_pulls_to_neutral(self):
        fused, _ = fuse_scores(
            nlp=_nlp(0.9, 0.9),
            source=_src(0.9, 0.8),
            evidence_score=0.0,
            evidence_coverage=0.0,
            n_evidence=0,
        )
        assert fused < 0.6  # strong style/source alone cannot fully convince

    def test_all_contradicting_evidence_pushes_negative(self):
        fused, _ = fuse_scores(
            nlp=_nlp(0.8, 0.9),
            source=_src(0.8, 0.8),
            evidence_score=-1.0,
            evidence_coverage=1.0,
            n_evidence=4,
        )
        assert fused < 0.0


class TestDecide:
    def test_supports(self):
        fused, breakdown = fuse_scores(
            nlp=_nlp(0.7), source=_src(0.8), evidence_score=0.8, evidence_coverage=0.8, n_evidence=4
        )
        verdict, conf, bd = decide(
            fused, [_claim(EvidenceStance.SUPPORTS, 0.8)], MLSignal(), MetadataSignal(), breakdown
        )
        assert verdict == Verdict.SUPPORTS

    def test_conflicts(self):
        fused, breakdown = fuse_scores(
            nlp=_nlp(0.3), source=_src(0.4), evidence_score=-0.8, evidence_coverage=0.8, n_evidence=4
        )
        verdict, conf, bd = decide(
            fused, [_claim(EvidenceStance.CONTRADICTS, -0.8)], MLSignal(), MetadataSignal(), breakdown
        )
        assert verdict == Verdict.CONFLICTS

    def test_majority_contradicted_claims_force_conflicts(self):
        fused, breakdown = fuse_scores(
            nlp=_nlp(0.9), source=_src(0.9), evidence_score=0.0, evidence_coverage=0.3, n_evidence=2
        )
        claims = [
            _claim(EvidenceStance.CONTRADICTS, -0.9),
            _claim(EvidenceStance.CONTRADICTS, -0.9),
            _claim(EvidenceStance.UNRELATED, 0.0),
        ]
        verdict, conf, bd = decide(fused, claims, MLSignal(), MetadataSignal(), breakdown)
        assert verdict == Verdict.CONFLICTS

    def test_all_unrelated_claims_force_unknown(self):
        fused, breakdown = fuse_scores(
            nlp=_nlp(0.9), source=_src(0.9), evidence_score=0.0, evidence_coverage=0.0, n_evidence=0
        )
        claims = [_claim(EvidenceStance.UNRELATED, 0.0), _claim(EvidenceStance.UNRELATED, 0.0)]
        verdict, conf, bd = decide(fused, claims, MLSignal(), MetadataSignal(), breakdown)
        assert verdict == Verdict.UNKNOWN

    def test_rule_trail_is_populated(self):
        fused, breakdown = fuse_scores(
            nlp=_nlp(0.7), source=_src(0.7), evidence_score=0.7, evidence_coverage=0.7, n_evidence=3
        )
        _, _, bd = decide(
            fused, [_claim(EvidenceStance.SUPPORTS, 0.7)], MLSignal(), MetadataSignal(), breakdown
        )
        assert bd.rule_trail
