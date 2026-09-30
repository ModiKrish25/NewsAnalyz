"""Pydantic models shared across the whole TruthLens pipeline."""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────
class Verdict(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONFLICTS = "CONFLICTS"
    UNKNOWN = "UNKNOWN"


class EvidenceStance(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    UNRELATED = "UNRELATED"


class SignalKind(str, Enum):
    NLP = "NLP"
    SOURCE = "SOURCE"
    EVIDENCE = "EVIDENCE"
    METADATA = "METADATA"


# ── Evidence Engine models ───────────────────────────────────────────────
class EvidenceItem(BaseModel):
    """A single retrieved piece of evidence for a claim."""
    source_name: str
    url: str
    title: str = ""
    snippet: str = ""
    published: Optional[str] = None
    provider: str = ""                      # wikipedia | newsapi | duckduckgo | corpus
    stance: EvidenceStance = EvidenceStance.UNRELATED
    stance_confidence: float = 0.0
    stance_rationale: str = ""
    relevance: float = 0.0                  # 0..1 lexical similarity to claim


class ClaimAssessment(BaseModel):
    """Result of running one claim through the Evidence Engine."""
    claim_id: int
    text: str
    entities: list[str] = Field(default_factory=list)
    verifiability: str = "claim"            # claim | opinion | speculative
    evidence: list[EvidenceItem] = Field(default_factory=list)
    n_supports: int = 0
    n_contradicts: int = 0
    n_unrelated: int = 0
    evidence_score: float = 0.0             # -1 .. +1 (contradicts .. supports)
    stance: EvidenceStance = EvidenceStance.UNRELATED
    confidence: float = 0.0                 # 0..1
    rationale: str = ""


# ── Analysis signals ─────────────────────────────────────────────────────
class NLPSignal(BaseModel):
    """Content-based (text-only) analysis."""
    score: float = 0.0                      # 0 = looks fake-prone, 1 = looks credible
    confidence: float = 0.5
    label: str = ""                         # fake-prone | credible | neutral
    important_words: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    readability: float = 0.0
    clickbait_score: float = 0.0
    sentiment: float = 0.0
    exclamation_ratio: float = 0.0
    caps_ratio: float = 0.0


class SourceInfo(BaseModel):
    url: str = ""
    domain: str = ""
    known_outlet: Optional[str] = None
    credibility: float = 0.5                # 0..1
    reasons: list[str] = Field(default_factory=list)


class SourceSignal(BaseModel):
    score: float = 0.5
    confidence: float = 0.5
    source: SourceInfo = Field(default_factory=SourceInfo)
    reasons: list[str] = Field(default_factory=list)


class MetadataSignal(BaseModel):
    score: float = 0.5
    confidence: float = 0.3
    word_count: int = 0
    sentence_count: int = 0
    has_author: bool = False
    has_date: bool = False
    has_title: bool = False
    language: str = "en"
    reasons: list[str] = Field(default_factory=list)


class EnsembleDetail(BaseModel):
    """Per-model outputs inside the ML layer."""
    model: str
    fake_probability: float
    available: bool = True


class MLSignal(BaseModel):
    """Output of the ML model layer / ensemble on text features."""
    fake_probability: float = 0.5
    credible_probability: float = 0.5
    confidence: float = 0.0
    models: list[EnsembleDetail] = Field(default_factory=list)
    important_words: list[str] = Field(default_factory=list)
    available: bool = False


# ── Decision & report ────────────────────────────────────────────────────
class SignalContribution(BaseModel):
    signal: SignalKind
    score: float
    weight: float
    weighted: float
    confidence: float
    reasons: list[str] = Field(default_factory=list)


class DecisionBreakdown(BaseModel):
    contributions: list[SignalContribution] = Field(default_factory=list)
    support_evidence: int = 0
    contradict_evidence: int = 0
    unrelated_evidence: int = 0
    fused_score: float = 0.0
    margin: float = 0.0
    rule_trail: list[str] = Field(default_factory=list)


class Report(BaseModel):
    """Final explainable report shown on the result page."""
    request_id: str
    verdict: Verdict
    confidence: float
    content_score: float                    # NLP signal
    source_score: float
    evidence_score: float
    decision: DecisionBreakdown
    nlp: NLPSignal
    source: SourceSignal
    metadata: MetadataSignal
    ml: MLSignal
    claims: list[ClaimAssessment] = Field(default_factory=list)
    warning_indicators: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    important_words: list[str] = Field(default_factory=list)
    source_info: SourceInfo
    processing_ms: int = 0
    created_at: float = Field(default_factory=time.time)


class AnalyzeRequest(BaseModel):
    text: str = ""
    url: str = ""
    title: str = ""
    author: str = ""
    published: str = ""
    max_claims: int = Field(default=5, ge=1, le=15)


class AnalyzeResponse(BaseModel):
    report: Report


class HealthResponse(BaseModel):
    status: str = "ok"
    spacy: bool = False
    models_trained: bool = False
    newsapi: bool = False
    version: str = "0.1.0"
