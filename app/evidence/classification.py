"""Step 3 of the Evidence Engine: Evidence Classification.

For each retrieved item we decide: SUPPORTS / CONTRADICTS / UNRELATED.

The rules are deliberately transparent (no black box): relevance is lexical
overlap between the claim and the evidence, and stance comes from explicit
linguistic markers ("no evidence", "denied", "confirmed", "according to", ...).
Every item carries a human-readable rationale that flows into the final report.
"""
from __future__ import annotations

import re

from app.core.nlp import content_words, get_nlp
from app.schemas import EvidenceItem, EvidenceStance

# Strong contradiction language
_CONTRADICT_PAT = re.compile(
    r"\b(no evidence|no link|no connection|not true|untrue|false|fake|"
    r"debunk\w*|refut\w*|den(y|ies|ied|ial)|myth|hoax|baseless|"
    r"unfounded|incorrect|inaccurate|misleading|contradict\w*|"
    r"contrary to|despite (claims|viral|widely)|never (said|announced|happened))\b",
    re.IGNORECASE,
)
# Support / corroboration language
_SUPPORT_PAT = re.compile(
    r"\b(confirmed|confirms|according to|verified|consistent with|"
    r"corroborat\w*|in line with|announced that|reported that|"
    r"stated that|shows that|found that|data shows|study found)\b",
    re.IGNORECASE,
)
_FACTCHECK_PAT = re.compile(r"\bfact[- ]?check\w*\b", re.IGNORECASE)

_NUM_RE = re.compile(r"\d+(?:\.\d+)?%?")


def _entity_texts(text: str) -> list[str]:
    nlp = get_nlp()
    if not nlp.available:
        return []
    doc = nlp(text)
    return [e.text.lower() for e in doc.ents if e.label_ in
            {"PERSON", "ORG", "GPE", "LOC", "EVENT", "LAW", "PRODUCT"}]


def relevance_score(claim: str, evidence_text: str, claim_entities: list[str]) -> float:
    """Lexical relevance in [0, 1]: content-word overlap + entity boosts."""
    claim_terms = set(content_words(claim))
    ev_terms = set(content_words(evidence_text))
    if not claim_terms or not ev_terms:
        return 0.0

    overlap = claim_terms & ev_terms
    base = len(overlap) / (len(claim_terms) ** 0.5 * len(ev_terms) ** 0.5)  # cosine-ish

    ev_lower = evidence_text.lower()
    entity_boost = 0.0
    for ent in claim_entities:
        if ent.lower() in ev_lower:
            entity_boost += 0.08  # each matched entity is a strong topical anchor

    return float(min(1.0, base + entity_boost))


def classify_evidence(claim: str, claim_entities: list[str], item: EvidenceItem) -> EvidenceItem:
    """Fill stance, confidence, rationale and relevance on one evidence item."""
    evidence_text = f"{item.title}. {item.snippet}".strip()
    item.relevance = relevance_score(claim, evidence_text, claim_entities)

    claim_terms = set(content_words(claim))
    shared = claim_terms & set(content_words(evidence_text))

    contradicts = bool(_CONTRADICT_PAT.search(evidence_text))
    supports = bool(_SUPPORT_PAT.search(evidence_text))
    factcheck = bool(_FACTCHECK_PAT.search(evidence_text))

    # Numbers: a matching figure is corroboration; relevant evidence citing a
    # different figure for the same topic is a soft contradiction signal.
    claim_nums = set(_NUM_RE.findall(claim))
    ev_nums = set(_NUM_RE.findall(evidence_text))
    number_match = bool(claim_nums & ev_nums) if claim_nums else False

    if item.relevance < 0.08:
        item.stance = EvidenceStance.UNRELATED
        item.stance_confidence = 0.7
        item.stance_rationale = "Topically unrelated (lexical overlap too low)."
        return item

    if contradicts and (item.relevance >= 0.15 or shared):
        strength = len(_CONTRADICT_PAT.findall(evidence_text))
        item.stance = EvidenceStance.CONTRADICTS
        item.stance_confidence = min(0.95, 0.55 + 0.15 * strength + 0.3 * item.relevance)
        item.stance_rationale = (
            f"Contains negating language (e.g. {_CONTRADICT_PAT.findall(evidence_text)[:2]}) "
            f"while sharing {len(shared)} topic terms."
        )
        return item

    # fact-check headline WITHOUT negating language → usually affirms the claim
    if factcheck and (supports or number_match) and item.relevance >= 0.15:
        item.stance = EvidenceStance.SUPPORTS
        item.stance_confidence = min(0.9, 0.6 + 0.2 * item.relevance)
        item.stance_rationale = "Fact-check source affirms the claim."
        return item

    if (supports or number_match) and item.relevance >= 0.18 and len(shared) >= 2:
        item.stance = EvidenceStance.SUPPORTS
        item.stance_confidence = min(0.9, 0.5 + 0.25 * item.relevance + (0.1 if number_match else 0))
        item.stance_rationale = (
            "Corroborating language"
            + (" and matching figures" if number_match else "")
            + f"; shares {len(shared)} topic terms."
        )
        return item

    if item.relevance >= 0.30:
        # On-topic background with no stance markers: weak corroboration only.
        item.stance = EvidenceStance.SUPPORTS
        item.stance_confidence = 0.35
        item.stance_rationale = (
            "On-topic background evidence without explicit stance markers "
            "(weak corroboration)."
        )
        return item

    item.stance = EvidenceStance.UNRELATED
    item.stance_confidence = 0.6
    item.stance_rationale = "Not enough overlap to judge stance."
    return item


def classify_all(claim: str, claim_entities: list[str], items: list[EvidenceItem]) -> list[EvidenceItem]:
    return [classify_evidence(claim, claim_entities, it) for it in items]
