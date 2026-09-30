"""Step 1 of the Evidence Engine: Article/Video → Claims.

Extracts discrete, checkable claims from raw content using a mix of:
  - spaCy NER (who/what/where claims)
  - syntactic cues (reported speech, numbers, dates, superlatives)
  - verifiability classification (claim vs opinion vs speculative)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.nlp import get_nlp

# Sentences that look like opinions or speculation, not checkable claims
_OPINION_PAT = re.compile(
    r"\b(i (think|feel|believe|guess)|in my opinion|personally|"
    r"seems like|looks like|probably|maybe|might be|could be|"
    r"i'?m sure|no doubt|obviously|everyone knows)\b",
    re.IGNORECASE,
)
_SPECULATIVE_PAT = re.compile(
    r"\b(rumor(s|ed)?|allegedly|reportedly claims|conspiracy|"
    r"what if|some say|people are saying|sources suggest|"
    r"it is believed|unconfirmed)\b",
    re.IGNORECASE,
)
# Markers that a sentence contains a checkable factual assertion
_CHECKABLE_PAT = re.compile(
    r"\b(announced|reported|said|stated|confirmed|denied|signed|"
    r"passed|banned|launched|won|lost|died|born|opened|released|"
    r"stud(y|ies)|shows|found|according to|percent|%|million|billion|"
    r"\d{4}|yesterday|today|last (week|month|year)|this (week|month|year))\b",
    re.IGNORECASE,
)

_QUESTION_PAT = re.compile(r"^\s*(who|what|when|where|why|how|is|are|do|does|did|can|will)\b.*\?", re.IGNORECASE)

MAX_SENTENCE_TOKENS = 80
MIN_SENTENCE_CHARS = 25


@dataclass
class RawClaim:
    text: str
    entities: list[str] = field(default_factory=list)
    verifiability: str = "claim"  # claim | opinion | speculative
    score: float = 0.0            # higher = more checkable


def _sentence_entities(sent) -> list[str]:
    """Extract named entities as the anchor points for evidence search."""
    ents: list[str] = []
    for ent in getattr(sent, "ents", []):
        label = ent.label_
        if label in {"PERSON", "ORG", "GPE", "LOC", "EVENT", "DATE", "PERCENT", "CARDINAL", "LAW", "PRODUCT", "WORK_OF_ART"}:
            text = ent.text.strip()
            if len(text) > 1 and text.lower() not in {e.lower() for e in ents}:
                ents.append(text)
    return ents


def _classify_verifiability(sentence: str, has_entities: bool) -> tuple[str, float]:
    """Return (verifiability, checkable_score) for one sentence."""
    score = 0.0

    if _CHECKABLE_PAT.search(sentence):
        score += 2.0
    if re.search(r"\d", sentence):
        score += 1.0
    if has_entities:
        score += 1.5

    if _SPECULATIVE_PAT.search(sentence):
        return "speculative", 0.5
    if _OPINION_PAT.search(sentence):
        return "opinion", 1.0

    return "claim", score


def extract_claims(text: str, max_claims: int = 5) -> list[RawClaim]:
    """Split content into sentences and pick the most checkable ones."""
    if not text or not text.strip():
        return []

    doc = get_nlp()(text)

    candidates: list[RawClaim] = []
    for sent in doc.sents:
        sentence = sent.text.strip()
        if _QUESTION_PAT.match(sentence):
            continue
        if len(sentence) < MIN_SENTENCE_CHARS:
            continue
        if len(sentence.split()) > MAX_SENTENCE_TOKENS:
            sentence = " ".join(sentence.split()[:MAX_SENTENCE_TOKENS])

        entities = _sentence_entities(sent)
        verifiability, score = _classify_verifiability(sentence, bool(entities))
        if verifiability == "claim" and not entities:
            # A factual claim with no entities is hard to search for
            score -= 1.0

        candidates.append(
            RawClaim(text=sentence, entities=entities, verifiability=verifiability, score=score)
        )

    claims = [c for c in candidates if c.verifiability == "claim" and c.score > 1.0]
    claims.sort(key=lambda c: c.score, reverse=True)

    if not claims:
        # Fallback: return the most entity-dense sentences as weak claims
        fallback = sorted(candidates, key=lambda c: len(c.entities), reverse=True)
        claims = [c for c in fallback if c.entities][:max_claims]

    return claims[:max_claims]
