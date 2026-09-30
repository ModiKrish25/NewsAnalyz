"""Shared spaCy pipeline (loaded once) and light NLP utilities."""
from __future__ import annotations

import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

_FALLBACK_LEMMAS = {
    "is": "be", "are": "be", "was": "be", "were": "be", "announced": "announce",
}


class NLPEngine:
    """Wraps spaCy; degrades to a regex sentence splitter when unavailable."""

    def __init__(self) -> None:
        self.nlp = None
        try:
            import spacy

            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy en_core_web_sm loaded")
        except Exception as exc:  # noqa: BLE001 — keep the app alive without spaCy
            logger.warning("spaCy unavailable (%s); falling back to regex NLP", exc)

    @property
    def available(self) -> bool:
        return self.nlp is not None

    def __call__(self, text: str):
        """Return a spaCy Doc, or a tiny shim with `.sents` when spaCy is missing."""
        if self.nlp is not None:
            return self.nlp(text)

        class _Sent:
            def __init__(self, s: str) -> None:
                self.text = s
                self.ents = []

        class _Doc:
            def __init__(self, s: str) -> None:
                self.sents = [_Sent(x) for x in _split_sentences(s)]

        return _Doc(text)


def _split_sentences(text: str) -> list[str]:
    #Abbreviation-aware naive splitter (good enough for fallback mode)
    protected = re.sub(r"\b(e\.g|i\.e|etc|vs|Dr|Mr|Mrs|Ms|St|No)\.", r"\1<DOT>", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


@lru_cache
def get_nlp() -> NLPEngine:
    return NLPEngine()


STOPWORDS: frozenset[str] = frozenset(
    """a about above after again against all am an and any are aren't as at be because been
    before being below between both but by can't cannot could couldn't did didn't do does
    doesn't doing don't down during each few for from further had hadn't has hasn't have
    haven't having he he'd he'll he's her here here's hers herself him himself his how
    how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
    mustn't my myself no nor not of off on once only or other ought our ours ourselves out
    over own same shan't she she'd she'll she's should shouldn't so some such than that
    that's the their theirs them themselves then there there's these they they'd they'll
    they're they've this those through to too under until up very was wasn't we we'd we'll
    we're we've were weren't what what's when when's where where's which while who who's
    whom why why's with won't would wouldn't you you'd you'll you're you've your yours
    yourself yourselves just really very much many also said says""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def content_words(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


_NEGATORS = {"not", "no", "never", "without", "denies", "denied", "deny", "refutes", "refuted", "false", "fake", "hoax", "myth", "debunked", "untrue", "incorrect", "misleading"}
_POSITIVE = {"support", "supports", "confirmed", "confirms", "verified", "true", "correct", "accurate", "according"}
_AFFECT_POS = {"good", "great", "amazing", "wonderful", "best", "win", "wins", "success", "successful", "hope", "improve", "improved", "breakthrough"}
_AFFECT_NEG = {"bad", "terrible", "awful", "worst", "lose", "loses", "lost", "failure", "failed", "crisis", "disaster", "scandal", "shocking", "outrageous", "evil", "dangerous"}


def lexicon_sentiment(text: str) -> float:
    """Tiny lexicon sentiment in [-1, 1]; no external model needed."""
    words = tokenize(text)
    if not words:
        return 0.0
    score = 0.0
    hits = 0
    for w in words:
        if w in _AFFECT_POS:
            score += 1.0
            hits += 1
        elif w in _AFFECT_NEG:
            score -= 1.0
            hits += 1
    return max(-1.0, min(1.0, score / max(hits, 1))) if hits else 0.0
