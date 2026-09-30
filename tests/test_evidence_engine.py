"""Tests for the Evidence Engine (claims → retrieval → classification)."""
from __future__ import annotations

import pytest

from app.evidence.claims import extract_claims
from app.evidence.classification import classify_evidence, relevance_score
from app.evidence.retrieval import OfflineCorpusProvider, EvidenceRetriever
from app.schemas import EvidenceItem, EvidenceStance


class TestClaimExtraction:
    def test_extracts_checkable_claim(self):
        text = (
            "The World Health Organization announced a new vaccination framework on Monday. "
            "I think it seems like a good idea overall."
        )
        claims = extract_claims(text, max_claims=3)
        assert claims, "should extract at least one claim"
        assert any("announced" in c.text.lower() for c in claims)
        # Opinion sentence must not be selected as a claim
        assert all("i think" not in c.text.lower() for c in claims)

    def test_filters_opinion_and_questions(self):
        claims = extract_claims("Is this true? Maybe it seems like nothing happened.", max_claims=5)
        assert claims == []

    def test_empty_text(self):
        assert extract_claims("", max_claims=3) == []

    def test_entities_are_captured(self):
        text = "NASA confirmed the Parker Solar Probe completed its closest approach to the sun."
        claims = extract_claims(text, max_claims=1)
        assert claims
        joined = " ".join(claims[0].entities).lower()
        assert "nasa" in joined or "parker" in joined


class TestRetrieval:
    def test_offline_corpus_matches_topic(self):
        provider = OfflineCorpusProvider()
        items = provider.search("WHO announced ABC vaccination framework", k=3)
        assert items
        assert all(it.provider == "offline_corpus" for it in items)

    def test_offline_corpus_no_match(self):
        provider = OfflineCorpusProvider()
        items = provider.search("zzz qqq unknown topic words", k=3)
        assert items == []

    def test_dedupe_removes_duplicate_urls(self):
        r = EvidenceRetriever(offline_fallback=False)
        items = [
            EvidenceItem(source_name="A", url="https://x.com/1", title="t"),
            EvidenceItem(source_name="B", url="https://x.com/1", title="t"),
        ]
        assert len(r._dedupe(items)) == 1


class TestClassification:
    def test_contradiction_detection(self):
        item = EvidenceItem(
            source_name="Reuters",
            url="https://reuters.com/factcheck",
            title="Fact check: claim is false",
            snippet="Fact checkers found no evidence that the organization announced ABC. "
                    "The claim has been debunked and is baseless.",
        )
        out = classify_evidence(
            "XYZ organization announced ABC.", ["XYZ organization"], item
        )
        assert out.stance == EvidenceStance.CONTRADICTS

    def test_support_detection(self):
        item = EvidenceItem(
            source_name="WHO",
            url="https://who.int/news",
            title="WHO announces new vaccination framework",
            snippet="The World Health Organization announced a new vaccination framework "
                    "aimed at raising coverage according to officials.",
        )
        out = classify_evidence(
            "WHO announced a new vaccination framework.", ["WHO"], item
        )
        assert out.stance == EvidenceStance.SUPPORTS

    def test_unrelated_low_overlap(self):
        item = EvidenceItem(
            source_name="Sports",
            url="https://sports.com/x",
            title="Football match result",
            snippet="The team won the championship final after a penalty shootout.",
        )
        out = classify_evidence(
            "The central bank raised interest rates by 25 basis points.", ["central bank"], item
        )
        assert out.stance == EvidenceStance.UNRELATED

    def test_relevance_bounds(self):
        assert relevance_score("the cat sat", "the cat sat on the mat", []) >= 0
        assert relevance_score("", "anything", []) == 0.0
