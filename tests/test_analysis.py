"""Tests for features + analysis signals."""
from __future__ import annotations

from app.analysis.metadata_analysis import analyze_metadata
from app.analysis.source_analysis import analyze_source
from app.core.features import build_features, flesch_reading_ease
from app.core.nlp import content_words, lexicon_sentiment, tokenize


class TestFeatures:
    def test_clickbait_detected(self):
        feats = build_features("SHOCKING: You won't believe this miracle cure!!! Doctors HATE it!")
        assert feats["clickbait_count"] >= 2
        assert feats["exclamation_count"] >= 2

    def test_neutral_news_style(self):
        feats = build_features(
            "The central bank raised its benchmark interest rate by 25 basis points, "
            "according to an official statement released on Tuesday."
        )
        assert feats["clickbait_count"] == 0
        assert feats["hedge_count"] >= 1

    def test_readability_range(self):
        score = flesch_reading_ease("The cat sat on the mat. It was warm.")
        assert 0 <= score <= 121.22

    def test_tokenize_filters_punct(self):
        toks = tokenize("Hello, world! It's 2026.")
        assert "hello" in toks and "2026" in toks

    def test_content_words_drop_stopwords(self):
        words = content_words("The quick brown fox jumps over the lazy dog")
        assert "quick" in words and "the" not in words


class TestSentiment:
    def test_positive(self):
        assert lexicon_sentiment("great success win improve") > 0

    def test_negative(self):
        assert lexicon_sentiment("terrible failure disaster crisis") < 0

    def test_neutral(self):
        assert lexicon_sentiment("the report lists figures and dates") == 0.0


class TestSourceAnalysis:
    def test_known_outlet(self):
        sig = analyze_source("https://www.reuters.com/article")
        assert sig.score >= 0.9
        assert sig.source.known_outlet == "Reuters"

    def test_satire(self):
        sig = analyze_source("https://waterfordwhispersnews.com/x")
        assert sig.score <= 0.2

    def test_gov_domain(self):
        sig = analyze_source("https://www.cdc.gov/outbreak")
        assert sig.score >= 0.85

    def test_unknown_domain_neutral(self):
        sig = analyze_source("https://some-random-site.xyz/post")
        assert 0.3 <= sig.score <= 0.7

    def test_empty_url(self):
        sig = analyze_source("")
        assert sig.score == 0.5


class TestMetadata:
    def test_complete_metadata_scores_higher(self):
        full = analyze_metadata("word " * 300, title="T", author="A", published="2026-01-01")
        bare = analyze_metadata("word " * 10)
        assert full.score > bare.score
        assert full.has_author and full.has_date

    def test_short_text_flags(self):
        sig = analyze_metadata("tiny")
        assert sig.word_count < 10
