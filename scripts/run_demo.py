"""End-to-end demo — works fully offline (no API keys needed).

Runs the complete Evidence Engine on two sample articles and prints the
claim → evidence → stance trail plus the fused decision.

Usage: python scripts/run_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles default to cp1252; force UTF-8 with safe fallback
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pipeline import run_pipeline
from app.schemas import AnalyzeRequest

SAMPLES = [
    {
        "title": "WHO announces new vaccination framework",
        "url": "https://example-news.com/who-announcement",
        "author": "Maria Chen",
        "published": "2026-09-28",
        "text": (
            "The World Health Organization announced a new vaccination framework "
            "at its annual assembly on Monday. Officials stated the program aims to "
            "raise coverage in developing regions from 72 percent to 86 percent by 2030. "
            "According to the WHO director, the initiative was confirmed by member states "
            "after a two-day session. The organization reported that funding has been "
            "secured for the first phase."
        ),
    },
    {
        "title": "SHOCKING: They don't want you to know this!!!",
        "url": "https://conspiracy-nut.blogspot.com/miracle-cure",
        "author": "",
        "published": "",
        "text": (
            "BREAKING: You won't believe what the government is hiding!!! Doctors HATE "
            "this one weird trick that cures everything overnight. The mainstream media "
            "refuses to report the TRUTH about this miracle cure. WAKE UP! Insiders "
            "expose the insane secret they're hiding from you. Share before it gets "
            "deleted!!!"
        ),
    },
]


def main() -> None:
    for sample in SAMPLES:
        print("=" * 78)
        print(f"ARTICLE: {sample['title']}")
        print("=" * 78)
        report = run_pipeline(AnalyzeRequest(**sample, max_claims=3))

        print(f"\nVERDICT: {report.verdict.value}  (confidence {report.confidence:.0%})")
        print(f"content={report.content_score:.2f}  source={report.source_score:.2f}  "
              f"evidence={report.evidence_score:.2f}  [{report.processing_ms} ms]")

        print("\nDecision breakdown:")
        for c in report.decision.contributions:
            print(f"  {c.signal.value:<9} score={c.score:+.2f} w={c.weight:<4} "
                  f"conf={c.confidence:.2f} -> {c.weighted:+.4f}")
        print(f"  fused={report.decision.fused_score:+.3f}")
        for rule in report.decision.rule_trail:
            print(f"  rule: {rule}")

        print("\nEvidence Engine:")
        if not report.claims:
            print("  (no checkable claims extracted)")
        for claim in report.claims:
            print(f"  CLAIM {claim.claim_id} [{claim.stance.value}] "
                  f"score={claim.evidence_score:+.2f} -- {claim.text[:80]}...")
            print(f"    entities: {', '.join(claim.entities) or '—'}")
            for ev in claim.evidence[:4]:
                print(f"      {ev.stance.value:<11} rel={ev.relevance:.2f} "
                      f"conf={ev.stance_confidence:.2f}  {ev.source_name}: {ev.title[:60]}")

        print("\nWarnings:")
        for w in report.warning_indicators or ["  (none)"]:
            print(f"  - {w}")
        print()


if __name__ == "__main__":
    main()
