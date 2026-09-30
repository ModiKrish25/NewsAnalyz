"""SQLite-backed analysis history (powers the admin dashboard)."""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone

from app.config import settings
from app.schemas import Report

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    request_id TEXT PRIMARY KEY,
    created_at REAL,
    verdict TEXT,
    confidence REAL,
    content_score REAL,
    source_score REAL,
    evidence_score REAL,
    n_claims INTEGER,
    n_support INTEGER,
    n_contradict INTEGER,
    n_unknown INTEGER,
    processing_ms INTEGER,
    report_json TEXT
);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(_SCHEMA)


def save_analysis(report: Report) -> None:
    """Persist a report; storage problems never break the pipeline."""
    try:
        init_db()
        n_sup = sum(1 for c in report.claims if c.stance.value == "SUPPORTS")
        n_con = sum(1 for c in report.claims if c.stance.value == "CONTRADICTS")
        n_unk = sum(1 for c in report.claims if c.stance.value == "UNRELATED")
        with _conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO analyses
                   (request_id, created_at, verdict, confidence, content_score,
                    source_score, evidence_score, n_claims, n_support,
                    n_contradict, n_unknown, processing_ms, report_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    report.request_id,
                    report.created_at,
                    report.verdict.value,
                    report.confidence,
                    report.content_score,
                    report.source_score,
                    report.evidence_score,
                    len(report.claims),
                    n_sup,
                    n_con,
                    n_unk,
                    report.processing_ms,
                    report.model_dump_json(),
                ),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to persist analysis: %s", exc)


def list_analyses(limit: int = 100) -> list[dict]:
    try:
        with _conn() as conn:
            rows = conn.execute(
                """SELECT request_id, created_at, verdict, confidence,
                          content_score, source_score, evidence_score,
                          n_claims, n_support, n_contradict, n_unknown,
                          processing_ms
                   FROM analyses ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to list analyses: %s", exc)
        return []


def verdict_stats() -> dict:
    rows = list_analyses(limit=1000)
    if not rows:
        return {"total": 0}
    verdicts = [r["verdict"] for r in rows]
    return {
        "total": len(rows),
        "SUPPORTS": verdicts.count("SUPPORTS"),
        "CONFLICTS": verdicts.count("CONFLICTS"),
        "UNKNOWN": verdicts.count("UNKNOWN"),
        "avg_confidence": round(sum(r["confidence"] for r in rows) / len(rows), 3),
        "avg_processing_ms": round(sum(r["processing_ms"] for r in rows) / len(rows)),
    }


def load_report(request_id: str) -> Report | None:
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT report_json FROM analyses WHERE request_id = ?", (request_id,)
            ).fetchone()
        return Report.model_validate_json(row["report_json"]) if row else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to load report %s: %s", request_id, exc)
        return None
