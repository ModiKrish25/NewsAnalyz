"""JSONL audit log — one line per analysis, consumed by the admin dashboard."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from app.config import settings
from app.schemas import Report

logger = logging.getLogger(__name__)


def audit(report: Report) -> None:
    """Append a compact audit record; logging failures never break analysis."""
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "request_id": report.request_id,
            "verdict": report.verdict.value,
            "confidence": report.confidence,
            "content_score": report.content_score,
            "source_score": report.source_score,
            "evidence_score": report.evidence_score,
            "n_claims": len(report.claims),
            "processing_ms": report.processing_ms,
            "warnings": report.warning_indicators,
        }
        settings.audit_log.parent.mkdir(parents=True, exist_ok=True)
        with settings.audit_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("audit log write failed: %s", exc)


def read_audit(limit: int = 200) -> list[dict]:
    """Read the most recent audit records (oldest → newest)."""
    if not settings.audit_log.exists():
        return []
    try:
        lines = settings.audit_log.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("audit log read failed: %s", exc)
        return []
