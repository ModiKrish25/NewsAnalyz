"""Central configuration (env-driven, with sane defaults for offline use)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOG_DIR = DATA_DIR / "logs"
MODELS_DIR = ROOT / "models"
DB_PATH = DATA_DIR / "history.db"

for _d in (DATA_DIR, LOG_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class Settings:
    """Reads .env if present; everything has a default for offline demos."""

    def __init__(self) -> None:
        self._load_dotenv(ROOT / ".env")

        self.app_name: str = "TruthLens"
        self.version: str = "0.1.0"

        # External providers
        self.newsapi_key: str = os.getenv("NEWSAPI_KEY", "").strip()
        self.user_agent: str = os.getenv(
            "USER_AGENT", "TruthLens/0.1 (misinformation research)"
        )
        self.request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "6"))

        # Signal weights (must sum ~1.0)
        self.evidence_weight: float = float(os.getenv("EVIDENCE_WEIGHT", "0.5"))
        self.nlp_weight: float = float(os.getenv("NLP_WEIGHT", "0.3"))
        self.source_weight: float = float(os.getenv("SOURCE_WEIGHT", "0.2"))

        # Decision thresholds
        self.support_threshold: float = float(os.getenv("SUPPORT_THRESHOLD", "0.20"))
        self.conflict_threshold: float = float(os.getenv("CONFLICT_THRESHOLD", "-0.20"))
        self.uncertain_band: float = float(os.getenv("UNCERTAIN_BAND", "0.12"))

        # Evidence engine
        self.max_claims_default: int = 5
        self.evidence_per_claim: int = int(os.getenv("EVIDENCE_PER_CLAIM", "6"))
        self.min_evidence_for_verdict: int = int(os.getenv("MIN_EVIDENCE_FOR_VERDICT", "2"))

        # Paths
        self.data_dir = DATA_DIR
        self.log_dir = LOG_DIR
        self.models_dir = MODELS_DIR
        self.db_path = DB_PATH
        self.audit_log = LOG_DIR / "audit.jsonl"

    @staticmethod
    def _load_dotenv(path: Path) -> None:
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
