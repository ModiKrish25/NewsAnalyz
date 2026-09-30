"""Step 2 of the Evidence Engine: Evidence Retrieval.

Multi-provider retrieval with graceful degradation:
  - Wikipedia (always available, offline-safe fallback stub included)
  - NewsAPI (if NEWSAPI_KEY is set)
  - DuckDuckGo HTML (no key needed, best-effort)

Every provider returns EvidenceItem objects with stance left UNRELATED;
classification happens in the next stage.
"""
from __future__ import annotations

import logging
import re
from typing import Protocol

import requests

from app.config import settings
from app.schemas import EvidenceItem

logger = logging.getLogger(__name__)

_TIMEOUT = settings.request_timeout


class EvidenceProvider(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[EvidenceItem]: ...


# ── Wikipedia ────────────────────────────────────────────────────────────
_WIKI_API = "https://en.wikipedia.org/w/api.php"


class WikipediaProvider:
    name = "wikipedia"

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()

    def search(self, query: str, k: int) -> list[EvidenceItem]:
        try:
            resp = self.session.get(
                _WIKI_API,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": min(k, 5),
                    "format": "json",
                },
                timeout=_TIMEOUT,
                headers={"User-Agent": settings.user_agent},
            )
            resp.raise_for_status()
            hits = resp.json().get("query", {}).get("search", [])
        except Exception as exc:  # noqa: BLE001 — retrieval must never crash the pipeline
            logger.warning("wikipedia search failed: %s", exc)
            return []

        items: list[EvidenceItem] = []
        for hit in hits:
            snippet = re.sub(r"<[^>]+>", "", hit.get("snippet", ""))
            items.append(
                EvidenceItem(
                    source_name="Wikipedia",
                    url=f"https://en.wikipedia.org/wiki/{hit['title'].replace(' ', '_')}",
                    title=hit.get("title", ""),
                    snippet=snippet,
                    provider=self.name,
                )
            )
        return items


# ── NewsAPI ──────────────────────────────────────────────────────────────
_NEWSAPI_URL = "https://newsapi.org/v2/everything"


class NewsAPIProvider:
    name = "newsapi"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.newsapi_key

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, k: int) -> list[EvidenceItem]:
        if not self.available:
            return []
        try:
            resp = requests.get(
                _NEWSAPI_URL,
                params={
                    "q": query,
                    "pageSize": min(k, 6),
                    "language": "en",
                    "sortBy": "relevancy",
                },
                headers={"X-Api-Key": self.api_key},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            arts = resp.json().get("articles", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("newsapi search failed: %s", exc)
            return []

        return [
            EvidenceItem(
                source_name=(a.get("source") or {}).get("name", "News"),
                url=a.get("url", ""),
                title=a.get("title", "") or "",
                snippet=(a.get("description", "") or "")[:400],
                published=(a.get("publishedAt", "") or "")[:10] or None,
                provider=self.name,
            )
            for a in arts
            if a.get("title")
        ]


# ── DuckDuckGo (keyless, best-effort) ────────────────────────────────────
_DDG_URL = "https://html.duckduckgo.com/html/"

_AUTHORITY_BONUS = {
    "reuters.com": 0.95, "apnews.com": 0.95, "bbc.com": 0.9, "bbc.co.uk": 0.9,
    "nytimes.com": 0.9, "theguardian.com": 0.85, "washingtonpost.com": 0.85,
    "who.int": 0.97, "un.org": 0.9, "nature.com": 0.95, "science.org": 0.95,
    "cdc.gov": 0.95, "nih.gov": 0.95, "nasa.gov": 0.95, ".gov": 0.9, ".edu": 0.85,
}


class DuckDuckGoProvider:
    name = "duckduckgo"

    def search(self, query: str, k: int) -> list[EvidenceItem]:
        try:
            from bs4 import BeautifulSoup

            resp = requests.post(
                _DDG_URL,
                data={"q": query, "b": ""},
                headers={"User-Agent": settings.user_agent},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as exc:  # noqa: BLE001
            logger.warning("duckduckgo search failed: %s", exc)
            return []

        items: list[EvidenceItem] = []
        for res in soup.select("div.result")[:k]:
            a = res.select_one("a.result__a")
            snippet_el = res.select_one(".result__snippet")
            if not a:
                continue
            url = a.get("href", "")
            # DDG wraps URLs like //duckduckgo.com/l/?uddg=<encoded>&rut=...
            m = re.search(r"uddg=([^&]+)", url)
            if m:
                from urllib.parse import unquote

                url = unquote(m.group(1))
            domain = _domain_of(url)
            credibility = 0.5
            for pat, val in _AUTHORITY_BONUS.items():
                if pat in domain:
                    credibility = val
                    break
            items.append(
                EvidenceItem(
                    source_name=domain or "Web",
                    url=url,
                    title=a.get_text(" ", strip=True),
                    snippet=snippet_el.get_text(" ", strip=True) if snippet_el else "",
                    provider=self.name,
                )
            )
        return items


def _domain_of(url: str) -> str:
    m = re.search(r"https?://([^/]+)", url or "")
    return (m.group(1) if m else "").replace("www.", "")


# ── Offline fallback corpus (zero-network demo mode) ─────────────────────
class OfflineCorpusProvider:
    """Tiny built-in corpus so the Evidence Engine demo works with no network.

    Matches on keyword overlap; good enough to demonstrate the full
    retrieval → classification → assessment flow offline.
    """

    name = "offline_corpus"

    _CORPUS: list[dict] = [
        {
            "title": "WHO announces ABC vaccination framework",
            "text": "The World Health Organization announced ABC, a new vaccination framework "
                    "aimed at improving coverage in developing regions, at its annual assembly.",
            "url": "https://example.org/who-abc",
            "source": "WHO Newsroom",
        },
        {
            "title": "Fact check: XYZ organization and the ABC announcement",
            "text": "Reuters fact-checkers found that the viral claim that XYZ organization "
                    "announced ABC distorts a 2019 press release; the organization denied "
                    "making any such announcement.",
            "url": "https://example.org/factcheck-xyz",
            "source": "Reuters Fact Check",
        },
        {
            "title": "Global vaccine coverage statistics",
            "text": "Global vaccination coverage rose from 72 percent to 86 percent between 2000 "
                    "and 2019 according to WHO and UNICEF estimates.",
            "url": "https://example.org/vaccine-stats",
            "source": "UNICEF Data",
        },
        {
            "title": "NASA launches new earth observation satellite",
            "text": "NASA launched a new earth observation satellite on a Falcon 9 rocket to "
                    "monitor climate indicators including sea level rise and ice sheet loss.",
            "url": "https://example.org/nasa-sat",
            "source": "NASA News",
        },
        {
            "title": "Central bank raises interest rates amid inflation",
            "text": "The central bank raised its benchmark interest rate by 25 basis points "
                    "citing persistent inflation above its 2 percent target.",
            "url": "https://example.org/rates",
            "source": "Financial Times",
        },
        {
            "title": "Study finds no link between XYZ and ABC",
            "text": "A peer-reviewed study in a major medical journal found no statistically "
                    "significant link between XYZ and ABC, contradicting widely shared posts.",
            "url": "https://example.org/study-no-link",
            "source": "Nature Medicine",
        },
        {
            "title": "City council approves new metro line",
            "text": "The city council voted 9-3 to approve the new metro line, with construction "
                    "scheduled to begin next year and finish in 2031.",
            "url": "https://example.org/metro",
            "source": "Local Gazette",
        },
        {
            "title": "Election results certified after recount",
            "text": "Officials certified the election results after a recount confirmed the "
                    "margin of victory at 1,200 votes, rejecting fraud allegations.",
            "url": "https://example.org/election",
            "source": "AP News",
        },
    ]

    def search(self, query: str, k: int) -> list[EvidenceItem]:
        q_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored: list[tuple[float, dict]] = []
        for doc in self._CORPUS:
            text = (doc["title"] + " " + doc["text"]).lower()
            terms = set(re.findall(r"[a-z0-9]+", text))
            overlap = len(q_terms & terms) / max(len(q_terms), 1)
            if overlap > 0.15:
                scored.append((overlap, doc))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            EvidenceItem(
                source_name=d["source"],
                url=d["url"],
                title=d["title"],
                snippet=d["text"],
                provider=self.name,
            )
            for _, d in scored[:k]
        ]


# ── Facade ───────────────────────────────────────────────────────────────
class EvidenceRetriever:
    """Fan-out to all available providers, merge, and de-duplicate."""

    def __init__(self, offline_fallback: bool = True) -> None:
        self.providers: list = [NewsAPIProvider(), WikipediaProvider(), DuckDuckGoProvider()]
        self.offline_fallback = OfflineCorpusProvider() if offline_fallback else None
        self._used_network = False

    def retrieve(self, query: str, k: int = 6) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for provider in self.providers:
            try:
                got = provider.search(query, k)
            except Exception as exc:  # noqa: BLE001
                logger.warning("provider %s crashed: %s", provider.name, exc)
                got = []
            if got:
                self._used_network = True
            items.extend(got)
            if len(items) >= k:
                break

        if not items and self.offline_fallback:
            items = self.offline_fallback.search(query, k)

        return self._dedupe(items)[:k]

    @staticmethod
    def _dedupe(items: list[EvidenceItem]) -> list[EvidenceItem]:
        seen: set[str] = set()
        out: list[EvidenceItem] = []
        for it in items:
            key = (it.url or it.title.lower())[:200]
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out
