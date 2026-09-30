"""SOURCE signal: domain credibility analysis."""
from __future__ import annotations

import re
from urllib.parse import urlparse

from app.schemas import SourceInfo, SourceSignal

# Illustrative registry — extend or replace with a maintained list/dataset.
_KNOWN_OUTLETS: dict[str, tuple[str, float]] = {
    "reuters.com": ("Reuters", 0.97),
    "apnews.com": ("Associated Press", 0.97),
    "bbc.com": ("BBC", 0.94),
    "bbc.co.uk": ("BBC", 0.94),
    "nytimes.com": ("The New York Times", 0.92),
    "theguardian.com": ("The Guardian", 0.9),
    "washingtonpost.com": ("The Washington Post", 0.9),
    "wsj.com": ("The Wall Street Journal", 0.92),
    "ft.com": ("Financial Times", 0.92),
    "npr.org": ("NPR", 0.92),
    "aljazeera.com": ("Al Jazeera", 0.85),
    "cnn.com": ("CNN", 0.82),
    "foxnews.com": ("Fox News", 0.75),
    "msnbc.com": ("MSNBC", 0.75),
    "bloomberg.com": ("Bloomberg", 0.93),
    "economist.com": ("The Economist", 0.93),
    "nature.com": ("Nature", 0.98),
    "science.org": ("Science", 0.98),
    "who.int": ("World Health Organization", 0.97),
    "un.org": ("United Nations", 0.93),
    "cdc.gov": ("CDC", 0.96),
    "nih.gov": ("NIH", 0.96),
    "nasa.gov": ("NASA", 0.97),
    "europa.eu": ("European Union", 0.9),
}

_GOVDOMAIN = re.compile(r"\.(gov|gov\.[a-z]{2}|edu|ac\.[a-z]{2}|int)$")
_SATIRE_HINTS = ("onion", "babylonbee", "waterfordwhispers", "duffelblog")
_LOW_HINTS = ("blogspot", "wordpress.com", "medium.com", "tumblr", "vk.com", "telegram")


def domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    return netloc[4:] if netloc.startswith("www.") else netloc


def analyze_source(url: str) -> SourceSignal:
    """Score the credibility of the article's source domain (0..1)."""
    domain = domain_of(url or "")
    info = SourceInfo(url=url, domain=domain)
    reasons: list[str] = []

    if not domain:
        info.credibility = 0.5
        reasons.append("No source URL given; source signal is neutral/weak.")
        return SourceSignal(score=0.5, confidence=0.2, source=info, reasons=reasons)

    if domain in _KNOWN_OUTLETS:
        name, cred = _KNOWN_OUTLETS[domain]
        info.known_outlet = name
        info.credibility = cred
        reasons.append(f"Recognized outlet: {name} (baseline credibility {cred:.2f}).")
    elif any(h in domain for h in _SATIRE_HINTS):
        info.credibility = 0.05
        reasons.append("Domain matches known satire sites.")
    elif _GOVDOMAIN.search(domain):
        info.credibility = 0.9
        reasons.append("Academic/government domain (.gov/.edu/.int).")
    elif any(h in domain for h in _LOW_HINTS):
        info.credibility = 0.35
        reasons.append("User-generated content platform; lower default trust.")
    else:
        info.credibility = 0.5
        reasons.append("Unknown domain; neutral default credibility.")

    # Long hyphenated domains are a mild spam signal
    if domain.count("-") >= 3:
        info.credibility -= 0.1
        reasons.append("Hyphen-heavy domain (common in spam sites).")
    if len(domain) > 30:
        info.credibility -= 0.05
        reasons.append("Unusually long domain name.")

    info.credibility = max(0.0, min(1.0, info.credibility))
    info.reasons = reasons
    return SourceSignal(
        score=round(info.credibility, 3),
        confidence=0.8 if info.known_outlet else 0.5,
        source=info,
        reasons=reasons,
    )
