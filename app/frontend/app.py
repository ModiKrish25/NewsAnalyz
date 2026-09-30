"""TruthLens web app (Streamlit).

Paste an article (or URL) → see the full evidence trail:
claims, retrieved evidence with stances, signal gauges, and the final verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(page_title="TruthLens", page_icon="🔎", layout="wide")


def api_ok() -> bool:
    try:
        return requests.get(f"{API}/health", timeout=2).ok
    except Exception:
        return False


@st.cache_data(ttl=60)
def health() -> dict:
    try:
        return requests.get(f"{API}/health", timeout=3).json()
    except Exception:
        return {}


def call_analyze(payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API}/analyze", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["report"]
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def gauge(label: str, value: float, color: str) -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(value * 100),
            number={"suffix": ""},
            title={"text": label},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#fee2e2"},
                    {"range": [40, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#dcfce7"},
                ],
            },
        )
    )
    fig.update_layout(height=190, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


VERDICT_STYLE = {
    "SUPPORTS": ("🟢", "Evidence supports the content", "normal"),
    "CONFLICTS": ("🔴", "Evidence contradicts the content", "normal"),
    "UNKNOWN": ("🟡", "Inconclusive — evidence is insufficient or mixed", "normal"),
}

st.title("🔎 TruthLens")
st.caption(
    "Evidence-first misinformation analysis: every claim is checked against "
    "retrieved sources before a verdict is reached."
)

with st.sidebar:
    st.header("Settings")
    max_claims = st.slider("Max claims to check", 1, 10, 4)
    st.divider()
    h = health()
    st.subheader("Engine status")
    st.write("spaCy:", "✅" if h.get("spacy") else "⚠️ regex fallback")
    st.write("ML models:", "✅ trained" if h.get("models_trained") else "⚠️ heuristics only")
    st.write("NewsAPI:", "✅ enabled" if h.get("newsapi") else "— not configured")
    st.divider()
    st.caption("Run `uvicorn app.main:app` to start the API.")

tab1, tab2 = st.tabs(["Analyze", "How it works"])

with tab1:
    col_text, col_meta = st.columns([3, 1])

    with col_text:
        url = st.text_input("Article URL (optional)", placeholder="https://…")
        text = st.text_area(
            "…or paste the article text",
            height=220,
            placeholder="Paste the article you want to fact-check…",
        )
        uploaded = st.file_uploader("…or upload a .txt file", type=["txt", "md"])
        if uploaded and not text:
            text = uploaded.read().decode("utf-8", errors="ignore")

    with col_meta:
        title = st.text_input("Title")
        author = st.text_input("Author")
        published = st.text_input("Published (date)")

    if st.button("🔍 Analyze", type="primary", disabled=not (text.strip() or url.strip())):
        if not api_ok():
            st.error("Backend not reachable. Start it with: `uvicorn app.main:app --reload`")
        else:
            with st.spinner("Extracting claims → retrieving evidence → fusing signals…"):
                payload = {
                    "text": text,
                    "url": url,
                    "title": title,
                    "author": author,
                    "published": published,
                    "max_claims": max_claims,
                }
                report = call_analyze(payload)

            if report:
                icon, blurb, _ = VERDICT_STYLE[report["verdict"]]
                st.success(f"{icon} **{report['verdict']}** — {blurb}", icon=icon)
                st.caption(
                    f"Confidence {report['confidence']:.0%} · "
                    f"{report['processing_ms']} ms · id `{report['request_id']}`"
                )

                # ── Signal gauges ────────────────────────────────
                g1, g2, g3 = st.columns(3)
                with g1:
                    gauge("Content (NLP)", report["content_score"], "#6366f1")
                with g2:
                    gauge("Source", report["source_score"], "#0ea5e9")
                with g3:
                    gauge("Evidence", report["evidence_score"], "#10b981")

                # ── Decision breakdown ───────────────────────────
                with st.expander("⚙️ Decision engine breakdown", expanded=True):
                    for c in report["decision"]["contributions"]:
                        st.markdown(
                            f"**{c['signal']}** — score `{c['score']:+.2f}` × "
                            f"weight `{c['weight']}` × conf `{c['confidence']:.2f}` "
                            f"= **{c['weighted']:+.3f}**"
                        )
                        for r in c["reasons"]:
                            st.markdown(f"  - {r}")
                    st.markdown(f"**Fused score:** `{report['decision']['fused_score']:+.3f}`")
                    for rule in report["decision"]["rule_trail"]:
                        st.markdown(f"- {rule}")

                # ── Evidence Engine: the centerpiece ─────────────
                st.subheader("🧾 Evidence Engine")
                if not report["claims"]:
                    st.info("No checkable claims were extracted from this text.")
                for claim in report["claims"]:
                    stance_icon = {
                        "SUPPORTS": "🟢",
                        "CONTRADICTS": "🔴",
                        "UNRELATED": "⚪",
                    }[claim["stance"]]
                    head = st.container()
                    with head:
                        st.markdown(
                            f"**Claim {claim['claim_id']}** {stance_icon} "
                            f"`{claim['stance']}` · evidence score `{claim['evidence_score']:+.2f}` · "
                            f"conf `{claim['confidence']:.0%}` — *{claim['rationale']}*"
                        )
                        st.markdown(f"> {claim['text']}")
                        if claim["entities"]:
                            st.caption("Entities: " + ", ".join(claim["entities"]))
                        for ev in claim["evidence"]:
                            ev_icon = {
                                "SUPPORTS": "🟢",
                                "CONTRADICTS": "🔴",
                                "UNRELATED": "⚪",
                            }[ev["stance"]]
                            st.markdown(
                                f"{ev_icon} **[{ev['source_name']}]({ev['url']})** — {ev['stance']} "
                                f"(`conf {ev['stance_confidence']:.2f}`, `rel {ev['relevance']:.2f}`)\n\n"
                                f"> {ev['title']} — {ev['snippet'][:220]}\n\n"
                                f"<small>{ev['stance_rationale']}</small>",
                                unsafe_allow_html=True,
                            )
                            st.divider()

                # ── Report extras ────────────────────────────────
                c1, c2 = st.columns(2)
                with c1:
                    with st.expander("⚠️ Warning indicators"):
                        if report["warning_indicators"]:
                            for w in report["warning_indicators"]:
                                st.markdown(f"- {w}")
                        else:
                            st.write("None.")
                    with st.expander("📝 Reasons"):
                        for r in report["reasons"]:
                            st.markdown(f"- {r}")
                with c2:
                    with st.expander("🔤 Important words"):
                        st.write(", ".join(f"`{w}`" for w in report["important_words"]) or "—")
                    with st.expander("🤖 ML model layer"):
                        ml = report["ml"]
                        if not ml["available"]:
                            st.write("Not trained — run `python scripts/train_baseline.py`.")
                        else:
                            st.write(f"Ensemble P(fake): **{ml['fake_probability']:.2f}**")
                            for m in ml["models"]:
                                st.markdown(f"- {m['model']}: `{m['fake_probability']:.3f}`")

with tab2:
    st.markdown(
        """
### The Evidence Engine (centerpiece)

```
Article/Video → Claims → Evidence Retrieval → Evidence Classification → Claim Assessment
```

Every extracted claim goes through:

1. **Claim extraction** — sentences with named entities, numbers, dates,
   reported speech; opinions and speculation are filtered out.
2. **Evidence retrieval** — Wikipedia, NewsAPI (optional), DuckDuckGo,
   with an offline fallback corpus.
3. **Evidence classification** — each item is labeled
   `SUPPORTS / CONTRADICTS / UNRELATED` with a human-readable rationale.
4. **Claim assessment** — weighted aggregation → claim-level evidence score.

### Final scoring — never one model

```
CONTENT → NLP score ─┐
SOURCE  → Source ────┼→ Decision Engine → SUPPORTS / CONFLICTS / UNKNOWN
EVIDENCE→ Evidence ──┘
```

The evidence signal carries the largest weight (configurable via `.env`),
and every rule that fired is shown in the report.
"""
    )
