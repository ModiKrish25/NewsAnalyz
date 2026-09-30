"""Admin dashboard: analytics over stored analyses + audit log."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(page_title="TruthLens Admin", page_icon="📊", layout="wide")
st.title("📊 TruthLens Admin")
st.caption("Analytics over every analysis handled by the backend.")

tab_dash, tab_log, tab_history = st.tabs(["Dashboard", "Audit log", "History"])


def _get(path: str, default):
    try:
        return requests.get(f"{API}{path}", timeout=5).json()
    except Exception:
        return default


with tab_dash:
    stats = _get("/stats", {"total": 0})
    if stats.get("total", 0) == 0:
        st.info("No analyses recorded yet. Run a few from the web app first.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total analyses", stats["total"])
        k2.metric("Avg confidence", f"{stats.get('avg_confidence', 0):.0%}")
        k3.metric("Avg latency", f"{stats.get('avg_processing_ms', 0)} ms")
        k4.metric(
            "Conflicted rate",
            f"{stats.get('CONFLICTS', 0) / max(stats['total'], 1):.0%}",
        )

        vdf = pd.DataFrame(
            {
                "verdict": ["SUPPORTS", "CONFLICTS", "UNKNOWN"],
                "count": [
                    stats.get("SUPPORTS", 0),
                    stats.get("CONFLICTS", 0),
                    stats.get("UNKNOWN", 0),
                ],
            }
        )
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.pie(vdf, names="verdict", values="count", hole=0.55, title="Verdict distribution"),
                use_container_width=True,
            )
        with c2:
            hist = _get("/history?limit=200", {"analyses": []}).get("analyses", [])
            if hist:
                hdf = pd.DataFrame(hist)
                hdf["created"] = pd.to_datetime(hdf["created_at"], unit="s")
                st.plotly_chart(
                    px.histogram(hdf, x="created", color="verdict", title="Analyses over time"),
                    use_container_width=True,
                )
            else:
                st.write("No history rows.")

with tab_log:
    import json
    from app.config import settings

    st.caption(f"Reading {settings.audit_log}")
    if settings.audit_log.exists():
        lines = settings.audit_log.read_text(encoding="utf-8").splitlines()[-100:]
        records = []
        for line in reversed(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if records:
            st.dataframe(pd.DataFrame(records), use_container_width=True)
        else:
            st.write("Audit log is empty.")
    else:
        st.write("No audit log yet.")

with tab_history:
    hist = _get("/history?limit=100", {"analyses": []}).get("analyses", [])
    if not hist:
        st.write("No analyses stored yet.")
    else:
        df = pd.DataFrame(hist)
        df["created"] = pd.to_datetime(df["created_at"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
        show = df[
            ["request_id", "created", "verdict", "confidence",
             "content_score", "source_score", "evidence_score", "n_claims", "processing_ms"]
        ]
        st.dataframe(show, use_container_width=True)

        with st.expander("Inspect a full report"):
            rid = st.selectbox("Request ID", df["request_id"].tolist())
            if st.button("Load report"):
                detail = _get(f"/history/{rid}", {"report": None}).get("report")
                if detail:
                    st.json(detail)
                else:
                    st.error("Report not found.")
