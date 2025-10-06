import os, json
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="Darwinacci Metrics", layout="wide")

mpath = os.getenv('DARWINACCI_METRICS_PATH', 'data/metrics.jsonl')

st.title("Darwinacci-Ω Live Metrics")
st.caption(f"Metrics file: {mpath}")

@st.cache_data(ttl=5.0)
def load_metrics(path: str):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows

rows = load_metrics(mpath)
if not rows:
    st.warning("No metrics yet.")
else:
    st.success(f"Loaded {len(rows)} rows")
    st.line_chart({
        'best_score': [r.get('best_score', 0.0) for r in rows],
        'coverage': [r.get('coverage', 0.0) for r in rows],
        'novelty_archive_size': [r.get('novelty_archive_size', 0) for r in rows],
    })

    st.subheader("Latest Record")
    last = rows[-1]
    st.json(last)
