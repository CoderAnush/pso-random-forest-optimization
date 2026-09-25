"""Demo entry point: page setup and navigation (``streamlit run app.py``)."""

from __future__ import annotations

import streamlit as st

from pso_rf.app import components, explain, live, replay, results

PAGES = {
    "🔁 Live closed-loop lab": live.render,
    "📊 Experiment results": results.render,
    "▶️ Swarm replay": replay.render,
    "📘 How it works": explain.render,
}


def main() -> None:
    """Render the selected page."""
    st.set_page_config(page_title="PSO × Random Forest", page_icon="🔁", layout="wide")
    st.markdown(components.CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### PSO × Random Forest")
        st.caption("Closed-loop hyperparameter optimization · Iris · Digits · Heart Disease")
        page = st.radio("View", list(PAGES), label_visibility="collapsed")
        st.divider()
    PAGES[page]()
