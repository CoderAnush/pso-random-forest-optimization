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


PERSISTENT_PREFIXES = ("live_", "replay_", "results_")


def _keep_settings() -> None:
    """Keep every page's settings when another page is shown.

    Streamlit drops the state of widgets that are not drawn in a run, so switching pages would reset the live
    lab's dataset, fold and swarm settings (and hide the finished run). Re-assigning the keys each run keeps them.
    """
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(PERSISTENT_PREFIXES) and key != "live_snapshot":
            st.session_state[key] = st.session_state[key]


def main() -> None:
    """Render the selected page."""
    st.set_page_config(page_title="PSO × Random Forest", page_icon="🔁", layout="wide")
    _keep_settings()
    st.markdown(components.CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### PSO × Random Forest")
        st.caption("Closed-loop hyperparameter optimization · Iris · Digits · Heart Disease")
        page = st.radio("View", list(PAGES), label_visibility="collapsed")
        st.divider()
    PAGES[page]()
