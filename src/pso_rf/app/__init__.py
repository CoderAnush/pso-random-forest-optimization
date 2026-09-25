"""Interactive demo (Streamlit + Plotly) for the final review (ADR-026).

This optional layer sits on top of ``pso_rf.experiments``: live runs go through the same ``run_fold`` code path
as the experiment (so the demo cannot drift from what was measured), and the results pages only read saved
files. Launch with ``python -m pso_rf demo`` or ``streamlit run app.py``.
"""
