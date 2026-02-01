# QuestionGenV2.py
# Streamlit all-in-one (incremental + formatter retries on validation errors):
# Topics CSV -> protos -> select k -> (optional canonicalize selected protos)
# -> build full -> verify -> (NEW: ensure resolution URL present) -> format single
# -> append to draft CSV incrementally
# -> optional rebalance -> (optional canonicalize post-rebalance) -> optional final clean -> export CSV
#
# Key additions in this version:
# 1) Multi-level web enrichment where Call 2 builds on Call 1, and Call 3 builds on Call 2.
# 2) URL accessibility filtering (drops 404/410/401/403/5xx; normalizes redirects) to avoid dead/unreachable pages.
# 3) Pre-format check that resolution_criteria contains at least one URL other than Metaculus credible sources policy;
#    if missing, inject a best accessible URL from web results into resolution_criteria AND description (minimal edit).
# 4) Stronger date-discipline instructions to reduce the “today/tomorrow” window bug (e.g., “between Jan 22 and Jan 23”).
# 5) Rebalance target updated to 60% binary / 20% numeric / 20% MCQ (per request).
#
# Install:
#   pip install streamlit requests pydantic
# Run:
#   streamlit run QuestionGenV2.py

from forecasting_app.ui_runner import run_app

run_app()
