"""Test isolation: ignore any local .streamlit/secrets.toml.

config.py reads Streamlit secrets before environment variables, so a developer's secrets.toml (e.g. FREE_TIER=true or a
real API_KEY) would override the env values the tests set. Replace the global st.secrets with an empty Secrets object for
the whole test run. This covers direct config calls and AppTest runs (AppTest only swaps st.secrets when a test passes
its own secrets). Tests that patch "streamlit.secrets" still take precedence.

Runs because the test commands import modules as the `tests` package:
    python -m unittest tests.test_deploy tests.test_limits ...      (README, CI)
    python -m unittest discover -s tests -t .                        (discover form needs -t .)
"""
import streamlit as st
from streamlit.runtime.secrets import Secrets

_empty = Secrets()
_empty._secrets = {}  # already "parsed": never reads secrets.toml
st.secrets = _empty
