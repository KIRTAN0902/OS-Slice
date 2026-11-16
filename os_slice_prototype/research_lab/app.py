"""
OS-Slice Research Lab Entrypoint
"""

import sys
from pathlib import Path
import streamlit as st

_PROTO_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PROTO_ROOT.parent
for _p in (str(_PROTO_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

st.set_page_config(
    page_title="OS-Slice Research Lab & Model Arena",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from research_lab.navigation import render_sidebar_and_route
from research_lab.styles.theme import inject_theme


def main():
    inject_theme()
    render_sidebar_and_route()


if __name__ == "__main__":
    main()
