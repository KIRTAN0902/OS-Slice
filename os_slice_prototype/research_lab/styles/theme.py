"""
OS-Slice Research Lab Theme & UI Helpers
Compatible with Streamlit 1.22+ and newer versions.
"""

import streamlit as st


def safe_rerun():
    """Reruns the Streamlit script safely across Streamlit 1.22+ and 1.30+."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def inject_theme():
    """Injects clean scientific dark-mode styling and badge classes."""
    st.markdown(
        """
        <style>
        .os-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .os-card-highlight {
            background: #111927;
            border: 1px solid #1f6feb;
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .os-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .badge-green { background: #064e3b; color: #34d399; border: 1px solid #059669; }
        .badge-amber { background: #451a03; color: #fbbf24; border: 1px solid #d97706; }
        .badge-red   { background: #450a0a; color: #f87171; border: 1px solid #dc2626; }
        .badge-blue  { background: #0c2d6b; color: #60a5fa; border: 1px solid #2563eb; }
        .badge-gray  { background: #21262d; color: #9ca3af; border: 1px solid #4b5563; }
        .jury-banner {
            background: linear-gradient(90deg, #0f2942 0%, #162032 100%);
            border: 1px solid #38bdf8;
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 14px;
        }
        .mono-box {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 10px 12px;
            font-family: monospace;
            font-size: 0.84rem;
            white-space: pre-wrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge_html(text: str, variant: str = "gray") -> str:
    return f'<span class="os-badge badge-{variant}">{text}</span>'


def state_badge(state: str) -> str:
    s = state.lower()
    if "validated" in s or "completed" in s or "success" in s or "implemented" in s:
        return badge_html(state, "green")
    if "ready" in s or "running" in s or "historical" in s:
        return badge_html(state, "blue")
    if "cloud" in s or "warning" in s:
        return badge_html(state, "amber")
    if "fail" in s or "error" in s or "invalid" in s:
        return badge_html(state, "red")
    return badge_html(state, "gray")
