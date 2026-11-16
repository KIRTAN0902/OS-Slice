"""
OS-Slice Research Lab Navigation & Page Router
"""

import streamlit as st

from research_lab.components.common import render_jury_banner
from research_lab.pages.overview_and_pipeline import (
    render_model_specifications,
    render_overview,
    render_transformation_pipeline,
)
from research_lab.pages.observability_pages import (
    render_behavioral_profiling,
    render_structural_slicing,
)
from research_lab.pages.arena_and_benchmarks import (
    render_benchmark_library,
    render_model_arena,
    render_performance_dashboard,
    render_task_evaluation,
)
from research_lab.pages.experiments_and_evidence import (
    render_ablation_study,
    render_compression_analysis,
    render_live_demo,
    render_research_evidence,
)
from research_lab.styles.theme import safe_rerun


ALL_PAGES = [
    "01 Overview",
    "02 Transformation Pipeline",
    "03 Model Specifications",
    "04 Behavioral Profiling",
    "05 Structural Slicing",
    "06 Model Arena",
    "07 Benchmark Library",
    "08 Performance Dashboard",
    "09 Task Evaluation",
    "10 Ablation Study",
    "11 Compression Analysis",
    "12 Live Demo",
    "13 Research Evidence",
]


def navigate_to(page_name: str):
    if page_name in ALL_PAGES:
        st.session_state["nav_page"] = page_name
        safe_rerun()


def render_sidebar_and_route():
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "01 Overview"

    with st.sidebar:
        st.markdown("## OS-Slice Research Lab")
        st.caption("Operation-Specific Behavioral Slicing")

        jury_on = st.checkbox("Jury Guided Walkthrough", value=st.session_state.get("jury_mode", False))
        st.session_state["jury_mode"] = jury_on

        st.markdown("---")
        current = st.session_state["nav_page"]
        idx = ALL_PAGES.index(current) if current in ALL_PAGES else 0
        selected = st.radio("Research Lab Sections", ALL_PAGES, index=idx)
        st.session_state["nav_page"] = selected

        st.markdown("---")
        st.caption("Shared Python/PyTorch Engine | Single Manifest & Artifact Registry")

    render_jury_banner(st.session_state["nav_page"], navigate_to)

    page = st.session_state["nav_page"]
    if page == "01 Overview":
        render_overview(navigate_to)
    elif page == "02 Transformation Pipeline":
        render_transformation_pipeline(navigate_to)
    elif page == "03 Model Specifications":
        render_model_specifications()
    elif page == "04 Behavioral Profiling":
        render_behavioral_profiling()
    elif page == "05 Structural Slicing":
        render_structural_slicing()
    elif page == "06 Model Arena":
        render_model_arena()
    elif page == "07 Benchmark Library":
        render_benchmark_library(navigate_to)
    elif page == "08 Performance Dashboard":
        render_performance_dashboard()
    elif page == "09 Task Evaluation":
        render_task_evaluation()
    elif page == "10 Ablation Study":
        render_ablation_study()
    elif page == "11 Compression Analysis":
        render_compression_analysis()
    elif page == "12 Live Demo":
        render_live_demo()
    elif page == "13 Research Evidence":
        render_research_evidence()
