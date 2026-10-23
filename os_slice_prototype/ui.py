"""
OS-Slice Model Catalog & Slicing Projection UI
Renders interactive tiered catalog of foundational models (360M to 120B)
with hardware sizing, cache detection, and theoretical slicing projections.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import altair as alt

# Ensure catalog import works
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from catalog.manager import SUPPORTED_MODELS, ModelCatalogManager


def render_catalog_ui():
    """Renders the comprehensive Tiered Model Catalog view."""
    st.header("📚 Foundational Model Catalog (SLMs to 120B)")
    st.markdown(
        "Explore foundational open-weights models evaluated for **Operation-Specific Behavioral Slicing (OS-Slice)**. "
        "From ultra-compact sub-1B edge devices to 120B parameter frontier giants, OS-Slice excises "
        "redundant generalist weights to isolate deterministic operational execution circuits."
    )

    # Top Filter Bar
    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        tier_options = ["All Tiers"] + ModelCatalogManager.get_all_tier_categories()
        selected_tier = st.selectbox("Filter by Architectural Tier:", tier_options)
    with f_col2:
        cache_filter = st.selectbox(
            "Filter by Local Availability:",
            ["All Models", "Cached Locally Only", "Remote (HF Hub)"]
        )

    # Filter Models
    models = SUPPORTED_MODELS
    if selected_tier != "All Tiers":
        models = [m for m in models if m.get("tier_category") == selected_tier]

    if cache_filter == "Cached Locally Only":
        models = [m for m in models if ModelCatalogManager.is_model_cached(m["id"])]
    elif cache_filter == "Remote (HF Hub)":
        models = [m for m in models if not ModelCatalogManager.is_model_cached(m["id"])]

    # Display Tiers
    tiers = ModelCatalogManager.get_all_tier_categories()
    if selected_tier != "All Tiers":
        tiers = [selected_tier]

    for tier_cat in tiers:
        tier_models = [m for m in models if m.get("tier_category") == tier_cat]
        if not tier_models:
            continue

        st.markdown(f"### {tier_cat}")
        
        # Display cards in grid
        cols = st.columns(2)
        for i, m in enumerate(tier_models):
            with cols[i % 2]:
                is_cached = ModelCatalogManager.is_model_cached(m["id"])
                cache_badge = "🟢 Cached Locally" if is_cached else "⚪ Remote (HF Hub)"
                
                with st.container(border=True):
                    st.markdown(f"#### {m['name']} `{m['params']}`")
                    st.caption(f"**HuggingFace ID:** `{m['id']}` | **Status:** {cache_badge}")

                    m_row1, m_row2 = st.columns(2)
                    m_row1.markdown(f"**Disk Footprint:** {m['disk_size']}")
                    m_row2.markdown(f"**Min VRAM:** {m['min_vram']}")

                    st.info(f"**OS-Slice Role:** {m['recommended_for']}")

    st.markdown("---")

    # Interactive Slicing & Hardware Sizing Projection
    st.subheader("⚡ Theoretical Slicing & VRAM Reduction Calculator")
    st.markdown(
        "Simulate operational parameter extraction and memory bandwidth reduction across all 11 catalog architectures."
    )

    calc_col1, calc_col2 = st.columns([2, 1])
    with calc_col1:
        model_names = [f"{m['name']} ({m['params']})" for m in SUPPORTED_MODELS]
        selected_model_str = st.selectbox("Select Target Model Architecture:", model_names, index=1)
        selected_model = SUPPORTED_MODELS[model_names.index(selected_model_str)]

    with calc_col2:
        sim_slicing_ratio = st.slider(
            "Projected Slicing Ratio:",
            min_value=0.10,
            max_value=0.70,
            value=0.30,
            step=0.05,
            format="%.0f%%"
        )

    # Compute Projections
    proj = ModelCatalogManager.calculate_slicing_projections(selected_model["id"], sim_slicing_ratio)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Base Parameters", f"{proj['total_params'] / 1e9:.2f} B" if proj['total_params'] >= 1e9 else f"{proj['total_params'] / 1e6:.0f} M", "100%")
    m2.metric("Active Sliced Parameters", f"{proj['active_params'] / 1e9:.2f} B" if proj['active_params'] >= 1e9 else f"{proj['active_params'] / 1e6:.0f} M", f"{100 - sim_slicing_ratio * 100:.0f}% Active")
    m3.metric("Excised Generalist Weights", f"{proj['excised_params'] / 1e9:.2f} B" if proj['excised_params'] >= 1e9 else f"{proj['excised_params'] / 1e6:.0f} M", f"-{sim_slicing_ratio * 100:.0f}% Sliced")
    m4.metric("VRAM Savings (FP16)", f"{proj['saved_fp16_gb']} GB", f"-{proj['saved_fp16_gb'] / (proj['orig_fp16_gb'] + 1e-6) * 100:.0f}% Memory")

    # Bar Chart of All Models Sliced
    st.markdown("#### Cross-Tier Memory Footprint: Base vs. OS-Slice (at current slicing ratio)")
    chart_data = []
    for m in SUPPORTED_MODELS:
        p = ModelCatalogManager.calculate_slicing_projections(m["id"], sim_slicing_ratio)
        chart_data.append({
            "Model": m["name"],
            "Configuration": "Base Generalist",
            "VRAM (GB)": p["orig_fp16_gb"],
            "Tier": m.get("tier_category", "Other")
        })
        chart_data.append({
            "Model": m["name"],
            "Configuration": f"OS-Slice (-{sim_slicing_ratio * 100:.0f}%)",
            "VRAM (GB)": p["sliced_fp16_gb"],
            "Tier": m.get("tier_category", "Other")
        })

    df_chart = pd.DataFrame(chart_data)
    chart = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X("Model:N", sort=None, title="Model Architecture"),
        y=alt.Y("VRAM (GB):Q", title="Memory Footprint in FP16 (GB)"),
        color=alt.Color("Configuration:N", scale=alt.Scale(range=["#64748b", "#10b981"])),
        xOffset="Configuration:N",
        tooltip=["Model", "Configuration", "VRAM (GB)", "Tier"]
    ).properties(height=380)

    st.altair_chart(chart, use_container_width=True)


def main():
    st.set_page_config(
        page_title="OS-Slice Model Catalog",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    render_catalog_ui()


if __name__ == "__main__":
    main()
