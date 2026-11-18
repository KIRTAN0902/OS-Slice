"""
Screens 04 (Behavioral Profiling) and 05 (Structural Slicing & Compensation)
Exposes internal token weights, operational losses, Fisher tensors, heatmaps,
masks, protected-layer verification, and compensation parameter deltas.
"""

import sys
from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st

_PROTO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTO_ROOT))

from research_lab.components.common import (
    get_artifact_manager,
    get_selected_manifest,
    load_experiment_json,
    load_experiment_tensor,
)
from research_lab.styles.theme import badge_html


def render_behavioral_profiling():
    st.title("04 · Behavioral Profiling (Stage 1 & Stage 2)")
    st.markdown(
        "Inspect calibration records, token-level operational boundary weights, "
        "per-sample operational losses, and empirical Fisher saliency tensors."
    )

    am = get_artifact_manager()
    manifest = get_selected_manifest(am.list_manifests())
    if manifest is None:
        st.info("Run or select an experiment to inspect Stage 1 calibration and Stage 2 Fisher tensors.")
        return

    calib_meta = load_experiment_json(manifest.experiment_id, "calibration/calibration_metadata.json")
    prof_summary = load_experiment_json(manifest.experiment_id, "profiles/profile_summary.json")
    prof_tensor = load_experiment_tensor(manifest.experiment_id, "profiles/fisher_profile.pt")

    st.subheader("Stage 1 · Boundary-Conditioned Operational Token Weighting")
    if calib_meta:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Calibration Samples", calib_meta.get("sample_count", 0))
        c2.metric("Boundary Weight", f"{calib_meta.get('boundary_token_weight', 6.0):.1f}x")
        c3.metric("Numeric Weight", f"{calib_meta.get('numeric_token_weight', 4.2):.1f}x")
        c4.metric("Payload Weight", f"{calib_meta.get('content_token_weight', 1.0):.1f}x")
        st.caption(f"Prompt labels masked as `{calib_meta.get('prompt_label_id', -100)}` with weight `{calib_meta.get('prompt_weight', 0.0)}`. Dataset SHA-256: `{calib_meta.get('dataset_hash', '')[:16]}...`")

        inspections = calib_meta.get("inspections", [])
        if inspections:
            domains = [f"[{i}] {rec['domain']}" for i, rec in enumerate(inspections)]
            sel_idx = st.selectbox("Inspect Calibration Record", range(len(domains)), format_func=lambda i: domains[i])
            rec = inspections[sel_idx]
            st.markdown("**Category Token Counts:** " + ", ".join(f"`{k}: {v}`" for k, v in rec.get("category_counts", {}).items()))
            tok_df = pd.DataFrame(rec.get("completion_tokens", []))
            if not tok_df.empty:
                st.dataframe(tok_df, use_container_width=True)
    else:
        st.warning("Calibration metadata artifact not found for this experiment.")

    st.markdown("---")
    st.subheader("Stage 2 · Empirical Fisher Saliency Profiling")
    if prof_summary and prof_tensor:
        meta = prof_summary.get("metadata", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Head Tensor Shape", str(prof_summary.get("head_saliency_shape")))
        c2.metric("MLP Tensor Shape", str(prof_summary.get("mlp_saliency_shape")))
        c3.metric("Mean Operational Loss", f"{meta.get('mean_operational_loss', 0.0):.4f}")
        c4.metric("Profile Source", prof_summary.get("profile_source", "unknown"))
        st.caption(f"Model/Task Cache Key: `{prof_summary.get('cache_key', 'N/A')}`")

        head_sal = prof_tensor["head_saliency"].float()  # [L, H]
        num_layers, num_heads = head_sal.shape
        heat_rows = []
        for l in range(num_layers):
            for h in range(num_heads):
                heat_rows.append({"Layer": l, "Head": h, "FisherSaliency": float(head_sal[l, h].item())})
        heat_df = pd.DataFrame(heat_rows)

        chart = (
            alt.Chart(heat_df)
            .mark_rect()
            .encode(
                x=alt.X("Head:O", title="Attention Head Index"),
                y=alt.Y("Layer:O", title="Transformer Layer"),
                color=alt.Color("FisherSaliency:Q", scale=alt.Scale(scheme="viridis"), title="Fisher Saliency"),
                tooltip=["Layer", "Head", "FisherSaliency"],
            )
            .properties(height=360, title="Per-Layer Attention Head Empirical Fisher Saliency")
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("**Sortable Attention Head Saliency Ranking**")
        st.dataframe(heat_df.sort_values("FisherSaliency", ascending=False), use_container_width=True)

        mlp_imp = meta.get("layer_mlp_importance", [])
        if mlp_imp:
            mlp_df = pd.DataFrame({"Layer": list(range(len(mlp_imp))), "MeanMLPSaliency": mlp_imp})
            st.bar_chart(mlp_df.set_index("Layer"))
    else:
        st.warning("Fisher profile artifacts not found for this experiment.")


def render_structural_slicing():
    st.title("05 · Structural Slicing & Compensation (Stage 3 & Stage 4)")
    st.markdown(
        "Inspect layer-aware structural masks, protected interface layers, reconciled parameter counts, "
        "and diagonal ridge-inspired compensation deltas."
    )

    am = get_artifact_manager()
    manifest = get_selected_manifest(am.list_manifests())
    if manifest is None:
        st.info("Run or select an experiment to inspect Stage 3 slicing masks and Stage 4 compensation.")
        return

    slicing_rep = load_experiment_json(manifest.experiment_id, "slicing/slicing_report.json")
    comp_rep = load_experiment_json(manifest.experiment_id, "compensation/compensation_report.json")

    st.subheader("Stage 3 · Layer-Aware Structural Slicing")
    if slicing_rep:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Tensor Params", f"{slicing_rep['actual_tensor_total_params']:,}")
        c2.metric("Active Structural Params", f"{slicing_rep['active_params_structural']:,}")
        c3.metric("Zeroed Tensor Entries", f"{slicing_rep['actual_tensor_zeroed_params']:,}")
        c4.metric("Active Reduction", f"-{slicing_rep['effective_active_sparsity_pct']:.2f}%")

        pv = slicing_rep.get("protected_layers_verified", False)
        badge = badge_html("Protected Layers Verified (0% Sliced)", "green") if pv else badge_html("Protected Layer Violation", "red")
        st.markdown(f"**Protected Interface Layers:** `{slicing_rep.get('protected_layers')}` &nbsp; {badge}", unsafe_allow_html=True)

        layer_df = pd.DataFrame(slicing_rep.get("layers", []))
        if not layer_df.empty:
            layer_df["head_retention_pct"] = (layer_df["retained_heads_count"] / layer_df["total_heads"].clip(lower=1)) * 100.0
            layer_df["mlp_retention_pct"] = (layer_df["retained_mlp_channels"] / layer_df["total_mlp_channels"].clip(lower=1)) * 100.0

            ret_chart = (
                alt.Chart(layer_df)
                .transform_fold(["head_retention_pct", "mlp_retention_pct"], as_=["Component", "RetentionPct"])
                .mark_bar()
                .encode(
                    x=alt.X("Component:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("RetentionPct:Q", title="Retained Parameters (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("Component:N", title="Component"),
                    column=alt.Column("layer:O", title="Transformer Layer", header=alt.Header(labelOrient="bottom", titleOrient="bottom")),
                    tooltip=["layer:O", "role:N", "Component:N", "RetentionPct:Q"],
                )
                .properties(height=240, title="Per-Layer Component Retention (Protected Interface vs Intermediate Sliced)")
            )
            st.altair_chart(ret_chart, use_container_width=True)

            display_cols = [
                "layer", "role", "retained_heads_count", "sliced_heads_count",
                "head_variance_scale", "retained_mlp_channels", "sliced_mlp_channels", "mlp_variance_scale"
            ]
            st.dataframe(layer_df[display_cols], use_container_width=True)
    else:
        st.warning("Slicing report artifact not found.")

    st.markdown("---")
    st.subheader("Stage 4 · Localized Error Compensation")
    if comp_rep:
        st.markdown(f"**Method Executed:** `{comp_rep.get('method_label', comp_rep.get('method'))}`")
        st.markdown(
            f"**Ridge Lambda:** `{comp_rep.get('ridge_lambda')}` | "
            f"**Compensated Layers:** `{comp_rep.get('compensated_layers')}` | "
            f"**Status:** `{comp_rep.get('status')}`"
        )
        l_reports = comp_rep.get("layer_reports", [])
        if l_reports:
            st.dataframe(pd.DataFrame(l_reports), use_container_width=True)
    else:
        st.warning("Compensation report artifact not found.")
