"""
Screens 01 (Overview), 02 (Transformation Pipeline), 03 (Model Specifications)
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import streamlit as st

_PROTO_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _PROTO_ROOT.parent
for _p in (str(_PROTO_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import OSSliceConfig
from catalog.manager import SUPPORTED_MODELS, ModelCatalogManager
from research.benchmark_registry import BenchmarkRegistry
from research.event_logger import EventLogger
from research.experiment_manager import ExperimentManager
from research.experiment_schema import ExperimentManifest, ExperimentStatus
from research_lab.components.common import (
    get_artifact_manager,
    get_selected_manifest,
    load_experiment_json,
    load_hf_model_and_tokenizer,
)
from research_lab.styles.theme import badge_html, safe_rerun, state_badge


def render_overview(on_navigate):
    st.title("OS-Slice Research Lab")
    st.markdown(
        "**Operation-Specific Behavioral Slicing (OS-Slice)** isolates the parameter sub-circuits "
        "responsible for deterministic enterprise operations (such as Strict JSON Schema Extraction) "
        "and zero-masks non-salient generalist attention heads and MLP channels."
    )

    st.info(
        "Truthful Implemented Pipeline: Pretrained Base LLM -> Stage 1: Operational Calibration & Boundary Token Weighting "
        "-> Stage 2: Empirical Fisher Saliency Profiling -> Stage 3: Layer-Aware Structural Slicing "
        "-> Stage 4: Diagonal Ridge-Inspired Compensation -> Paired Evaluation. "
        "(Teacher-student distillation is not part of the current implementation.)"
    )

    am = get_artifact_manager()
    manifests = am.list_manifests()
    completed = [m for m in manifests if m.status == ExperimentStatus.COMPLETED]
    hist_card = BenchmarkRegistry(am).get_historical_reference_card()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="os-card">', unsafe_allow_html=True)
        st.markdown("#### Validated Model & Task")
        st.markdown("**Model:** `Qwen/Qwen2.5-0.5B-Instruct`")
        st.markdown("**Task:** `Structured JSON Extraction`")
        st.markdown(badge_html("Verified Prototype Target", "green"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="os-card">', unsafe_allow_html=True)
        st.markdown("#### Latest Measured Metrics")
        if completed and completed[0].metrics:
            m = completed[0].metrics
            st.markdown(f"**Experiment:** `{completed[0].experiment_id}` (n={m.get('sample_count', 0)})")
            st.markdown(f"**Strict JSON:** Base `{m.get('base_strict_valid_rate', 0)*100:.1f}%` vs Sliced `{m.get('sliced_strict_valid_rate', 0)*100:.1f}%`")
            st.markdown(f"**Active Reduction:** `-{m.get('active_param_reduction_pct', 0):.2f}%`")
        elif hist_card:
            st.markdown(f"**Historical Reference:** `{hist_card['card_id']}` (n={hist_card['sample_count']})")
            st.markdown("**Normalized JSON:** `100%` (4/4, markdown-fenced)")
            st.markdown(f"**Active Reduction:** `-{hist_card['active_reduction_pct']:.2f}%`")
        else:
            st.markdown("No completed runs yet.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="os-card">', unsafe_allow_html=True)
        st.markdown("#### Quick Actions")
        if st.button("Start Guided Jury Demo"):
            st.session_state["jury_mode"] = True
            st.session_state["jury_step_idx"] = 0
            on_navigate("03 Model Specifications")
        if st.button("Open Transformation Pipeline"):
            on_navigate("02 Transformation Pipeline")
        if st.button("Open Model Arena"):
            on_navigate("06 Model Arena")
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Recent Experiments")
    if manifests:
        rows = [
            {
                "Experiment ID": m.experiment_id,
                "Status": m.status.value,
                "Model": m.model.model_id,
                "Operation": m.operation,
                "Ratio": f"{m.slicing.ratio * 100:.0f}%",
                "Interface": m.runtime.source_interface,
                "Created": m.created_at[:19].replace("T", " "),
            }
            for m in manifests[:10]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.caption("No live experiments recorded in `experiments/` yet. Run one in Transformation Pipeline or CLI.")


def render_transformation_pipeline(on_navigate):
    st.title("02 · Transformation Pipeline")
    st.markdown(
        "Execute the real 4-stage OS-Slice pipeline end-to-end or inspect any recorded experiment's "
        "stage statuses, timings, event trace, and artifact links."
    )

    am = get_artifact_manager()
    with st.expander("Run New End-to-End OS-Slice Experiment", expanded=False):
        tier1_models = [m["id"] for m in SUPPORTED_MODELS if m.get("tier") == "Tier 1: Micro-Edge"]
        col1, col2, col3 = st.columns(3)
        with col1:
            model_id = st.selectbox("Foundational Model", tier1_models, index=1 if len(tier1_models) > 1 else 0)
        with col2:
            slicing_ratio = st.slider("Slicing Ratio", 0.05, 0.50, 0.20, 0.05)
        with col3:
            ridge_lambda = st.number_input("Ridge Lambda", min_value=0.0, max_value=0.1, value=0.001, step=0.0005, format="%.4f")

        c_opt1, c_opt2, c_opt3 = st.columns(3)
        with c_opt1:
            comp_enabled = st.checkbox("Enable Diagonal Ridge Compensation", value=True)
        with c_opt2:
            force_reprofile = st.checkbox("Force Fresh Fisher Profiling", value=False)
        with c_opt3:
            eval_limit = st.number_input("Eval Samples (max 12)", min_value=1, max_value=12, value=4)

        exp_label = st.text_input("Experiment Label (optional)", value="")

        if st.button("Execute Real 4-Stage Pipeline"):
            cfg = OSSliceConfig(model_id=model_id, default_slicing_ratio=slicing_ratio)
            status_box = st.empty()
            prog_bar = st.progress(0)
            try:
                status_box.info(f"Loading `{model_id}` on `{cfg.device}`...")
                tokenizer, base_model = load_hf_model_and_tokenizer(model_id, cfg.device)

                from data.generate_calibration import CALIBRATION_RECORDS, EVALUATION_RECORDS

                def _cb(stage_name: str, frac: float):
                    stage_offsets = {"calibration": 0.1, "profiling": 0.4, "slicing": 0.6, "compensation": 0.7, "evaluation": 1.0}
                    val = min(1.0, max(0.05, stage_offsets.get(stage_name, 0.5) * frac))
                    prog_bar.progress(int(val * 100))
                    status_box.info(f"Running stage: `{stage_name}`...")

                mgr = ExperimentManager(am)
                res = mgr.run_full_experiment(
                    base_model=base_model,
                    tokenizer=tokenizer,
                    config=cfg,
                    slicing_ratio=slicing_ratio,
                    compensation_enabled=comp_enabled,
                    ridge_lambda=ridge_lambda,
                    calibration_records=CALIBRATION_RECORDS,
                    eval_records=EVALUATION_RECORDS[: int(eval_limit)],
                    source_interface="research_lab",
                    label=exp_label,
                    force_reprofile=force_reprofile,
                    progress_callback=_cb,
                )
                st.session_state["active_experiment_id"] = res["manifest"].experiment_id
                prog_bar.progress(100)
                status_box.success(f"Experiment `{res['manifest'].experiment_id}` completed!")
                safe_rerun()
            except Exception as exc:
                status_box.error(f"Experiment failed: {exc}")

    st.markdown("---")
    manifests = am.list_manifests()
    manifest = get_selected_manifest(manifests)
    if manifest is None:
        st.info("No experiment manifests found yet. Start one above or run `oslice` CLI.")
        return

    st.markdown(f"**Experiment:** `{manifest.experiment_id}` | **Status:** {state_badge(manifest.status.value)}", unsafe_allow_html=True)

    stage_order = ["calibration", "profiling", "slicing", "compensation", "evaluation"]
    stage_descs = {
        "calibration": "Tokenize enterprise records and apply boundary token weights (6.0x delimiters/keys, 4.2x numeric, 1.0x payload).",
        "profiling": "Compute empirical Fisher Information saliency (grad^2 * w^2) across attention heads and MLP channels.",
        "slicing": "Protect outer layers (0, 1, L-2, L-1) and zero-mask lowest-saliency intermediate heads/channels with variance scaling.",
        "compensation": "Apply diagonal ridge-inspired damping on surviving active down_proj and o_proj weights.",
        "evaluation": "Evaluate base vs sliced model on identical unseen evaluation records with strict JSON validation.",
    }

    cols = st.columns(5)
    for idx, sname in enumerate(stage_order):
        rec = manifest.stages.get(sname)
        with cols[idx]:
            st.markdown('<div class="os-card">', unsafe_allow_html=True)
            st.markdown(f"**Stage {idx + 1}: {sname.title()}**")
            if rec:
                st.markdown(state_badge(rec.status.value), unsafe_allow_html=True)
                st.caption(f"Duration: `{rec.duration_s or 0.0:.3f}s`")
            else:
                st.markdown(badge_html("pending", "gray"), unsafe_allow_html=True)
            st.caption(stage_descs[sname])
            st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Stage Details & Event Trace")
    exp_dir = am.get_experiment_dir(manifest.experiment_id)
    events = EventLogger(manifest.experiment_id, exp_dir / "execution_events.jsonl").read_events()
    if events:
        ev_rows = [
            {
                "Timestamp": e.timestamp[:19].replace("T", " "),
                "Stage": e.stage,
                "Event": e.event_name,
                "Status": e.status,
                "Metadata": str(e.metadata),
            }
            for e in events
        ]
        st.dataframe(pd.DataFrame(ev_rows), use_container_width=True)
    else:
        st.caption("No execution events recorded for this manifest.")


def render_model_specifications():
    st.title("03 · Model Specifications")
    st.markdown(
        "Inspect base and sliced model identities, architectural dimensions, active vs zero-masked parameters, "
        "and physical memory/disk footprint."
    )

    am = get_artifact_manager()
    manifests = am.list_manifests()
    manifest = get_selected_manifest(manifests)

    if manifest is None:
        st.info("Select or run an experiment to inspect exact model tensor specifications. Showing catalog hardware profile below:")
        hw = ModelCatalogManager.get_system_hardware_profile()
        st.json(hw)
        return

    slicing_rep = load_experiment_json(manifest.experiment_id, "slicing/slicing_report.json")
    m = manifest.model

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="os-card">', unsafe_allow_html=True)
        st.markdown("### Base Foundational Model")
        st.markdown(f"- **Model ID:** `{m.model_id}`")
        st.markdown(f"- **Revision:** `{m.revision}`")
        st.markdown(f"- **Architecture:** `{m.architecture}`")
        st.markdown(f"- **Dtype / Device:** `{m.dtype}` on `{manifest.runtime.device}`")
        st.markdown(f"- **Layers / Heads / KV Heads:** `{m.num_layers}` / `{m.num_heads}` / `{m.num_kv_heads}`")
        st.markdown(f"- **Hidden / Intermediate Size:** `{m.hidden_size}` / `{m.intermediate_size}`")
        st.markdown(f"- **Total Parameters:** `{m.total_params:,}`" if m.total_params else "- **Total Parameters:** N/A")
        if m.physical_size_bytes:
            st.markdown(f"- **In-Memory Tensor Size:** `{m.physical_size_bytes / (1024**2):.2f} MB`")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="os-card">', unsafe_allow_html=True)
        st.markdown("### OS-Slice Operational Model")
        st.markdown(f"- **Slice ID:** `{manifest.slice_id}`")
        st.markdown(f"- **Target Operation:** `{manifest.operation}`")
        st.markdown(f"- **Configured Slicing Ratio:** `{manifest.slicing.ratio * 100:.1f}%`")
        if slicing_rep:
            st.markdown(f"- **Active Structural Params:** `{slicing_rep['active_params_structural']:,}`")
            st.markdown(f"- **Zero-Masked Structural Params:** `{slicing_rep['sliced_params_structural']:,}` (`-{slicing_rep['effective_active_sparsity_pct']:.2f}%`)")
            st.markdown(f"- **Actual Tensor Non-Zero Params:** `{slicing_rep['actual_tensor_nonzero_params']:,}`")
            st.markdown(f"- **Actual Tensor Zeroed Params:** `{slicing_rep['actual_tensor_zeroed_params']:,}`")
        st.warning(
            "Physical Compression Note: OS-Slice currently zero-masks dense tensors. "
            "Physical tensor dimensions and serialized disk size are NOT reduced by zero-masking."
        )
        st.markdown("</div>", unsafe_allow_html=True)
