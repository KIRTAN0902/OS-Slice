"""
Screens 10 (Ablation Study), 11 (Compression Analysis), 12 (Live Demo), 13 (Research Evidence)
"""

import json
import sys
from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st

_PROTO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTO_ROOT))

from config import OSSliceConfig
from data.generate_calibration import CALIBRATION_RECORDS, EVALUATION_RECORDS, build_calibration_dataset, format_prompt
from engine.compensator import OperationalCompensator
from engine.profiler import OperationalProfiler
from engine.slicer import StructuralSlicer
from research.event_logger import EventLogger
from research.evidence_exporter import EvidenceExporter
from research.experiment_manager import ExperimentManager
from research.model_arena import run_single_generation
from research_lab.components.common import (
    get_artifact_manager,
    get_selected_manifest,
    load_experiment_json,
    load_hf_model_and_tokenizer,
)
from research_lab.styles.theme import badge_html, state_badge


def render_ablation_study():
    st.title("10 · Ablation Study")
    st.markdown(
        "Isolates the contribution of Fisher behavioral slicing and diagonal ridge compensation "
        "against the unmodified base model (`E0`) and a matched-budget L1 magnitude pruning baseline (`E2`)."
    )

    am = get_artifact_manager()
    manifest = get_selected_manifest(am.list_manifests())

    with st.expander("Run Controlled Ablation Study (E0, E2, E3, E4)", expanded=False):
        model_id = st.text_input("Model ID", value="Qwen/Qwen2.5-0.5B-Instruct", key="abl_model")
        ratio = st.slider("Matched Slicing Ratio", 0.05, 0.40, 0.20, 0.05, key="abl_ratio")
        eval_n = st.number_input("Evaluation Samples", 1, 12, 4, key="abl_n")
        if st.button("Execute Ablation Suite"):
            cfg = OSSliceConfig(model_id=model_id, default_slicing_ratio=ratio)
            with st.spinner("Running controlled ablation configurations E0, E2, E3, E4..."):
                tok, model = load_hf_model_and_tokenizer(model_id, cfg.device)
                mgr = ExperimentManager(am)
                abl = mgr.run_ablation_study(
                    base_model=model,
                    tokenizer=tok,
                    config=cfg,
                    slicing_ratio=ratio,
                    eval_records=EVALUATION_RECORDS[: int(eval_n)],
                    calibration_records=CALIBRATION_RECORDS,
                    parent_manifest=manifest,
                )
                st.session_state["last_ablation"] = [a.model_dump() for a in abl]

    abl_data = None
    if manifest is not None:
        abl_data = load_experiment_json(manifest.experiment_id, "evaluation/ablation_results.json")
    if abl_data is None:
        abl_data = st.session_state.get("last_ablation")

    if abl_data:
        df = pd.DataFrame(abl_data)
        abl_chart = (
            alt.Chart(df)
            .transform_fold(["schema_adherence", "strict_valid_rate"], as_=["Metric", "Score"])
            .mark_bar()
            .encode(
                x=alt.X("config_id:N", title="Ablation Configuration"),
                y=alt.Y("Score:Q", title="Score (0-1)", scale=alt.Scale(domain=[0, 1.05])),
                color=alt.Color("Metric:N", title="Metric"),
                xOffset="Metric:N",
                tooltip=["config_id", "config_name", "Metric:N", "Score:Q", "active_reduction_pct"],
            )
            .properties(height=300, title="Ablation Study: Behavioral Slicing & Compensation Contribution")
        )
        st.altair_chart(abl_chart, use_container_width=True)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No ablation results recorded yet. Expand above to execute a controlled ablation run.")
        planned = [
            {"ID": "E0", "Configuration": "Original base model", "Research Question": "What is the unmodified baseline?"},
            {"ID": "E2", "Configuration": "L1 magnitude pruning (matched ratio & protected layers)", "Research Question": "How does conventional magnitude pruning compare?"},
            {"ID": "E3", "Configuration": "Behavioral Fisher slicing (no compensation)", "Research Question": "What does OS-Slice's saliency selection contribute?"},
            {"ID": "E4", "Configuration": "Behavioral Fisher slicing + diagonal ridge compensation", "Research Question": "Does compensation improve or preserve behavior?"},
        ]
        st.dataframe(pd.DataFrame(planned), use_container_width=True)


def render_compression_analysis():
    st.title("11 · Compression-Level & Pareto Analysis")
    st.markdown(
        "Multi-ratio sweep measuring task validity and schema adherence across slicing ratios. "
        "Separates active parameter reduction from physical memory/model-size reduction."
    )

    am = get_artifact_manager()
    manifest = get_selected_manifest(am.list_manifests())

    with st.expander("Run Multi-Ratio Compression Sweep", expanded=False):
        model_id = st.text_input("Model ID", value="Qwen/Qwen2.5-0.5B-Instruct", key="sweep_model")
        ratios = st.multiselect(
            "Target Slicing Ratios",
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
            default=[0.10, 0.20, 0.30, 0.40],
        )
        eval_n = st.number_input("Evaluation Samples", 1, 12, 4, key="sweep_n")
        if st.button("Run Compression Sweep") and ratios:
            cfg = OSSliceConfig(model_id=model_id)
            with st.spinner("Running multi-ratio sweep..."):
                tok, model = load_hf_model_and_tokenizer(model_id, cfg.device)
                mgr = ExperimentManager(am)
                pts = mgr.run_compression_sweep(
                    base_model=model,
                    tokenizer=tok,
                    config=cfg,
                    ratios=sorted(ratios),
                    eval_records=EVALUATION_RECORDS[: int(eval_n)],
                    calibration_records=CALIBRATION_RECORDS,
                    parent_manifest=manifest,
                )
                st.session_state["last_sweep"] = [p.model_dump() for p in pts]

    sweep_data = None
    if manifest is not None:
        sweep_data = load_experiment_json(manifest.experiment_id, "evaluation/compression_sweep.json")
    if sweep_data is None:
        sweep_data = st.session_state.get("last_sweep")

    if sweep_data:
        df = pd.DataFrame(sweep_data)
        st.dataframe(df, use_container_width=True)
        feasible_df = df[df["feasible"] == True]
        if not feasible_df.empty:
            points_layer = (
                alt.Chart(feasible_df)
                .mark_circle(size=120)
                .encode(
                    x=alt.X("active_reduction_pct:Q", title="Active Parameter Reduction (%)"),
                    y=alt.Y("schema_adherence:Q", title="Schema Key Adherence"),
                    color=alt.Color("is_pareto_efficient:N", title="Pareto Efficient"),
                    tooltip=["target_ratio", "active_reduction_pct", "strict_valid_rate", "schema_adherence", "avg_latency_s"],
                )
            )
            pareto_df = feasible_df[feasible_df["is_pareto_efficient"] == True].sort_values("active_reduction_pct")
            pareto_line = (
                alt.Chart(pareto_df)
                .mark_line(strokeDash=[4, 4], color="#38bdf8")
                .encode(x="active_reduction_pct:Q", y="schema_adherence:Q")
            )
            st.altair_chart(
                (points_layer + pareto_line).properties(height=320, title="Measured Accuracy vs Active Parameter Reduction (Pareto Frontier)"),
                use_container_width=True,
            )

            lat_chart = (
                alt.Chart(feasible_df)
                .mark_line(point=True, color="#fbbf24")
                .encode(
                    x=alt.X("target_ratio:Q", title="Target Slicing Ratio"),
                    y=alt.Y("avg_latency_s:Q", title="Measured Generation Latency (s)"),
                    tooltip=["target_ratio", "avg_latency_s", "throughput_tps", "peak_ram_mb"],
                )
                .properties(height=260, title="Measured Generation Latency vs Slicing Ratio (Dense Zero-Masked Tensors)")
            )
            st.altair_chart(lat_chart, use_container_width=True)
    else:
        st.info("No measured sweep points recorded yet. Run a sweep above to plot measured Pareto points.")


def render_live_demo():
    st.title("12 · Live Demo — Operational Inference Playground")
    st.markdown("Interactive single-prompt execution calling the real OS-Slice sliced model.")

    from catalog.manager import ModelCatalogManager
    hw = ModelCatalogManager.get_system_hardware_profile()
    c_sys, c_live = st.columns([1, 2])
    with c_sys:
        st.markdown("#### System Configuration")
        st.markdown(f"- **Compute:** `{'CUDA: ' + str(hw.get('cuda_device_name')) if hw.get('has_cuda') else 'CPU'}`")
        st.markdown(f"- **System RAM:** `{hw.get('ram_avail_gb')} GB free / {hw.get('ram_total_gb')} GB total`")
        st.markdown("- **Target Model:** `Qwen/Qwen2.5-0.5B-Instruct`")
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        ratio = st.slider("Slicing Ratio", 0.05, 0.50, 0.20, 0.05, key="demo_ratio")

    with c_live:
        preset = EVALUATION_RECORDS[0]
        ctx = st.text_area("Input Context", value=preset["context"], height=100, key="demo_ctx")
        instr = st.text_input("Schema Instruction", value=preset["schema_instruction"], key="demo_instr")

        if st.button("Run Real Sliced Model"):
            cfg = OSSliceConfig(model_id=model_id, default_slicing_ratio=ratio)
            with st.spinner("Running real OS-Slice inference..."):
                tok, base_model = load_hf_model_and_tokenizer(model_id, cfg.device)
                prof = OperationalProfiler(base_model, cfg)
                p_path = cfg.profiles_dir / "operational_fisher_profile.pt"
                if p_path.exists():
                    prof.load_profile(p_path)
                else:
                    prof.profile(build_calibration_dataset(tok, CALIBRATION_RECORDS))
                sliced = StructuralSlicer(cfg).apply_slicing(base_model, prof.head_saliency, prof.mlp_saliency, slicing_ratio=ratio, in_place=False)
                OperationalCompensator(cfg).compensate(sliced)
                res = run_single_generation(
                    sliced, tok, format_prompt(ctx, instr), "sliced", f"{model_id}::demo", "structured_json_extraction",
                    max_new_tokens=180, temperature=0.0, seed=123, device=cfg.device,
                )
                st.markdown(f"**Run Status:** `{res.run_status}` | **Latency:** `{res.total_latency_s:.3f}s` | **Throughput:** `{res.throughput_tps:.1f} tok/s`")
                st.code(res.raw_output, language="json")


def render_research_evidence():
    st.title("13 · Research Evidence & Artifact Audit")
    st.markdown("Inspect the complete experiment manifest, stage event log, SHA-256 checksums, and exportable evidence report.")

    am = get_artifact_manager()
    manifest = get_selected_manifest(am.list_manifests())
    if manifest is None:
        st.info("No experiment selected. Run an experiment in Transformation Pipeline or CLI to inspect its evidence bundle.")
        return

    verification = am.verify_artifacts(manifest)
    v_badge = badge_html("All Artifact Checksums Verified", "green") if verification["all_valid"] else badge_html("Artifact Verification Warning", "red")
    st.markdown(f"**Experiment:** `{manifest.experiment_id}` &nbsp; {state_badge(manifest.status.value)} &nbsp; {v_badge}", unsafe_allow_html=True)

    if st.button("Regenerate Evidence Report (JSON + Markdown)"):
        paths = EvidenceExporter(am).export(manifest)
        st.success(f"Exported report to `{paths['markdown_path']}`")

    tab1, tab2, tab3, tab4 = st.tabs(["Manifest", "Artifact Checksums", "Execution Events", "Raw Predictions"])
    with tab1:
        st.json(manifest.model_dump())
    with tab2:
        rows = [
            {"Artifact": k, "Path": v["relative_path"], "Exists": v["exists"], "ChecksumMatch": v["checksum_match"], "SHA256": v["expected_sha256"]}
            for k, v in verification["artifacts"].items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    with tab3:
        exp_dir = am.get_experiment_dir(manifest.experiment_id)
        events = EventLogger(manifest.experiment_id, exp_dir / "execution_events.jsonl").read_events()
        st.dataframe(pd.DataFrame([e.model_dump() for e in events]), use_container_width=True)
    with tab4:
        preds = load_experiment_json(manifest.experiment_id, "evaluation/predictions.json")
        if preds:
            st.json(preds)
        else:
            st.caption("No evaluation predictions artifact found for this experiment.")

    md_path = am.get_experiment_dir(manifest.experiment_id) / "report" / "experiment_report.md"
    if md_path.exists():
        st.download_button(
            "Download Experiment Report (Markdown)",
            data=md_path.read_text(encoding="utf-8"),
            file_name=f"{manifest.experiment_id}_report.md",
            mime="text/markdown",
        )
