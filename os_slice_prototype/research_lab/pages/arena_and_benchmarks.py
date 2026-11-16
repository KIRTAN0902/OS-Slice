"""
Screens 06 (Model Arena), 07 (Benchmark Library), 08 (Performance Dashboard), 09 (Task Evaluation)
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
from research.benchmark_comparator import BenchmarkComparator
from research.benchmark_registry import BenchmarkRegistry
from research.experiment_schema import ExperimentStatus
from research.model_arena import ModelArenaService
from research.model_pair_manager import ModelPairManager
from research.task_registry import TaskRegistry
from research_lab.components.common import (
    get_artifact_manager,
    get_selected_manifest,
    load_experiment_json,
    load_hf_model_and_tokenizer,
)
from research_lab.styles.theme import badge_html, state_badge


def render_model_arena():
    st.title("06 · Model Arena — Paired Live Inference")
    st.markdown(
        "Run the exact same input and decoding settings through a verified `(Base Model, OS-Slice Model)` pair. "
        "Raw outputs are preserved without silent repair before scoring."
    )

    models = ModelPairManager.list_base_models()
    model_ids = [m["id"] for m in models]
    tasks = TaskRegistry.list_tasks()
    task_ids = [t.task_id for t in tasks]

    col1, col2 = st.columns(2)
    with col1:
        default_m_idx = model_ids.index("Qwen/Qwen2.5-0.5B-Instruct") if "Qwen/Qwen2.5-0.5B-Instruct" in model_ids else 0
        sel_model_id = st.selectbox("1. Foundational Model", model_ids, index=default_m_idx)
    with col2:
        sel_task_id = st.selectbox(
            "2. Operational Task",
            task_ids,
            format_func=lambda tid: f"{TaskRegistry.get_task(tid).display_name} ({TaskRegistry.get_task(tid).readiness.value})",
        )

    readiness = ModelPairManager.check_pair_readiness(sel_model_id, sel_task_id)
    st.markdown(
        f"**Pair Readiness:** {state_badge(readiness['state'])} &nbsp; {readiness['reason']}",
        unsafe_allow_html=True,
    )

    preset_idx = st.selectbox(
        "Sample Input Preset (or edit below)",
        range(len(EVALUATION_RECORDS)),
        format_func=lambda i: f"[{i}] {EVALUATION_RECORDS[i]['domain']}",
    )
    preset = EVALUATION_RECORDS[preset_idx]

    context_txt = st.text_area("Context Input", value=preset["context"], height=110)
    schema_txt = st.text_input("Schema Instruction", value=preset["schema_instruction"])
    use_gt = st.checkbox("Score against preset ground-truth JSON", value=True)
    ground_truth = preset["target_json"] if use_gt else None

    c_cfg1, c_cfg2, c_cfg3, c_cfg4 = st.columns(4)
    with c_cfg1:
        slicing_ratio = st.slider("Slicing Ratio", 0.05, 0.50, 0.20, 0.05)
    with c_cfg2:
        max_new_tokens = st.number_input("Max New Tokens", 32, 512, 180, 16)
    with c_cfg3:
        temperature = st.number_input("Temperature", 0.0, 1.0, 0.0, 0.1)
    with c_cfg4:
        seed = st.number_input("Seed", 0, 99999, 123, 1)

    run_btn = st.button("Run Paired Comparison", disabled=not readiness["ready"])
    if run_btn and readiness["ready"]:
        cfg = OSSliceConfig(model_id=sel_model_id, default_slicing_ratio=slicing_ratio)
        with st.spinner(f"Running paired inference for {sel_model_id} on {cfg.device}..."):
            tokenizer, base_model = load_hf_model_and_tokenizer(sel_model_id, cfg.device)
            profiler = OperationalProfiler(base_model, cfg)
            profile_path = cfg.profiles_dir / "operational_fisher_profile.pt"
            if sel_model_id == "Qwen/Qwen2.5-0.5B-Instruct" and profile_path.exists():
                profiler.load_profile(profile_path)
            else:
                calib_ds = build_calibration_dataset(tokenizer, CALIBRATION_RECORDS, boundary_weight=cfg.boundary_token_weight)
                profiler.profile(calib_ds)

            slicer = StructuralSlicer(cfg)
            sliced_model = slicer.apply_slicing(base_model, profiler.head_saliency, profiler.mlp_saliency, slicing_ratio=slicing_ratio, in_place=False)
            OperationalCompensator(cfg).compensate(sliced_model)

            am = get_artifact_manager()
            manifests = am.list_manifests()
            active_id = st.session_state.get("active_experiment_id")
            active_man = next((m for m in manifests if m.experiment_id == active_id), None)

            prompt = format_prompt(context_txt, schema_txt)
            arena = ModelArenaService(am)
            rec = arena.run_paired_inference(
                base_model=base_model,
                sliced_model=sliced_model,
                tokenizer=tokenizer,
                model_id=sel_model_id,
                slice_id=readiness.get("slice_id") or f"{sel_model_id}-slice",
                task_id=sel_task_id,
                prompt=prompt,
                schema_instruction=schema_txt,
                ground_truth=ground_truth,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                seed=int(seed),
                device=cfg.device,
                manifest=active_man,
            )
            st.session_state["last_arena_record"] = rec.model_dump()

    rec_data = st.session_state.get("last_arena_record")
    if rec_data:
        st.markdown("---")
        st.subheader(f"Paired Result (`{rec_data['run_id']}`)")
        col_b, col_s = st.columns(2)
        for col, role_key, title in [(col_b, "base", "Base Generalist Model"), (col_s, "sliced", "OS-Slice Operational Model")]:
            r = rec_data[role_key]
            with col:
                st.markdown('<div class="os-card">', unsafe_allow_html=True)
                st.markdown(f"### {title}")
                v_badge = badge_html("Strict JSON Valid", "green") if r["is_valid"] else badge_html("Strict JSON Invalid", "red")
                st.markdown(f"**Status:** `{r['run_status']}` &nbsp; {v_badge}", unsafe_allow_html=True)
                if r.get("validation_error"):
                    st.caption(f"Validator (`{r['validator_name']}`): {r['validation_error']}")
                if r.get("normalized_valid") and not r["is_valid"]:
                    st.caption(f"Secondary Diagnostic: `{r['normalized_status']}` (trailing_text={r['has_trailing_text']})")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Latency (s)", f"{r['total_latency_s']:.3f}")
                m2.metric("First Token (s)", f"{r['first_token_latency_s']:.3f}" if r.get("first_token_latency_s") is not None else "N/A")
                m3.metric("Tokens", r["generated_tokens"])
                m4.metric("Throughput", f"{r['throughput_tps']:.1f} t/s")

                st.markdown(f"**Schema Adherence:** `{r['schema_adherence']*100:.1f}%` | **Task Score:** `{r['task_score']*100:.1f}%`")
                st.text_area(f"Raw Output ({role_key})", value=r["raw_output"], height=180, key=f"arena_raw_{role_key}")
                st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            "Download Raw Paired Artifact (JSON)",
            data=json.dumps(rec_data, indent=2),
            file_name=f"{rec_data['run_id']}.json",
            mime="application/json",
        )


def render_benchmark_library(on_navigate):
    st.title("07 · Benchmark Library — Tested Model Cards")
    st.markdown(
        "Historical and live registry of model/task pairs. Only completed, evidence-backed runs are marked "
        "**Validated benchmark**; catalog target profiles are explicitly labelled **Ready to execute** or **Cloud execution required**."
    )

    am = get_artifact_manager()
    cards = BenchmarkRegistry(am).list_benchmark_cards()

    f1, f2, f3 = st.columns(3)
    with f1:
        state_opts = ["All"] + sorted(list({c["state"] for c in cards}))
        sel_state = st.selectbox("Filter by State", state_opts)
    with f2:
        task_opts = ["All"] + sorted(list({c["operation"] for c in cards}))
        sel_task = st.selectbox("Filter by Task", task_opts)
    with f3:
        model_opts = ["All"] + sorted(list({c["model_id"] for c in cards}))
        sel_model = st.selectbox("Filter by Model", model_opts)

    filtered = [
        c for c in cards
        if (sel_state == "All" or c["state"] == sel_state)
        and (sel_task == "All" or c["operation"] == sel_task)
        and (sel_model == "All" or c["model_id"] == sel_model)
    ]

    for c in filtered:
        st.markdown('<div class="os-card">', unsafe_allow_html=True)
        hist_tag = badge_html("Historical KT Reference", "blue") if c.get("is_historical_reference") else ""
        st.markdown(f"#### {c['title']} &nbsp; {state_badge(c['state'])} &nbsp; {hist_tag}", unsafe_allow_html=True)
        st.caption(f"ID: `{c['card_id']}` | Model: `{c['model_id']}` | Task: `{c['operation']}` | Ratio: `{c['slicing_ratio']*100:.0f}%` | Date: `{c['date']}`")
        if c.get("active_params") is not None:
            st.markdown(
                f"**Params:** `{c['active_params']:,}` / `{c['base_params']:,}` (`-{c['active_reduction_pct']:.2f}%` active) | "
                f"**Strict JSON:** `{c['strict_valid_rate']*100:.1f}%` | "
                f"**Normalized JSON:** `{c['normalized_valid_rate']*100:.1f}%` | "
                f"**Adherence:** `{c['schema_adherence']*100:.1f}%` | "
                f"**Latency:** `{c['avg_latency_s']:.2f}s` | **Throughput:** `{c['throughput_tps']:.2f} tok/s` | "
                f"**Samples:** `n={c['sample_count']}`"
            )
        if c.get("notes"):
            st.caption(c["notes"])
        if c.get("manifest") is not None:
            if st.button(f"Inspect Evidence ({c['card_id']})", key=f"btn_{c['card_id']}"):
                st.session_state["active_experiment_id"] = c["card_id"]
                on_navigate("13 Research Evidence")
        st.markdown("</div>", unsafe_allow_html=True)


def render_performance_dashboard():
    st.title("08 · Performance Dashboard")
    st.markdown(
        "Consolidated paired comparison showing Base vs OS-Slice raw values, absolute differences, "
        "relative improvement percentages, direction, sample counts, and measurement conditions."
    )

    am = get_artifact_manager()
    manifests = [m for m in am.list_manifests() if m.status == ExperimentStatus.COMPLETED and m.metrics]
    manifest = get_selected_manifest(manifests)

    if manifest is not None:
        metrics_data = load_experiment_json(manifest.experiment_id, "evaluation/metrics.json")
        rows = metrics_data.get("comparison_rows") if metrics_data else BenchmarkComparator.compare(manifest.metrics)
        st.subheader(f"Live Experiment Comparison: `{manifest.experiment_id}`")

        sm = manifest.metrics
        chart_rows = [
            {"Metric": "Strict JSON (%)", "Model": "Base", "Value": sm.get("base_strict_valid_rate", 0) * 100},
            {"Metric": "Strict JSON (%)", "Model": "OS-Slice", "Value": sm.get("sliced_strict_valid_rate", 0) * 100},
            {"Metric": "Normalized JSON (%)", "Model": "Base", "Value": sm.get("base_normalized_valid_rate", 0) * 100},
            {"Metric": "Normalized JSON (%)", "Model": "OS-Slice", "Value": sm.get("sliced_normalized_valid_rate", 0) * 100},
            {"Metric": "Schema Adherence (%)", "Model": "Base", "Value": sm.get("base_schema_adherence", 0) * 100},
            {"Metric": "Schema Adherence (%)", "Model": "OS-Slice", "Value": sm.get("sliced_schema_adherence", 0) * 100},
        ]
        perf_chart = (
            alt.Chart(pd.DataFrame(chart_rows))
            .mark_bar()
            .encode(
                x=alt.X("Metric:N", title="Evaluation Metric"),
                y=alt.Y("Value:Q", title="Score (%)", scale=alt.Scale(domain=[0, 105])),
                color=alt.Color("Model:N", scale=alt.Scale(range=["#60a5fa", "#34d399"])),
                xOffset="Model:N",
                tooltip=["Metric", "Model", "Value"],
            )
            .properties(height=300, title="Base vs OS-Slice Quality & Validity Comparison")
        )
        st.altair_chart(perf_chart, use_container_width=True)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No completed live experiment selected yet. Showing Historical KT Reference below (not a substitute for a fresh run).")

    hist = BenchmarkRegistry(am).get_historical_reference_card()
    if hist:
        with st.expander("Historical Reported Benchmark (KT Reference Only, n=4)", expanded=(manifest is None)):
            st.caption(hist["notes"])
            m = hist["raw_data"]["metrics"]
            hist_chart_rows = [
                {"Metric": "Normalized JSON (%)", "Model": "Base", "Value": m["base_valid_rate"] * 100},
                {"Metric": "Normalized JSON (%)", "Model": "OS-Slice", "Value": m["sliced_valid_rate"] * 100},
                {"Metric": "Schema Adherence (%)", "Model": "Base", "Value": m["base_adherence_avg"] * 100},
                {"Metric": "Schema Adherence (%)", "Model": "OS-Slice", "Value": m["sliced_adherence_avg"] * 100},
                {"Metric": "Throughput (tok/s)", "Model": "Base", "Value": m["base_avg_throughput_tps"]},
                {"Metric": "Throughput (tok/s)", "Model": "OS-Slice", "Value": m["sliced_avg_throughput_tps"]},
            ]
            h_chart = (
                alt.Chart(pd.DataFrame(hist_chart_rows))
                .mark_bar()
                .encode(
                    x=alt.X("Metric:N", title="Metric (Historical KT Artifact, n=4)"),
                    y=alt.Y("Value:Q", title="Measured Value"),
                    color=alt.Color("Model:N", scale=alt.Scale(range=["#60a5fa", "#34d399"])),
                    xOffset="Model:N",
                    tooltip=["Metric", "Model", "Value"],
                )
                .properties(height=280, title="Historical KT Reference: Base vs OS-Slice (n=4)")
            )
            st.altair_chart(h_chart, use_container_width=True)
            hist_rows = [
                {"Metric": "Active Parameters", "Base": m["base_active_params"], "OS-Slice": m["sliced_active_params"], "Delta": f"-{m['param_reduction_pct']:.2f}%"},
                {"Metric": "Normalized JSON Parseability (Markdown-Stripped)", "Base": f"{m['base_valid_rate']*100:.1f}%", "OS-Slice": f"{m['sliced_valid_rate']*100:.1f}%", "Delta": "0.0% (4/4)"},
                {"Metric": "Schema Key Adherence", "Base": f"{m['base_adherence_avg']*100:.1f}%", "OS-Slice": f"{m['sliced_adherence_avg']*100:.1f}%", "Delta": "0.0%"},
                {"Metric": "Average Latency (s)", "Base": f"{m['base_avg_latency_s']:.2f}s", "OS-Slice": f"{m['sliced_avg_latency_s']:.2f}s", "Delta": f"{(m['base_avg_latency_s']-m['sliced_avg_latency_s'])/m['base_avg_latency_s']*100:+.1f}% (slower due to tails)"},
                {"Metric": "Throughput (tok/s)", "Base": f"{m['base_avg_throughput_tps']:.2f}", "OS-Slice": f"{m['sliced_avg_throughput_tps']:.2f}", "Delta": f"{(m['sliced_avg_throughput_tps']-m['base_avg_throughput_tps'])/m['base_avg_throughput_tps']*100:+.2f}%"},
            ]
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True)


def render_task_evaluation():
    st.title("09 · Task Evaluation & Readiness Registry")
    st.markdown(
        "Only tasks with implemented evaluators and supported model slices can be executed. "
        "Target catalog profiles are explicitly disabled until their evaluators and slices are implemented."
    )

    tasks = TaskRegistry.list_tasks()
    for t in tasks:
        st.markdown('<div class="os-card">', unsafe_allow_html=True)
        st.markdown(f"#### {t.display_name} &nbsp; {state_badge(t.readiness.value)}", unsafe_allow_html=True)
        st.markdown(t.description)
        st.caption(f"Task ID: `{t.task_id}` | Dataset Version: `{t.dataset_version}` | Metrics: `{', '.join(t.metric_names)}`")
        st.caption(f"Status Note: {t.status_note}")
        st.markdown("</div>", unsafe_allow_html=True)
