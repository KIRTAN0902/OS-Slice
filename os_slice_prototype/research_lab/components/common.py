"""
Shared Research Lab Components:
- Cached model loading (strictly separated from experiment result persistence)
- Experiment selector and artifact readers
- Jury Guided Demo banner
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st
import torch

_PROTO_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _PROTO_ROOT.parent
for _p in (str(_PROTO_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from research.artifact_manager import ArtifactManager
from research.event_logger import EventLogger
from research.experiment_schema import ExperimentManifest


@st.cache_resource(show_spinner=False)
def load_hf_model_and_tokenizer(model_id: str, device: str) -> Tuple[Any, Any]:
    """Loads tokenizer and base model once into memory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tokenizer, model


def get_artifact_manager() -> ArtifactManager:
    return ArtifactManager()


def load_experiment_json(exp_id: str, rel_path: str) -> Optional[Any]:
    am = get_artifact_manager()
    p = am.get_experiment_dir(exp_id) / rel_path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_experiment_tensor(exp_id: str, rel_path: str) -> Optional[Any]:
    am = get_artifact_manager()
    p = am.get_experiment_dir(exp_id) / rel_path
    if not p.exists():
        return None
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None


def get_selected_manifest(manifests: List[ExperimentManifest]) -> Optional[ExperimentManifest]:
    if not manifests:
        return None
    exp_ids = [m.experiment_id for m in manifests]
    default_id = st.session_state.get("active_experiment_id")
    idx = exp_ids.index(default_id) if default_id in exp_ids else 0
    selected_id = st.selectbox(
        "Active Experiment Record",
        exp_ids,
        index=idx,
        format_func=lambda eid: next(
            (f"{m.experiment_id} | {m.status.value.upper()} | {m.label}" for m in manifests if m.experiment_id == eid),
            eid,
        ),
    )
    st.session_state["active_experiment_id"] = selected_id
    return next((m for m in manifests if m.experiment_id == selected_id), None)


JURY_STEPS = [
    ("01 Overview", "1. What is OS-Slice & what problem does it solve?"),
    ("03 Model Specifications", "2. What does the original base model contribute?"),
    ("02 Transformation Pipeline", "3. Follow the 4-stage OS-Slice transformation."),
    ("04 Behavioral Profiling", "4. Inspect boundary weights & Fisher saliency."),
    ("05 Structural Slicing", "5. Inspect masks, protected layers & compensation."),
    ("06 Model Arena", "6. Compare base vs sliced raw outputs side by side."),
    ("08 Performance Dashboard", "7. Review paired benchmark metrics & deltas."),
    ("07 Benchmark Library", "8. Inspect historical & verified benchmark cards."),
    ("13 Research Evidence", "9. Audit manifest, SHA-256 hashes & export report."),
]


def render_jury_banner(current_page: str, on_navigate):
    if not st.session_state.get("jury_mode", False):
        return
    step_idx = st.session_state.get("jury_step_idx", 0)
    step_idx = max(0, min(step_idx, len(JURY_STEPS) - 1))
    target_page, prompt_text = JURY_STEPS[step_idx]

    st.markdown(
        f"""
        <div class="jury-banner">
            <strong>🎓 Jury Guided Walkthrough (Step {step_idx + 1} / {len(JURY_STEPS)})</strong> — {prompt_text}<br/>
            <span style="color:#9ca3af; font-size:0.85rem;">
                Every number and artifact shown is traced to an actual experiment manifest or explicitly labelled as a historical reference.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    with c1:
        if st.button("⬅ Previous Step", disabled=(step_idx == 0)):
            st.session_state["jury_step_idx"] = step_idx - 1
            on_navigate(JURY_STEPS[step_idx - 1][0])
    with c2:
        if st.button("Next Step ➡", disabled=(step_idx >= len(JURY_STEPS) - 1)):
            st.session_state["jury_step_idx"] = step_idx + 1
            on_navigate(JURY_STEPS[step_idx + 1][0])
    with c3:
        if current_page != target_page:
            if st.button(f"Jump to Recommended Screen ({target_page})"):
                on_navigate(target_page)
    with c4:
        if st.button("Exit Guided Mode"):
            st.session_state["jury_mode"] = False
            on_navigate(current_page)
