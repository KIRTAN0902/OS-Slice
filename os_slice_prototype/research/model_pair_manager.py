"""
OS-Slice Model-Pair Manager
Resolves matching (base_model, task_sliced_model) pairs, enforces lineage
and task-pair integrity rules, and reports readiness states.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .experiment_schema import CardState
from .task_registry import TaskRegistry

# Ensure catalog package resolves cleanly
_PROTO_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PROTO_ROOT.parent
for _p in (str(_PROTO_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from catalog.manager import SUPPORTED_MODELS, ModelCatalogManager
from catalog.sliced_models import SLICED_MODELS


_SLICE_TASK_MAP = {
    "qwen2.5-0.5b-json-slice": "structured_json_extraction",
    "smollm2-360m-sql-slice": "sql_synthesis",
    "llama-3.2-1b-tool-slice": "api_tool_calling",
    "qwen2.5-1.5b-pii-slice": "ner_pii_extraction",
    "qwen2.5-7b-react-slice": "operational_action_planning",
    "llama-3.1-8b-legal-slice": "structured_json_extraction",
}


class ModelPairManager:
    """Validates and resolves base/sliced model pairs for a specific operational task."""

    @staticmethod
    def list_base_models() -> List[Dict[str, Any]]:
        return list(SUPPORTED_MODELS)

    @staticmethod
    def get_task_for_slice(slice_id: str) -> str:
        return _SLICE_TASK_MAP.get(slice_id, "structured_json_extraction")

    @staticmethod
    def check_pair_readiness(model_id: str, task_id: str) -> Dict[str, Any]:
        """
        Determines whether a valid (base, sliced) pair can be executed for (model_id, task_id).
        Enforces Model-pair integrity rules from Section 7 of the specification.
        """
        task_spec = TaskRegistry.get_task(task_id)
        if task_spec is None:
            return {
                "ready": False,
                "state": CardState.NOT_EVALUATED.value,
                "reason": f"Unknown task '{task_id}'.",
                "slice_id": None,
                "base_cached": False,
                "profile_available": False,
            }

        if not TaskRegistry.is_task_implemented(task_id):
            return {
                "ready": False,
                "state": CardState.NOT_EVALUATED.value,
                "reason": task_spec.status_note or f"Task '{task_spec.display_name}' has no implemented evaluator.",
                "slice_id": None,
                "base_cached": ModelCatalogManager.is_model_cached(model_id),
                "profile_available": False,
            }

        # Find matching slice entry for (model_id, task_id)
        matching_slice = None
        for s in SLICED_MODELS:
            if s["base_model_id"] == model_id and ModelPairManager.get_task_for_slice(s["id"]) == task_id:
                matching_slice = s
                break

        base_cached = ModelCatalogManager.is_model_cached(model_id)
        model_meta = ModelCatalogManager.get_model_by_id(model_id) or {}
        tier = model_meta.get("tier", "")
        if tier in ("Tier 2: Mid-Range", "Tier 3: Frontier"):
            hw = ModelCatalogManager.get_system_hardware_profile()
            if not hw.get("has_cuda", False):
                return {
                    "ready": False,
                    "state": CardState.CLOUD_EXECUTION_REQUIRED.value,
                    "reason": f"{model_id} ({model_meta.get('min_vram', 'high VRAM')}) requires cloud/GPU execution.",
                    "slice_id": matching_slice["id"] if matching_slice else None,
                    "base_cached": base_cached,
                    "profile_available": False,
                }

        # Check if a precomputed profile or on-demand slicing is supported
        profile_path = _PROTO_ROOT / "cache" / "profiles" / "operational_fisher_profile.pt"
        profile_available = (
            matching_slice is not None
            and matching_slice.get("profile_cache_path") is not None
            and profile_path.exists()
        )

        # For implemented tasks on local-capable Tier 1 models, we can run if model is cached or downloadable
        if matching_slice is not None and matching_slice.get("status") == "LOCAL_ACTIVE" and profile_available:
            return {
                "ready": True,
                "state": CardState.VALIDATED_BENCHMARK.value if base_cached else CardState.READY_TO_EXECUTE.value,
                "reason": (
                    "Matched base model and precomputed task-specific Fisher profile are ready."
                    if base_cached
                    else "Precomputed Fisher profile exists; base model will be downloaded from HuggingFace Hub on first run."
                ),
                "slice_id": matching_slice["id"],
                "base_cached": base_cached,
                "profile_available": True,
            }

        # Other Tier 1 models can be sliced on-demand for structured_json_extraction
        if tier == "Tier 1: Micro-Edge":
            return {
                "ready": True,
                "state": CardState.READY_TO_EXECUTE.value,
                "reason": (
                    f"Ready to profile and slice {model_id} for {task_spec.display_name} on demand."
                    + ("" if base_cached else " (Base model will be fetched from HF Hub.)")
                ),
                "slice_id": f"{model_id.split('/')[-1].lower()}-json-slice",
                "base_cached": base_cached,
                "profile_available": False,
            }

        return {
            "ready": False,
            "state": CardState.NOT_EVALUATED.value,
            "reason": f"No validated slice or local execution profile available for {model_id} on {task_spec.display_name}.",
            "slice_id": matching_slice["id"] if matching_slice else None,
            "base_cached": base_cached,
            "profile_available": False,
        }
