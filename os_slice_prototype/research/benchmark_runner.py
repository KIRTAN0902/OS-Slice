"""
OS-Slice Shared Benchmark Runner, Comparator, and Registry
Executes controlled paired evaluation, computes metrics, persists predictions and
validation errors, and manages the historical Benchmark Library.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from transformers import PreTrainedTokenizerBase

from .artifact_manager import ArtifactManager
from .experiment_schema import (
    BenchmarkSummary,
    CardState,
    ExperimentManifest,
    ExperimentStatus,
)
from .metrics_collector import MetricsCollector
from .model_arena import run_single_generation

_PROTO_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PROTO_ROOT.parent
for _p in (str(_PROTO_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.generate_calibration import format_prompt


def _measure_process_ram_mb() -> Optional[float]:
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return round(proc.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return None


def _measure_tensor_bytes(model: Any) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return int(total)


class BenchmarkRunner:
    """Runs controlled paired evaluation across evaluation records."""

    def __init__(self, artifact_manager: Optional[ArtifactManager] = None):
        self.artifact_manager = artifact_manager or ArtifactManager()

    def run(
        self,
        base_model: Any,
        sliced_model: Any,
        tokenizer: PreTrainedTokenizerBase,
        eval_records: List[Dict[str, Any]],
        model_id: str,
        slice_id: str,
        slicing_ratio: float,
        task_id: str = "structured_json_extraction",
        max_new_tokens: int = 180,
        temperature: float = 0.0,
        seed: int = 123,
        device: str = "cpu",
        dtype_str: str = "float32",
        manifest: Optional[ExperimentManifest] = None,
        warmup: bool = True,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Executes paired evaluation with optional warm-up excluded from timing."""
        if warmup and eval_records:
            warm_prompt = format_prompt(eval_records[0]["context"], eval_records[0]["schema_instruction"])
            _ = run_single_generation(
                base_model, tokenizer, warm_prompt, "base", model_id, task_id,
                max_new_tokens=8, temperature=temperature, seed=seed, device=device, measure_first_token=False,
            )
            _ = run_single_generation(
                sliced_model, tokenizer, warm_prompt, "sliced", f"{model_id}::{slice_id}", task_id,
                max_new_tokens=8, temperature=temperature, seed=seed, device=device, measure_first_token=False,
            )

        base_samples: List[Dict[str, Any]] = []
        sliced_samples: List[Dict[str, Any]] = []
        validation_errors: List[Dict[str, Any]] = []

        total_n = len(eval_records)
        for idx, item in enumerate(eval_records):
            prompt = format_prompt(item["context"], item["schema_instruction"])
            expected = item.get("target_json")

            b_res = run_single_generation(
                base_model, tokenizer, prompt, "base", model_id, task_id,
                max_new_tokens=max_new_tokens, temperature=temperature, seed=seed,
                device=device, ground_truth=expected, measure_first_token=True,
            )
            s_res = run_single_generation(
                sliced_model, tokenizer, prompt, "sliced", f"{model_id}::{slice_id}", task_id,
                max_new_tokens=max_new_tokens, temperature=temperature, seed=seed,
                device=device, ground_truth=expected, measure_first_token=True,
            )

            b_row = b_res.model_dump()
            b_row["idx"] = idx
            b_row["domain"] = item.get("domain", f"Sample {idx}")
            b_row["expected_json"] = expected
            base_samples.append(b_row)

            s_row = s_res.model_dump()
            s_row["idx"] = idx
            s_row["domain"] = item.get("domain", f"Sample {idx}")
            s_row["expected_json"] = expected
            sliced_samples.append(s_row)

            if not b_res.is_valid:
                validation_errors.append({
                    "idx": idx, "role": "base", "domain": b_row["domain"],
                    "error": b_res.validation_error, "normalized_valid": b_res.normalized_valid,
                    "has_trailing_text": b_res.has_trailing_text,
                })
            if not s_res.is_valid:
                validation_errors.append({
                    "idx": idx, "role": "sliced", "domain": s_row["domain"],
                    "error": s_res.validation_error, "normalized_valid": s_res.normalized_valid,
                    "has_trailing_text": s_res.has_trailing_text,
                })

            if progress_callback is not None:
                progress_callback(idx + 1, total_n)

        n = max(len(eval_records), 1)
        base_total_params = sum(p.numel() for p in base_model.parameters())
        sliced_active_params = int(getattr(sliced_model, "active_params", base_total_params))
        reduction_pct = (1.0 - (sliced_active_params / max(base_total_params, 1))) * 100.0

        b_ft = [r["first_token_latency_s"] for r in base_samples if r["first_token_latency_s"] is not None]
        s_ft = [r["first_token_latency_s"] for r in sliced_samples if r["first_token_latency_s"] is not None]
        b_em = [r["field_exact_match"] for r in base_samples if r["field_exact_match"] is not None]
        s_em = [r["field_exact_match"] for r in sliced_samples if r["field_exact_match"] is not None]

        base_bytes = _measure_tensor_bytes(base_model)
        sliced_bytes = _measure_tensor_bytes(sliced_model.model if hasattr(sliced_model, "model") else sliced_model)

        summary = BenchmarkSummary(
            experiment_id=manifest.experiment_id if manifest else "ADHOC",
            model_id=model_id,
            operation=task_id,
            slice_id=slice_id,
            slicing_ratio=slicing_ratio,
            sample_count=len(eval_records),
            base_strict_valid_rate=sum(1 for r in base_samples if r["is_valid"]) / n,
            sliced_strict_valid_rate=sum(1 for r in sliced_samples if r["is_valid"]) / n,
            base_normalized_valid_rate=sum(1 for r in base_samples if r.get("normalized_valid")) / n,
            sliced_normalized_valid_rate=sum(1 for r in sliced_samples if r.get("normalized_valid")) / n,
            base_schema_adherence=sum(r["schema_adherence"] for r in base_samples) / n,
            sliced_schema_adherence=sum(r["schema_adherence"] for r in sliced_samples) / n,
            base_field_exact_match=(sum(b_em) / len(b_em)) if b_em else None,
            sliced_field_exact_match=(sum(s_em) / len(s_em)) if s_em else None,
            base_avg_latency_s=sum(r["total_latency_s"] for r in base_samples) / n,
            sliced_avg_latency_s=sum(r["total_latency_s"] for r in sliced_samples) / n,
            base_avg_first_token_s=(sum(b_ft) / len(b_ft)) if b_ft else None,
            sliced_avg_first_token_s=(sum(s_ft) / len(s_ft)) if s_ft else None,
            base_avg_throughput_tps=sum(r["throughput_tps"] for r in base_samples) / n,
            sliced_avg_throughput_tps=sum(r["throughput_tps"] for r in sliced_samples) / n,
            base_total_params=base_total_params,
            sliced_active_params=sliced_active_params,
            active_param_reduction_pct=round(reduction_pct, 4),
            base_physical_size_bytes=base_bytes,
            sliced_physical_size_bytes=sliced_bytes,
            peak_ram_mb=_measure_process_ram_mb(),
            device=device,
            dtype=dtype_str,
            warmup_included=False,
            timing_scope="generation_wall_time_excluding_load_and_validation",
        )

        summary_dict = summary.model_dump()
        comparison_rows = MetricsCollector.build_paired_comparison_rows(summary_dict)

        payload = {
            "summary": summary_dict,
            "comparison_rows": comparison_rows,
            "base_samples": base_samples,
            "sliced_samples": sliced_samples,
            "validation_errors": validation_errors,
        }

        if manifest is not None:
            self.artifact_manager.save_json_artifact(
                manifest, "predictions", "evaluation/predictions.json",
                {"base_samples": base_samples, "sliced_samples": sliced_samples},
            )
            self.artifact_manager.save_json_artifact(
                manifest, "metrics", "evaluation/metrics.json",
                {"summary": summary_dict, "comparison_rows": comparison_rows},
            )
            self.artifact_manager.save_json_artifact(
                manifest, "validation_errors", "evaluation/validation_errors.json",
                validation_errors,
            )
            manifest.metrics = summary_dict
            self.artifact_manager.save_manifest(manifest)

        return payload


class BenchmarkComparator:
    """Wrapper for building standardized paired comparison tables."""

    @staticmethod
    def compare(summary_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        return MetricsCollector.build_paired_comparison_rows(summary_dict)


class BenchmarkRegistry:
    """
    Provides the Benchmark Library: combines real completed experiments from
    experiments/ with the labelled historical KT benchmark reference and
    catalog target profiles (explicitly marked as Ready to execute / Cloud / Not evaluated).
    """

    def __init__(self, artifact_manager: Optional[ArtifactManager] = None):
        self.artifact_manager = artifact_manager or ArtifactManager()

    def get_historical_reference_card(self) -> Optional[Dict[str, Any]]:
        hist_path = _PROTO_ROOT / "cache" / "benchmarks" / "latest_benchmark_results.json"
        if not hist_path.exists():
            return None
        try:
            data = json.loads(hist_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        m = data.get("metrics", {})
        return {
            "card_id": "OSSL-HIST-0001",
            "title": "Qwen 2.5 (0.5B) - Strict JSON Slice (Historical KT Artifact)",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "architecture": "qwen2",
            "operation": "structured_json_extraction",
            "slice_id": "qwen2.5-0.5b-json-slice",
            "slicing_ratio": 0.20,
            "date": "2026-historical",
            "state": CardState.VALIDATED_BENCHMARK.value,
            "is_historical_reference": True,
            "sample_count": len(data.get("base_samples", [])),
            "base_params": m.get("base_active_params", 494032768),
            "active_params": m.get("sliced_active_params", 437190528),
            "active_reduction_pct": round(m.get("param_reduction_pct", 11.51), 2),
            "normalized_valid_rate": m.get("sliced_valid_rate", 1.0),
            "strict_valid_rate": 0.0,  # Historical outputs were wrapped in ```json fences
            "schema_adherence": m.get("sliced_adherence_avg", 1.0),
            "avg_latency_s": m.get("sliced_avg_latency_s", 10.64),
            "throughput_tps": m.get("sliced_avg_throughput_tps", 15.62),
            "raw_artifact_path": str(hist_path),
            "raw_data": data,
            "notes": (
                "Historical 4-sample artifact from KT baseline. Note: raw outputs were wrapped in "
                "```json``` markdown fences (normalized parseability 4/4; strict raw parseability 0/4) "
                "and sliced latency was 10.64s vs 5.02s due to post-JSON generation tails."
            ),
        }

    def list_benchmark_cards(self) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        # 1. Real experiment manifests
        for man in self.artifact_manager.list_manifests():
            if man.status == ExperimentStatus.COMPLETED and man.metrics:
                state = CardState.VALIDATED_BENCHMARK.value
            elif man.status == ExperimentStatus.FAILED:
                state = CardState.FAILED_INCOMPLETE.value
            else:
                state = CardState.FAILED_INCOMPLETE.value

            m = man.metrics or {}
            cards.append({
                "card_id": man.experiment_id,
                "title": man.label or f"{man.model.model_id} — {man.operation}",
                "model_id": man.model.model_id,
                "architecture": man.model.architecture,
                "operation": man.operation,
                "slice_id": man.slice_id or man.experiment_id,
                "slicing_ratio": man.slicing.ratio,
                "date": man.created_at[:10],
                "state": state,
                "is_historical_reference": man.is_historical_reference,
                "sample_count": m.get("sample_count", 0),
                "base_params": m.get("base_total_params", man.model.total_params),
                "active_params": m.get("sliced_active_params"),
                "active_reduction_pct": m.get("active_param_reduction_pct"),
                "strict_valid_rate": m.get("sliced_strict_valid_rate"),
                "normalized_valid_rate": m.get("sliced_normalized_valid_rate"),
                "schema_adherence": m.get("sliced_schema_adherence"),
                "avg_latency_s": m.get("sliced_avg_latency_s"),
                "throughput_tps": m.get("sliced_avg_throughput_tps"),
                "manifest": man,
                "notes": man.error or "",
            })

        # 2. Historical reference card
        hist = self.get_historical_reference_card()
        if hist:
            cards.append(hist)

        # 3. Target catalog profiles (explicitly NOT marked as validated)
        from catalog.sliced_models import SLICED_MODELS
        from .model_pair_manager import ModelPairManager

        for s in SLICED_MODELS:
            if s["id"] == "qwen2.5-0.5b-json-slice":
                continue  # Already represented by historical/live cards
            task_id = ModelPairManager.get_task_for_slice(s["id"])
            status = s.get("status", "READY_TO_SLICE")
            state = (
                CardState.CLOUD_EXECUTION_REQUIRED.value
                if status == "CLOUD_TARGET"
                else CardState.READY_TO_EXECUTE.value
            )
            cards.append({
                "card_id": f"TARGET-{s['id']}",
                "title": s["name"],
                "model_id": s["base_model_id"],
                "architecture": "catalog_target",
                "operation": task_id,
                "slice_id": s["id"],
                "slicing_ratio": round(s.get("reduction_pct", 0.0) / 100.0, 2),
                "date": "Target Profile",
                "state": state,
                "is_historical_reference": False,
                "sample_count": 0,
                "base_params": s.get("base_params"),
                "active_params": None,
                "active_reduction_pct": None,
                "strict_valid_rate": None,
                "normalized_valid_rate": None,
                "schema_adherence": None,
                "avg_latency_s": None,
                "throughput_tps": None,
                "notes": (
                    f"Target profile only ({s.get('target_operation')}). "
                    "No completed benchmark artifact exists yet."
                ),
            })

        return cards
