"""
OS-Slice Unified Experiment Manager
Orchestrates the 4-stage OS-Slice pipeline, evaluation, ablation studies,
and multi-ratio compression sweeps through a single shared service layer
for both CLI Workbench and Research Lab UI.
"""

import copy
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from .artifact_manager import ArtifactManager, build_profile_cache_key
from .benchmark_runner import BenchmarkRunner
from .event_logger import EventLogger
from .evidence_exporter import EvidenceExporter
from .experiment_schema import (
    AblationRunRecord,
    CompensationConfig,
    CompressionSweepPoint,
    DatasetRef,
    ExperimentManifest,
    ExperimentStatus,
    ModelIdentity,
    RuntimeInfo,
    SlicingConfig,
    StageName,
    StageRecord,
    StageStatus,
    utc_now_iso,
)

_PROTO_ROOT = Path(__file__).resolve().parent.parent
if str(_PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTO_ROOT))

from data.generate_calibration import (
    CALIBRATION_RECORDS,
    EVALUATION_RECORDS,
    build_calibration_dataset,
    compute_dataset_hash,
    inspect_calibration_record,
)
from engine.profiler import OperationalProfiler
from engine.slicer import StructuralSlicer
from engine.compensator import OperationalCompensator


def _get_software_versions() -> Dict[str, str]:
    import transformers
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def _extract_model_identity(model: Any, model_id: str, tokenizer_id: str = "") -> ModelIdentity:
    cfg = model.config
    total_params = sum(p.numel() for p in model.parameters())
    physical_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    dtype_str = str(next(model.parameters()).dtype).replace("torch.", "")
    return ModelIdentity(
        model_id=model_id,
        revision=getattr(cfg, "_commit_hash", None) or "main",
        architecture=getattr(cfg, "model_type", "unknown"),
        tokenizer_id=tokenizer_id or model_id,
        dtype=dtype_str,
        num_layers=getattr(cfg, "num_hidden_layers", None),
        num_heads=getattr(cfg, "num_attention_heads", None),
        num_kv_heads=getattr(cfg, "num_key_value_heads", None),
        hidden_size=getattr(cfg, "hidden_size", None),
        intermediate_size=getattr(cfg, "intermediate_size", None),
        total_params=total_params,
        physical_size_bytes=physical_bytes,
    )


class ExperimentManager:
    """Central lifecycle orchestrator for OS-Slice experiments."""

    def __init__(self, artifact_manager: Optional[ArtifactManager] = None):
        self.artifact_manager = artifact_manager or ArtifactManager()
        self.benchmark_runner = BenchmarkRunner(self.artifact_manager)
        self.evidence_exporter = EvidenceExporter(self.artifact_manager)
        self.profile_cache_dir = _PROTO_ROOT / "cache" / "profiles"
        self.profile_cache_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        model: Any,
        model_id: str,
        operation: str = "structured_json_extraction",
        slicing_ratio: float = 0.20,
        boundary_weight: float = 6.0,
        content_weight: float = 1.0,
        compensation_enabled: bool = True,
        ridge_lambda: float = 1e-3,
        device: str = "cpu",
        seed: int = 123,
        source_interface: str = "research_lab",
        label: str = "",
        calibration_records: Optional[List[Dict[str, Any]]] = None,
        eval_records: Optional[List[Dict[str, Any]]] = None,
    ) -> ExperimentManifest:
        calib_recs = calibration_records if calibration_records is not None else CALIBRATION_RECORDS
        eval_recs = eval_records if eval_records is not None else EVALUATION_RECORDS

        exp_id = self.artifact_manager.generate_experiment_id()
        model_ident = _extract_model_identity(model, model_id)
        num_layers = model_ident.num_layers or 0
        protected = [0, 1, num_layers - 2, num_layers - 1] if num_layers >= 4 else list(range(num_layers))

        manifest = ExperimentManifest(
            experiment_id=exp_id,
            status=ExperimentStatus.CREATED,
            label=label or f"{model_id} ({operation}, {slicing_ratio * 100:.0f}%)",
            model=model_ident,
            operation=operation,
            slice_id=f"{model_id.split('/')[-1].lower()}-{operation}-r{int(slicing_ratio * 100)}",
            calibration=DatasetRef(
                dataset_id="enterprise_json_calibration",
                dataset_version="1.0.0",
                dataset_hash=compute_dataset_hash(calib_recs),
                sample_count=len(calib_recs),
                sample_ids=[f"cal-{i:02d}" for i in range(len(calib_recs))],
            ),
            evaluation=DatasetRef(
                dataset_id="enterprise_json_evaluation",
                dataset_version="1.1.0",
                dataset_hash=compute_dataset_hash(eval_recs),
                sample_count=len(eval_recs),
                sample_ids=[f"eval-{i:02d}" for i in range(len(eval_recs))],
            ),
            slicing=SlicingConfig(ratio=slicing_ratio, protected_layers=protected),
            compensation=CompensationConfig(
                enabled=compensation_enabled,
                method="diagonal_ridge_damping" if compensation_enabled else "none",
                ridge_lambda=ridge_lambda,
            ),
            runtime=RuntimeInfo(
                device=device,
                software_versions=_get_software_versions(),
                seed=seed,
                source_interface=source_interface,
            ),
        )
        self.artifact_manager.save_manifest(manifest)
        return manifest

    def run_full_experiment(
        self,
        base_model: Any,
        tokenizer: Any,
        config: Any,
        slicing_ratio: Optional[float] = None,
        compensation_enabled: bool = True,
        ridge_lambda: float = 1e-3,
        calibration_records: Optional[List[Dict[str, Any]]] = None,
        eval_records: Optional[List[Dict[str, Any]]] = None,
        source_interface: str = "research_lab",
        label: str = "",
        force_reprofile: bool = False,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Executes all 4 OS-Slice stages + evaluation and persists every artifact
        and event under experiments/<experiment_id>/.
        """
        ratio = slicing_ratio if slicing_ratio is not None else config.default_slicing_ratio
        calib_recs = calibration_records if calibration_records is not None else CALIBRATION_RECORDS
        eval_recs = eval_records if eval_records is not None else EVALUATION_RECORDS

        manifest = self.create_experiment(
            model=base_model,
            model_id=config.model_id,
            operation="structured_json_extraction",
            slicing_ratio=ratio,
            boundary_weight=config.boundary_token_weight,
            content_weight=config.content_token_weight,
            compensation_enabled=compensation_enabled,
            ridge_lambda=ridge_lambda,
            device=config.device,
            source_interface=source_interface,
            label=label,
            calibration_records=calib_recs,
            eval_records=eval_recs,
        )

        exp_dir = self.artifact_manager.get_experiment_dir(manifest.experiment_id)
        logger = EventLogger(manifest.experiment_id, exp_dir / "execution_events.jsonl")
        manifest.status = ExperimentStatus.RUNNING
        self.artifact_manager.save_manifest(manifest)

        # Save model info artifact
        self.artifact_manager.save_json_artifact(
            manifest, "model_info", "model/model_info.json", manifest.model.model_dump()
        )

        current_stage = StageName.CALIBRATION.value
        try:
            # --- Stage 1: Calibration & Boundary Token Weighting ---
            current_stage = StageName.CALIBRATION.value
            t0 = time.perf_counter()
            logger.stage_started(current_stage, {"sample_count": len(calib_recs)})
            manifest.stages[current_stage] = StageRecord(stage=StageName.CALIBRATION, status=StageStatus.RUNNING, started_at=utc_now_iso())

            calib_ds = build_calibration_dataset(
                tokenizer,
                records=calib_recs,
                max_length=config.max_seq_len,
                boundary_weight=config.boundary_token_weight,
                content_weight=config.content_token_weight,
            )
            token_inspections = [
                inspect_calibration_record(
                    tokenizer, rec, boundary_weight=config.boundary_token_weight,
                    content_weight=config.content_token_weight, max_length=config.max_seq_len
                )
                for rec in calib_recs
            ]
            calib_meta = {
                "dataset_id": manifest.calibration.dataset_id,
                "dataset_hash": manifest.calibration.dataset_hash,
                "sample_count": len(calib_ds),
                "boundary_token_weight": config.boundary_token_weight,
                "numeric_token_weight": round(config.boundary_token_weight * 0.7, 4),
                "content_token_weight": config.content_token_weight,
                "prompt_label_id": -100,
                "prompt_weight": 0.0,
                "inspections": token_inspections,
            }
            cal_ref = self.artifact_manager.save_json_artifact(
                manifest, "calibration_metadata", "calibration/calibration_metadata.json", calib_meta
            )
            dur = time.perf_counter() - t0
            manifest.stages[current_stage] = StageRecord(
                stage=StageName.CALIBRATION, status=StageStatus.COMPLETED,
                completed_at=utc_now_iso(), duration_s=round(dur, 4),
                summary={"sample_count": len(calib_ds), "dataset_hash": manifest.calibration.dataset_hash},
            )
            logger.stage_completed(current_stage, dur, artifacts={"calibration_metadata": cal_ref.sha256})
            if progress_callback:
                progress_callback("calibration", 1.0)

            # --- Stage 2: Empirical Fisher Saliency Profiling ---
            current_stage = StageName.PROFILING.value
            t0 = time.perf_counter()
            cache_key = build_profile_cache_key(
                model_id=manifest.model.model_id,
                revision=manifest.model.revision,
                architecture=manifest.model.architecture,
                operation=manifest.operation,
                calibration_hash=manifest.calibration.dataset_hash,
                boundary_weight=config.boundary_token_weight,
                content_weight=config.content_token_weight,
            )
            cached_profile_path = self.profile_cache_dir / f"{cache_key}.pt"
            logger.stage_started(current_stage, {"cache_key": cache_key, "force_reprofile": force_reprofile})
            manifest.stages[current_stage] = StageRecord(stage=StageName.PROFILING, status=StageStatus.RUNNING, started_at=utc_now_iso())

            profiler = OperationalProfiler(base_model, config)
            if cached_profile_path.exists() and not force_reprofile:
                profiler.load_profile(cached_profile_path)
                profile_source = "cache_hit_exact_key"
            else:
                def _prof_cb(step: int, total: int, loss_val: float):
                    logger.progress(current_stage, {"sample": step, "total": total, "operational_loss": round(loss_val, 6)})
                    if progress_callback:
                        progress_callback("profiling", step / max(total, 1))
                profiler.profile(calib_ds, save_path=cached_profile_path, progress_callback=_prof_cb)
                profile_source = "computed_fresh"

            prof_tensor_ref = self.artifact_manager.save_tensor_artifact(
                manifest, "fisher_profile", "profiles/fisher_profile.pt",
                {
                    "head_saliency": profiler.head_saliency,
                    "mlp_saliency": profiler.mlp_saliency,
                    "metadata": profiler.profile_metadata,
                    "cache_key": cache_key,
                },
            )
            prof_summary = {
                "cache_key": cache_key,
                "profile_source": profile_source,
                "head_saliency_shape": list(profiler.head_saliency.shape),
                "mlp_saliency_shape": list(profiler.mlp_saliency.shape),
                "metadata": profiler.profile_metadata,
            }
            prof_sum_ref = self.artifact_manager.save_json_artifact(
                manifest, "profile_summary", "profiles/profile_summary.json", prof_summary
            )
            dur = time.perf_counter() - t0
            manifest.stages[current_stage] = StageRecord(
                stage=StageName.PROFILING, status=StageStatus.COMPLETED,
                completed_at=utc_now_iso(), duration_s=round(dur, 4),
                summary={"profile_source": profile_source, "cache_key": cache_key},
            )
            logger.stage_completed(current_stage, dur, artifacts={"fisher_profile": prof_tensor_ref.sha256, "profile_summary": prof_sum_ref.sha256})

            # --- Stage 3: Layer-Aware Structural Slicing ---
            current_stage = StageName.SLICING.value
            t0 = time.perf_counter()
            logger.stage_started(current_stage, {"slicing_ratio": ratio})
            manifest.stages[current_stage] = StageRecord(stage=StageName.SLICING, status=StageStatus.RUNNING, started_at=utc_now_iso())

            slicer = StructuralSlicer(config)
            sliced_model = slicer.apply_slicing(
                base_model,
                profiler.head_saliency,
                profiler.mlp_saliency,
                slicing_ratio=ratio,
                in_place=False,
            )
            slicing_report = slicer.build_slicing_report(sliced_model, base_model)
            hm_ref = self.artifact_manager.save_tensor_artifact(
                manifest, "attention_masks", "slicing/attention_masks.pt", sliced_model.head_mask
            )
            mm_ref = self.artifact_manager.save_tensor_artifact(
                manifest, "mlp_masks", "slicing/mlp_masks.pt", sliced_model.mlp_mask
            )
            sr_ref = self.artifact_manager.save_json_artifact(
                manifest, "slicing_report", "slicing/slicing_report.json", slicing_report
            )
            dur = time.perf_counter() - t0
            manifest.stages[current_stage] = StageRecord(
                stage=StageName.SLICING, status=StageStatus.COMPLETED,
                completed_at=utc_now_iso(), duration_s=round(dur, 4),
                summary={
                    "active_params": sliced_model.active_params,
                    "total_params": sliced_model.total_params,
                    "active_reduction_pct": slicing_report["effective_active_sparsity_pct"],
                },
            )
            logger.stage_completed(current_stage, dur, artifacts={"attention_masks": hm_ref.sha256, "mlp_masks": mm_ref.sha256, "slicing_report": sr_ref.sha256})
            if progress_callback:
                progress_callback("slicing", 1.0)

            # --- Stage 4: Localized Error Compensation ---
            current_stage = StageName.COMPENSATION.value
            t0 = time.perf_counter()
            logger.stage_started(current_stage, {"enabled": compensation_enabled, "ridge_lambda": ridge_lambda})
            manifest.stages[current_stage] = StageRecord(stage=StageName.COMPENSATION, status=StageStatus.RUNNING, started_at=utc_now_iso())

            if compensation_enabled:
                compensator = OperationalCompensator(config)
                comp_report = compensator.compensate(sliced_model, calib_ds, ridge_lambda=ridge_lambda)
            else:
                comp_report = {
                    "method": "none",
                    "method_label": "Compensation disabled (ablation / uncompensated slice)",
                    "ridge_lambda": 0.0,
                    "compensated_layers": 0,
                    "layer_reports": [],
                    "status": "Skipped",
                }
            comp_ref = self.artifact_manager.save_json_artifact(
                manifest, "compensation_report", "compensation/compensation_report.json", comp_report
            )
            dur = time.perf_counter() - t0
            manifest.stages[current_stage] = StageRecord(
                stage=StageName.COMPENSATION,
                status=StageStatus.COMPLETED if compensation_enabled else StageStatus.SKIPPED,
                completed_at=utc_now_iso(), duration_s=round(dur, 4),
                summary={"method": comp_report["method"], "compensated_layers": comp_report["compensated_layers"]},
            )
            logger.stage_completed(current_stage, dur, artifacts={"compensation_report": comp_ref.sha256})
            if progress_callback:
                progress_callback("compensation", 1.0)

            # --- Stage 5: Paired Evaluation ---
            current_stage = StageName.EVALUATION.value
            t0 = time.perf_counter()
            logger.stage_started(current_stage, {"eval_samples": len(eval_recs)})
            manifest.stages[current_stage] = StageRecord(stage=StageName.EVALUATION, status=StageStatus.RUNNING, started_at=utc_now_iso())

            def _eval_cb(step: int, total: int):
                logger.progress(current_stage, {"sample": step, "total": total})
                if progress_callback:
                    progress_callback("evaluation", step / max(total, 1))

            bench_payload = self.benchmark_runner.run(
                base_model=base_model,
                sliced_model=sliced_model,
                tokenizer=tokenizer,
                eval_records=eval_recs,
                model_id=config.model_id,
                slice_id=manifest.slice_id,
                slicing_ratio=ratio,
                task_id=manifest.operation,
                max_new_tokens=config.eval_max_new_tokens,
                temperature=config.eval_temperature,
                seed=manifest.runtime.seed,
                device=config.device,
                dtype_str=manifest.model.dtype,
                manifest=manifest,
                warmup=True,
                progress_callback=_eval_cb,
            )
            dur = time.perf_counter() - t0
            manifest.stages[current_stage] = StageRecord(
                stage=StageName.EVALUATION, status=StageStatus.COMPLETED,
                completed_at=utc_now_iso(), duration_s=round(dur, 4),
                summary=bench_payload["summary"],
            )
            logger.stage_completed(current_stage, dur)

            manifest.status = ExperimentStatus.COMPLETED
            self.artifact_manager.save_manifest(manifest)
            report_paths = self.evidence_exporter.export(manifest)

            return {
                "manifest": manifest,
                "sliced_model": sliced_model,
                "profiler": profiler,
                "slicing_report": slicing_report,
                "compensation_report": comp_report,
                "benchmark": bench_payload,
                "report_paths": report_paths,
            }

        except Exception as exc:
            manifest.status = ExperimentStatus.FAILED
            manifest.error = str(exc)
            manifest.stages[current_stage] = StageRecord(
                stage=StageName(current_stage),
                status=StageStatus.FAILED,
                completed_at=utc_now_iso(),
                error=str(exc),
            )
            logger.stage_failed(current_stage, str(exc))
            self.artifact_manager.save_manifest(manifest)
            raise

    @staticmethod
    def _apply_magnitude_pruning_baseline(
        base_model: Any,
        slicing_ratio: float,
        config: Any,
    ) -> Any:
        """
        E2 Conventional Pruning Baseline:
        Layer-wise L1 weight magnitude pruning of attention heads and MLP channels
        at the exact same slicing_ratio and protected outer layers (0, 1, L-2, L-1),
        without Fisher operational saliency or boundary token weighting.
        """
        num_layers = base_model.config.num_hidden_layers
        num_heads = base_model.config.num_attention_heads
        hidden_size = base_model.config.hidden_size
        head_dim = hidden_size // num_heads
        intermediate_size = base_model.config.intermediate_size

        head_scores = torch.zeros((num_layers, num_heads), dtype=torch.float32)
        mlp_scores = torch.zeros((num_layers, intermediate_size), dtype=torch.float32)

        layers = None
        for attr in ["model.layers", "transformer.h", "layers"]:
            curr = base_model
            ok = True
            for part in attr.split("."):
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                else:
                    ok = False
                    break
            if ok:
                layers = curr
                break

        with torch.no_grad():
            for l_idx, layer in enumerate(layers):
                attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
                if attn is not None:
                    q_mag = attn.q_proj.weight.data.float().abs().view(num_heads, head_dim, -1).sum(dim=(1, 2)).cpu()
                    o_mag = attn.o_proj.weight.data.float().abs().view(-1, num_heads, head_dim).sum(dim=(0, 2)).cpu()
                    head_scores[l_idx] = q_mag + o_mag
                mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
                if mlp is not None:
                    g_mag = mlp.gate_proj.weight.data.float().abs().sum(dim=1).cpu()
                    d_mag = mlp.down_proj.weight.data.float().abs().sum(dim=0).cpu()
                    mlp_scores[l_idx] = g_mag + d_mag

        slicer = StructuralSlicer(config)
        return slicer.apply_slicing(base_model, head_scores, mlp_scores, slicing_ratio=slicing_ratio, in_place=False)

    def run_ablation_study(
        self,
        base_model: Any,
        tokenizer: Any,
        config: Any,
        slicing_ratio: float = 0.20,
        eval_records: Optional[List[Dict[str, Any]]] = None,
        calibration_records: Optional[List[Dict[str, Any]]] = None,
        parent_manifest: Optional[ExperimentManifest] = None,
    ) -> List[AblationRunRecord]:
        """
        Runs controlled ablation configurations E0, E2, E3, E4 on identical inputs:
        - E0: Original base model
        - E2: Conventional L1 magnitude pruning baseline (same ratio & protected layers)
        - E3: Behavioral Fisher slicing without compensation
        - E4: Behavioral Fisher slicing with diagonal ridge-inspired compensation
        """
        calib_recs = calibration_records if calibration_records is not None else CALIBRATION_RECORDS
        eval_recs = eval_records if eval_records is not None else EVALUATION_RECORDS

        calib_ds = build_calibration_dataset(
            tokenizer, records=calib_recs, max_length=config.max_seq_len,
            boundary_weight=config.boundary_token_weight, content_weight=config.content_token_weight,
        )
        profiler = OperationalProfiler(base_model, config)
        profiler.profile(calib_ds)

        slicer = StructuralSlicer(config)
        compensator = OperationalCompensator(config)
        total_params = sum(p.numel() for p in base_model.parameters())

        configs = [
            ("E0", "Original Base Model", "What is the unmodified baseline?", copy.deepcopy(base_model)),
            ("E2", "Conventional L1 Magnitude Pruning", "How does L1 magnitude pruning at matched budget compare?",
             self._apply_magnitude_pruning_baseline(base_model, slicing_ratio, config)),
            ("E3", "Behavioral Slicing (No Compensation)", "What does OS-Slice's Fisher saliency selection contribute?",
             slicer.apply_slicing(base_model, profiler.head_saliency, profiler.mlp_saliency, slicing_ratio=slicing_ratio, in_place=False)),
        ]
        e4_model = slicer.apply_slicing(base_model, profiler.head_saliency, profiler.mlp_saliency, slicing_ratio=slicing_ratio, in_place=False)
        compensator.compensate(e4_model, calib_ds)
        configs.append(("E4", "Behavioral Slicing + Compensation", "Does diagonal ridge compensation improve or preserve behavior?", e4_model))

        results: List[AblationRunRecord] = []
        for cid, cname, rq, cand_model in configs:
            bench = self.benchmark_runner.run(
                base_model=base_model,
                sliced_model=cand_model,
                tokenizer=tokenizer,
                eval_records=eval_recs,
                model_id=config.model_id,
                slice_id=f"ablation-{cid}",
                slicing_ratio=0.0 if cid == "E0" else slicing_ratio,
                max_new_tokens=config.eval_max_new_tokens,
                temperature=config.eval_temperature,
                device=config.device,
                warmup=False,
            )
            sm = bench["summary"]
            active_p = int(getattr(cand_model, "active_params", total_params))
            results.append(
                AblationRunRecord(
                    config_id=cid,
                    config_name=cname,
                    research_question=rq,
                    experiment_id=parent_manifest.experiment_id if parent_manifest else None,
                    status="completed",
                    active_params=active_p,
                    total_params=total_params,
                    active_reduction_pct=round((1.0 - active_p / max(total_params, 1)) * 100.0, 2),
                    strict_valid_rate=sm["sliced_strict_valid_rate"],
                    normalized_valid_rate=sm["sliced_normalized_valid_rate"],
                    schema_adherence=sm["sliced_schema_adherence"],
                    avg_latency_s=sm["sliced_avg_latency_s"],
                    throughput_tps=sm["sliced_avg_throughput_tps"],
                )
            )

        if parent_manifest is not None:
            self.artifact_manager.save_json_artifact(
                parent_manifest, "ablation_results", "evaluation/ablation_results.json",
                [r.model_dump() for r in results],
            )

        return results

    def run_compression_sweep(
        self,
        base_model: Any,
        tokenizer: Any,
        config: Any,
        ratios: Optional[List[float]] = None,
        eval_records: Optional[List[Dict[str, Any]]] = None,
        calibration_records: Optional[List[Dict[str, Any]]] = None,
        parent_manifest: Optional[ExperimentManifest] = None,
    ) -> List[CompressionSweepPoint]:
        """Runs a multi-ratio slicing sweep and computes measured Pareto frontier."""
        if ratios is None:
            ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
        calib_recs = calibration_records if calibration_records is not None else CALIBRATION_RECORDS
        eval_recs = eval_records if eval_records is not None else EVALUATION_RECORDS

        calib_ds = build_calibration_dataset(
            tokenizer, records=calib_recs, max_length=config.max_seq_len,
            boundary_weight=config.boundary_token_weight, content_weight=config.content_token_weight,
        )
        profiler = OperationalProfiler(base_model, config)
        profiler.profile(calib_ds)

        slicer = StructuralSlicer(config)
        compensator = OperationalCompensator(config)
        total_params = sum(p.numel() for p in base_model.parameters())

        points: List[CompressionSweepPoint] = []
        for r in ratios:
            try:
                sm_model = slicer.apply_slicing(
                    base_model, profiler.head_saliency, profiler.mlp_saliency, slicing_ratio=r, in_place=False
                )
                compensator.compensate(sm_model, calib_ds)
                bench = self.benchmark_runner.run(
                    base_model=base_model,
                    sliced_model=sm_model,
                    tokenizer=tokenizer,
                    eval_records=eval_recs,
                    model_id=config.model_id,
                    slice_id=f"sweep-r{int(r * 100)}",
                    slicing_ratio=r,
                    max_new_tokens=config.eval_max_new_tokens,
                    temperature=config.eval_temperature,
                    device=config.device,
                    warmup=False,
                )
                sm = bench["summary"]
                points.append(
                    CompressionSweepPoint(
                        target_ratio=r,
                        experiment_id=parent_manifest.experiment_id if parent_manifest else None,
                        status="completed",
                        feasible=True,
                        active_params=sm["sliced_active_params"],
                        total_params=total_params,
                        active_reduction_pct=sm["active_param_reduction_pct"],
                        strict_valid_rate=sm["sliced_strict_valid_rate"],
                        normalized_valid_rate=sm["sliced_normalized_valid_rate"],
                        schema_adherence=sm["sliced_schema_adherence"],
                        avg_latency_s=sm["sliced_avg_latency_s"],
                        throughput_tps=sm["sliced_avg_throughput_tps"],
                        physical_size_bytes=sm.get("sliced_physical_size_bytes"),
                        peak_ram_mb=sm.get("peak_ram_mb"),
                    )
                )
            except Exception as exc:
                points.append(
                    CompressionSweepPoint(
                        target_ratio=r,
                        experiment_id=parent_manifest.experiment_id if parent_manifest else None,
                        status="failed",
                        feasible=False,
                        active_params=total_params,
                        total_params=total_params,
                        active_reduction_pct=0.0,
                        strict_valid_rate=0.0,
                        normalized_valid_rate=0.0,
                        schema_adherence=0.0,
                        avg_latency_s=0.0,
                        throughput_tps=0.0,
                        error=str(exc),
                    )
                )

        # Compute Pareto-efficient points (maximize active_reduction_pct and schema_adherence)
        feasible_pts = [p for p in points if p.feasible]
        for p in feasible_pts:
            dominated = any(
                (other.active_reduction_pct >= p.active_reduction_pct and other.schema_adherence >= p.schema_adherence)
                and (other.active_reduction_pct > p.active_reduction_pct or other.schema_adherence > p.schema_adherence)
                for other in feasible_pts
            )
            p.is_pareto_efficient = not dominated

        if parent_manifest is not None:
            self.artifact_manager.save_json_artifact(
                parent_manifest, "compression_sweep", "evaluation/compression_sweep.json",
                [p.model_dump() for p in points],
            )

        return points
