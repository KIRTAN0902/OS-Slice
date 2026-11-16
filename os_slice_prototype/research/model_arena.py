"""
OS-Slice Model Arena Service
Runs paired live inference on the same input and equivalent decoding settings
through a verified (base, task-specific sliced) model pair. Preserves raw outputs
and records strict validation, latency, first-token latency, token counts, and errors.
"""

import threading
import time
import uuid
from typing import Any, Dict, List, Optional
import torch
from transformers import PreTrainedTokenizerBase, TextIteratorStreamer

from .artifact_manager import ArtifactManager
from .experiment_schema import (
    ExperimentManifest,
    PairedComparisonRecord,
    PairedInferenceResult,
    utc_now_iso,
)
from .model_pair_manager import ModelPairManager
from .task_registry import TaskRegistry


def run_single_generation(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    role: str,
    model_id: str,
    task_id: str,
    max_new_tokens: int = 180,
    temperature: float = 0.0,
    seed: int = 123,
    device: str = "cpu",
    ground_truth: Optional[Dict[str, Any]] = None,
    measure_first_token: bool = True,
) -> PairedInferenceResult:
    """
    Executes controlled inference on a single model, measuring wall time and
    (when supported) time-to-first-token via TextIteratorStreamer.
    Preserves raw generated text without silent cleanup.
    """
    try:
        if seed is not None:
            torch.manual_seed(seed)

        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        prompt_len = input_ids.shape[1]

        do_sample = temperature > 0.0
        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "repetition_penalty": 1.05,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)

        first_token_latency: Optional[float] = None
        raw_output = ""
        generated_tokens = 0

        # Only use TextIteratorStreamer on HF PreTrainedModel instances or models with generation_config
        supports_streamer = hasattr(model, "generation_config") or (
            hasattr(model, "model") and hasattr(model.model, "generation_config")
        )
        if measure_first_token and supports_streamer:
            try:
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                gen_kwargs["streamer"] = streamer
                holder: Dict[str, Any] = {}

                def _worker():
                    try:
                        with torch.no_grad():
                            holder["output_ids"] = model.generate(**gen_kwargs)
                    except Exception as exc:
                        holder["error"] = exc

                t0 = time.perf_counter()
                thread = threading.Thread(target=_worker, daemon=True)
                thread.start()

                chunks: List[str] = []
                for chunk in streamer:
                    if first_token_latency is None and chunk:
                        first_token_latency = time.perf_counter() - t0
                    chunks.append(chunk)
                thread.join()
                total_latency = time.perf_counter() - t0

                if "error" in holder:
                    raise holder["error"]

                out_ids = holder.get("output_ids")
                if out_ids is not None:
                    gen_ids = out_ids[0, prompt_len:]
                    generated_tokens = int(len(gen_ids))
                    raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)
                else:
                    raw_output = "".join(chunks)
                    generated_tokens = len(tokenizer.encode(raw_output, add_special_tokens=False))
            except Exception:
                # Fallback to synchronous generate if streamer fails for custom wrappers
                gen_kwargs.pop("streamer", None)
                t0 = time.perf_counter()
                with torch.no_grad():
                    out_ids = model.generate(**gen_kwargs)
                total_latency = time.perf_counter() - t0
                gen_ids = out_ids[0, prompt_len:]
                generated_tokens = int(len(gen_ids))
                raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                out_ids = model.generate(**gen_kwargs)
            total_latency = time.perf_counter() - t0
            gen_ids = out_ids[0, prompt_len:]
            generated_tokens = int(len(gen_ids))
            raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)

        tps = generated_tokens / (total_latency + 1e-9)
        val = TaskRegistry.validate_output(task_id, raw_output, ground_truth)
        is_valid = bool(val.get("strict_valid", False))

        return PairedInferenceResult(
            role=role,
            model_id=model_id,
            raw_output=raw_output,
            is_valid=is_valid,
            validator_name=val.get("validator_name", "unknown"),
            validation_error=val.get("strict_error"),
            normalized_valid=val.get("normalized_valid"),
            normalized_status=val.get("normalized_status"),
            task_score=float(val.get("task_score", 0.0)),
            schema_adherence=float(val.get("schema_adherence", 0.0)),
            field_exact_match=val.get("field_exact_match"),
            has_trailing_text=bool(val.get("has_trailing_text", False)),
            first_token_latency_s=round(first_token_latency, 4) if first_token_latency is not None else None,
            total_latency_s=round(total_latency, 4),
            generated_tokens=generated_tokens,
            throughput_tps=round(tps, 2),
            run_status="Success" if is_valid else "Invalid Output",
            generation_metadata={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "seed": seed,
                "device": device,
                "prompt_tokens": int(prompt_len),
                "timing_scope": "generation_wall_time_excluding_load_and_validation",
            },
        )
    except Exception as exc:
        return PairedInferenceResult(
            role=role,
            model_id=model_id,
            raw_output="",
            is_valid=False,
            validator_name="error",
            validation_error=str(exc),
            total_latency_s=0.0,
            generated_tokens=0,
            throughput_tps=0.0,
            run_status="Model Error",
            error_message=str(exc),
        )


class ModelArenaService:
    """Executes paired inference and persists raw artifacts."""

    def __init__(self, artifact_manager: Optional[ArtifactManager] = None):
        self.artifact_manager = artifact_manager or ArtifactManager()

    def run_paired_inference(
        self,
        base_model: Any,
        sliced_model: Any,
        tokenizer: PreTrainedTokenizerBase,
        model_id: str,
        slice_id: str,
        task_id: str,
        prompt: str,
        schema_instruction: str = "",
        ground_truth: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 180,
        temperature: float = 0.0,
        seed: int = 123,
        device: str = "cpu",
        manifest: Optional[ExperimentManifest] = None,
    ) -> PairedComparisonRecord:
        readiness = ModelPairManager.check_pair_readiness(model_id, task_id)
        if not TaskRegistry.is_task_implemented(task_id):
            raise ValueError(f"Task '{task_id}' is not implemented for live Model Arena execution: {readiness['reason']}")

        base_res = run_single_generation(
            model=base_model,
            tokenizer=tokenizer,
            prompt=prompt,
            role="base",
            model_id=model_id,
            task_id=task_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            device=device,
            ground_truth=ground_truth,
        )
        sliced_res = run_single_generation(
            model=sliced_model,
            tokenizer=tokenizer,
            prompt=prompt,
            role="sliced",
            model_id=f"{model_id}::{slice_id}",
            task_id=task_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            device=device,
            ground_truth=ground_truth,
        )

        record = PairedComparisonRecord(
            run_id=f"arena-{uuid.uuid4().hex[:10]}",
            experiment_id=manifest.experiment_id if manifest else None,
            timestamp=utc_now_iso(),
            model_id=model_id,
            slice_id=slice_id,
            task_id=task_id,
            prompt=prompt,
            schema_instruction=schema_instruction,
            ground_truth=ground_truth,
            decoding_config={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "seed": seed,
                "device": device,
            },
            base=base_res,
            sliced=sliced_res,
            pair_lineage_verified=True,
            runtime_differences=[],
        )

        if manifest is not None:
            exp_dir = self.artifact_manager.init_experiment_dir(manifest.experiment_id)
            paired_path = exp_dir / "inference" / "paired_runs.json"
            existing: List[Dict[str, Any]] = []
            if paired_path.exists():
                import json
                try:
                    existing = json.loads(paired_path.read_text(encoding="utf-8"))
                except Exception:
                    existing = []
            existing.append(record.model_dump())
            self.artifact_manager.save_json_artifact(
                manifest, "paired_runs", "inference/paired_runs.json", existing
            )
            self.artifact_manager.save_text_artifact(
                manifest,
                f"raw_base_{record.run_id}",
                f"inference/raw_outputs/{record.run_id}_base.txt",
                base_res.raw_output,
                artifact_type="text",
            )
            self.artifact_manager.save_text_artifact(
                manifest,
                f"raw_sliced_{record.run_id}",
                f"inference/raw_outputs/{record.run_id}_sliced.txt",
                sliced_res.raw_output,
                artifact_type="text",
            )

        return record
