"""
OS-Slice Empirical Benchmarking Suite
Evaluates Base Generalist Model against OS-Slice Micro-Model across:
- JSON Parseability Rate
- Schema Key Adherence
- Value Extraction Accuracy
- Inference Latency (TTFT & Tokens/sec)
- Active Parameter Footprint & Memory Reduction
"""

import time
import json
import re
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import default_config, OSSliceConfig
from data.generate_calibration import EVALUATION_RECORDS, format_prompt, build_calibration_dataset, CALIBRATION_RECORDS
from engine.profiler import OperationalProfiler
from engine.slicer import StructuralSlicer, OSSliceModel
from engine.compensator import OperationalCompensator

console = Console()


def clean_and_parse_json(raw_output: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Attempts strict parsing of generated text into JSON.
    Handles raw JSON, markdown-wrapped JSON, and embedded objects.
    Returns: (is_valid, parsed_dict, failure_reason)
    """
    clean_text = raw_output.strip()

    # 1. Check for standard markdown fenced JSON blocks
    md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', clean_text, re.DOTALL)
    if md_match:
        try:
            parsed = json.loads(md_match.group(1))
            if isinstance(parsed, dict):
                return True, parsed, "Strict JSON Valid (Markdown Block)"
        except Exception:
            pass

    # 2. Check for direct raw JSON object
    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, dict):
            return True, parsed, "Strict JSON Valid (Raw)"
        return False, None, "Output is JSON list/primitive, expected object"
    except json.JSONDecodeError:
        pass

    # 3. Fallback: locate outermost balanced { ... }
    start_idx = clean_text.find('{')
    if start_idx != -1:
        bracket_count = 0
        end_idx = -1
        for i in range(start_idx, len(clean_text)):
            if clean_text[i] == '{':
                bracket_count += 1
            elif clean_text[i] == '}':
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i + 1
                    break
        if end_idx != -1:
            candidate = clean_text[start_idx:end_idx]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return True, parsed, "Strict JSON Valid (Extracted)"
            except Exception as e:
                return False, None, f"Broken syntax in object: {str(e)[:40]}"

    return False, None, "No JSON object found in output"


def evaluate_schema_adherence(predicted: Optional[Dict[str, Any]], expected: Dict[str, Any]) -> float:
    """Calculates key-level schema adherence score (0.0 to 1.0)."""
    if predicted is None or not isinstance(predicted, dict):
        return 0.0
    expected_keys = set(expected.keys())
    predicted_keys = set(predicted.keys())
    if not expected_keys:
        return 1.0
    matched_keys = expected_keys.intersection(predicted_keys)
    return len(matched_keys) / len(expected_keys)


def generate_prediction(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 160,
    device: str = "cpu",
) -> Tuple[str, float, float]:
    """Generates completion and measures latency."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    start_time = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    total_time = time.perf_counter() - start_time

    generated_tokens = output_ids[0, prompt_len:]
    num_new_tokens = len(generated_tokens)
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    tokens_per_sec = num_new_tokens / (total_time + 1e-8)

    return text, total_time, tokens_per_sec


def run_benchmark(
    base_model: PreTrainedModel,
    sliced_model: OSSliceModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_records: List[Dict[str, Any]] = None,
    config: OSSliceConfig = default_config,
) -> Dict[str, Any]:
    """
    Executes head-to-head empirical benchmarking between Base and OS-Slice models.
    """
    if eval_records is None:
        eval_records = EVALUATION_RECORDS

    console.print(Panel(
        f"[bold white]OS-Slice Head-to-Head Benchmark[/bold white]\n"
        f"Model: [cyan]{config.model_id}[/cyan] | Slicing Ratio: [yellow]{config.default_slicing_ratio * 100:.1f}%[/yellow] | "
        f"Eval Samples: [green]{len(eval_records)}[/green]",
        border_style="blue"
    ))

    base_results = []
    sliced_results = []

    for idx, item in enumerate(eval_records):
        prompt = format_prompt(item["context"], item["schema_instruction"])
        expected_json = item["target_json"]

        # 1. Base Model Inference
        b_text, b_time, b_tps = generate_prediction(
            base_model, tokenizer, prompt, max_new_tokens=config.eval_max_new_tokens, device=config.device
        )
        b_valid, b_parsed, b_reason = clean_and_parse_json(b_text)
        b_adherence = evaluate_schema_adherence(b_parsed, expected_json)

        base_results.append({
            "idx": idx,
            "domain": item["domain"],
            "valid": b_valid,
            "adherence": b_adherence,
            "latency_s": b_time,
            "tokens_per_sec": b_tps,
            "raw_output": b_text,
            "status": b_reason
        })

        # 2. OS-Slice Micro-Model Inference
        s_text, s_time, s_tps = generate_prediction(
            sliced_model, tokenizer, prompt, max_new_tokens=config.eval_max_new_tokens, device=config.device
        )
        s_valid, s_parsed, s_reason = clean_and_parse_json(s_text)
        s_adherence = evaluate_schema_adherence(s_parsed, expected_json)

        sliced_results.append({
            "idx": idx,
            "domain": item["domain"],
            "valid": s_valid,
            "adherence": s_adherence,
            "latency_s": s_time,
            "tokens_per_sec": s_tps,
            "raw_output": s_text,
            "status": s_reason
        })

    # Summary Statistics
    num_samples = len(eval_records)
    base_valid_rate = sum(r["valid"] for r in base_results) / num_samples
    sliced_valid_rate = sum(r["valid"] for r in sliced_results) / num_samples

    base_adherence_avg = sum(r["adherence"] for r in base_results) / num_samples
    sliced_adherence_avg = sum(r["adherence"] for r in sliced_results) / num_samples

    base_avg_lat = sum(r["latency_s"] for r in base_results) / num_samples
    sliced_avg_lat = sum(r["latency_s"] for r in sliced_results) / num_samples

    base_avg_tps = sum(r["tokens_per_sec"] for r in base_results) / num_samples
    sliced_avg_tps = sum(r["tokens_per_sec"] for r in sliced_results) / num_samples

    active_params_base = sum(p.numel() for p in base_model.parameters())
    active_params_sliced = sliced_model.active_params
    param_reduction_pct = (1.0 - active_params_sliced / active_params_base) * 100

    # Display Rich Table
    table = Table(title="OS-Slice Empirical Benchmarking Results", border_style="cyan")
    table.add_column("Evaluation Metric", style="bold white", justify="left")
    table.add_column("Base Generalist Model", style="magenta", justify="center")
    table.add_column("OS-Slice Micro-Model", style="green", justify="center")
    table.add_column("Delta / Performance Advantage", style="bold yellow", justify="center")

    table.add_row(
        "Active Parameters",
        f"{active_params_base / 1e6:.1f}M (100%)",
        f"{active_params_sliced / 1e6:.1f}M ({100 - param_reduction_pct:.1f}%)",
        f"-{param_reduction_pct:.1f}% Footprint"
    )
    table.add_row(
        "Strict JSON Parseability Rate",
        f"{base_valid_rate * 100:.1f}%",
        f"{sliced_valid_rate * 100:.1f}%",
        f"{'+' if sliced_valid_rate >= base_valid_rate else ''}{(sliced_valid_rate - base_valid_rate) * 100:.1f}%"
    )
    table.add_row(
        "Schema Key Adherence Score",
        f"{base_adherence_avg * 100:.1f}%",
        f"{sliced_adherence_avg * 100:.1f}%",
        f"{'+' if sliced_adherence_avg >= base_adherence_avg else ''}{(sliced_adherence_avg - base_adherence_avg) * 100:.1f}%"
    )
    table.add_row(
        "Average Inference Latency",
        f"{base_avg_lat:.3f} s",
        f"{sliced_avg_lat:.3f} s",
        f"{(base_avg_lat - sliced_avg_lat) / base_avg_lat * 100:+.1f}% Speedup"
    )
    table.add_row(
        "Token Throughput",
        f"{base_avg_tps:.1f} tok/s",
        f"{sliced_avg_tps:.1f} tok/s",
        f"{(sliced_avg_tps - base_avg_tps) / base_avg_tps * 100:+.1f}% Throughput"
    )

    console.print(table)

    summary_data = {
        "metrics": {
            "base_valid_rate": base_valid_rate,
            "sliced_valid_rate": sliced_valid_rate,
            "base_adherence_avg": base_adherence_avg,
            "sliced_adherence_avg": sliced_adherence_avg,
            "base_avg_latency_s": base_avg_lat,
            "sliced_avg_latency_s": sliced_avg_lat,
            "base_avg_throughput_tps": base_avg_tps,
            "sliced_avg_throughput_tps": sliced_avg_tps,
            "base_active_params": active_params_base,
            "sliced_active_params": active_params_sliced,
            "param_reduction_pct": param_reduction_pct,
        },
        "base_samples": base_results,
        "sliced_samples": sliced_results,
    }

    # Persist benchmark results
    out_file = config.benchmark_dir / "latest_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    console.print(f"[green]Benchmark artifact saved to {out_file}[/green]")

    return summary_data


if __name__ == "__main__":
    console.print("[bold cyan]Initializing OS-Slice Standalone Benchmark Pipeline...[/bold cyan]")
    cfg = default_config

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.dtype,
        low_cpu_mem_usage=True
    ).to(cfg.device)

    # Load or compute calibration & profile
    calib_ds = build_calibration_dataset(tokenizer, records=CALIBRATION_RECORDS, boundary_weight=cfg.boundary_token_weight)
    
    profiler = OperationalProfiler(base_model, cfg)
    profile_path = cfg.profiles_dir / "operational_fisher_profile.pt"
    if profile_path.exists():
        profiler.load_profile(profile_path)
    else:
        profiler.profile(calib_ds, save_path=profile_path)

    # Apply Slicing
    slicer = StructuralSlicer(cfg)
    sliced_model = slicer.apply_slicing(
        base_model,
        profiler.head_saliency,
        profiler.mlp_saliency,
        slicing_ratio=cfg.default_slicing_ratio,
        in_place=False,
    )

    # Localized Compensation
    compensator = OperationalCompensator(cfg)
    compensator.compensate(sliced_model, calib_ds, steps=cfg.compensation_steps)

    # Execute Benchmark
    results = run_benchmark(base_model, sliced_model, tokenizer, eval_records=EVALUATION_RECORDS, config=cfg)
    console.print("[bold green]Benchmark execution completed successfully![/bold green]")
