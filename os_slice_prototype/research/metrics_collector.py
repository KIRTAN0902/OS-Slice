"""
OS-Slice Metrics Collector
Computes paired comparison rows with explicit metric definitions, absolute deltas,
relative change percentages (lower-is-better vs higher-is-better), direction,
sample counts, and hardware/measurement conditions.
"""

from typing import Any, Dict, List, Optional


def compute_relative_change(
    base_val: Optional[float],
    sliced_val: Optional[float],
    higher_is_better: bool,
) -> Optional[float]:
    """
    Implements Section 9 relative change formulas:
    - lower-is-better: (Base - Sliced) / Base * 100
    - higher-is-better: (Sliced - Base) / Base * 100
    Preserves negative values when sliced degrades.
    """
    if base_val is None or sliced_val is None or base_val == 0:
        return None
    if higher_is_better:
        return ((sliced_val - base_val) / base_val) * 100.0
    return ((base_val - sliced_val) / base_val) * 100.0


class MetricsCollector:
    """Produces standardized paired comparison tables from benchmark summaries."""

    @staticmethod
    def build_paired_comparison_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        sample_count = summary.get("sample_count", 0)
        device = summary.get("device", "unknown")
        dtype = summary.get("dtype", "unknown")
        warmup = summary.get("warmup_included", False)
        timing_scope = summary.get("timing_scope", "generation_wall_time")
        conditions = f"n={sample_count}, device={device}, dtype={dtype}, warmup={warmup}, scope={timing_scope}"

        specs = [
            {
                "metric": "Strict JSON Parseability",
                "definition": "Raw output parses via json.loads without fence stripping or cleanup",
                "base": summary.get("base_strict_valid_rate"),
                "sliced": summary.get("sliced_strict_valid_rate"),
                "unit": "ratio",
                "higher_is_better": True,
            },
            {
                "metric": "Normalized JSON Parseability",
                "definition": "Output parses after markdown fence stripping / object extraction (secondary diagnostic)",
                "base": summary.get("base_normalized_valid_rate"),
                "sliced": summary.get("sliced_normalized_valid_rate"),
                "unit": "ratio",
                "higher_is_better": True,
            },
            {
                "metric": "Schema Key Adherence",
                "definition": "Fraction of required schema keys present in parsed JSON output",
                "base": summary.get("base_schema_adherence"),
                "sliced": summary.get("sliced_schema_adherence"),
                "unit": "ratio",
                "higher_is_better": True,
            },
            {
                "metric": "Field Exact Match",
                "definition": "Fraction of ground-truth key/value fields matching exactly",
                "base": summary.get("base_field_exact_match"),
                "sliced": summary.get("sliced_field_exact_match"),
                "unit": "ratio",
                "higher_is_better": True,
            },
            {
                "metric": "Active Parameters",
                "definition": "Non-zeroed parameters participating in forward pass (active reduction != physical compression)",
                "base": summary.get("base_total_params"),
                "sliced": summary.get("sliced_active_params"),
                "unit": "count",
                "higher_is_better": False,
            },
            {
                "metric": "Total Generation Latency",
                "definition": "Mean generation wall time per sample (excluding model load & validation)",
                "base": summary.get("base_avg_latency_s"),
                "sliced": summary.get("sliced_avg_latency_s"),
                "unit": "seconds",
                "higher_is_better": False,
            },
            {
                "metric": "First-Token Latency",
                "definition": "Mean time to first generated token (when streaming instrumentation is active)",
                "base": summary.get("base_avg_first_token_s"),
                "sliced": summary.get("sliced_avg_first_token_s"),
                "unit": "seconds",
                "higher_is_better": False,
            },
            {
                "metric": "Throughput",
                "definition": "Generated tokens divided by generation wall time",
                "base": summary.get("base_avg_throughput_tps"),
                "sliced": summary.get("sliced_avg_throughput_tps"),
                "unit": "tok/s",
                "higher_is_better": True,
            },
            {
                "metric": "Physical Model Size (Disk)",
                "definition": "Serialized tensor footprint on disk (zero-masking keeps dense tensor shape)",
                "base": summary.get("base_physical_size_bytes"),
                "sliced": summary.get("sliced_physical_size_bytes"),
                "unit": "bytes",
                "higher_is_better": False,
            },
        ]

        rows: List[Dict[str, Any]] = []
        for s in specs:
            b = s["base"]
            sl = s["sliced"]
            if b is None and sl is None:
                continue
            abs_diff = (sl - b) if (b is not None and sl is not None) else None
            rel_pct = compute_relative_change(b, sl, s["higher_is_better"])
            if rel_pct is None:
                direction = "N/A"
            elif abs(rel_pct) < 1e-6:
                direction = "Parity"
            elif rel_pct > 0:
                direction = "Improved"
            else:
                direction = "Degraded"

            rows.append(
                {
                    "metric": s["metric"],
                    "definition": s["definition"],
                    "base_value": b,
                    "sliced_value": sl,
                    "absolute_diff": abs_diff,
                    "relative_improvement_pct": rel_pct,
                    "direction": direction,
                    "higher_is_better": s["higher_is_better"],
                    "unit": s["unit"],
                    "sample_count": sample_count,
                    "conditions": conditions,
                }
            )
        return rows
