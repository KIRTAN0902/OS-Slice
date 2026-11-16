"""
OS-Slice Evidence Exporter
Generates self-contained JSON and Markdown experiment reports with full lineage,
artifact SHA-256 checksums, metrics, and limitations.
"""

from typing import Any, Dict, Optional

from .artifact_manager import ArtifactManager
from .event_logger import EventLogger
from .experiment_schema import ExperimentManifest


class EvidenceExporter:
    """Exports verifiable experiment evidence bundles."""

    def __init__(self, artifact_manager: Optional[ArtifactManager] = None):
        self.artifact_manager = artifact_manager or ArtifactManager()

    def export(self, manifest: ExperimentManifest) -> Dict[str, str]:
        exp_dir = self.artifact_manager.get_experiment_dir(manifest.experiment_id)
        events = EventLogger(manifest.experiment_id, exp_dir / "execution_events.jsonl").read_events()
        verification = self.artifact_manager.verify_artifacts(manifest)

        report_json = {
            "experiment_id": manifest.experiment_id,
            "schema_version": manifest.schema_version,
            "status": manifest.status.value,
            "created_at": manifest.created_at,
            "updated_at": manifest.updated_at,
            "model": manifest.model.model_dump(),
            "operation": manifest.operation,
            "slice_id": manifest.slice_id,
            "calibration": manifest.calibration.model_dump() if manifest.calibration else None,
            "evaluation": manifest.evaluation.model_dump() if manifest.evaluation else None,
            "slicing": manifest.slicing.model_dump(),
            "compensation": manifest.compensation.model_dump(),
            "runtime": manifest.runtime.model_dump(),
            "metrics": manifest.metrics,
            "stages": {k: v.model_dump() for k, v in manifest.stages.items()},
            "artifacts": {k: v.model_dump() for k, v in manifest.artifacts.items()},
            "artifact_verification": verification,
            "event_count": len(events),
        }

        self.artifact_manager.save_json_artifact(
            manifest, "experiment_report_json", "report/experiment_report.json", report_json
        )

        m = manifest.metrics or {}
        md_lines = [
            f"# OS-Slice Experiment Report: `{manifest.experiment_id}`",
            "",
            f"- **Status:** `{manifest.status.value}`",
            f"- **Created:** `{manifest.created_at}`",
            f"- **Operation:** `{manifest.operation}`",
            f"- **Base Model:** `{manifest.model.model_id}` (arch: `{manifest.model.architecture}`, dtype: `{manifest.model.dtype}`)",
            f"- **Slicing Ratio:** `{manifest.slicing.ratio * 100:.1f}%` (protected layers: `{manifest.slicing.protected_layers}`)",
            f"- **Compensation:** `{manifest.compensation.method}` (enabled=`{manifest.compensation.enabled}`, ridge_lambda=`{manifest.compensation.ridge_lambda}`)",
            "",
            "## Measured Paired Metrics",
            "",
            "| Metric | Base | OS-Slice |",
            "| :--- | :---: | :---: |",
            f"| Strict JSON Parseability | {m.get('base_strict_valid_rate', 'N/A')} | {m.get('sliced_strict_valid_rate', 'N/A')} |",
            f"| Normalized JSON Parseability | {m.get('base_normalized_valid_rate', 'N/A')} | {m.get('sliced_normalized_valid_rate', 'N/A')} |",
            f"| Schema Key Adherence | {m.get('base_schema_adherence', 'N/A')} | {m.get('sliced_schema_adherence', 'N/A')} |",
            f"| Active Parameters | {m.get('base_total_params', 'N/A')} | {m.get('sliced_active_params', 'N/A')} (-{m.get('active_param_reduction_pct', 'N/A')}%) |",
            f"| Avg Generation Latency (s) | {m.get('base_avg_latency_s', 'N/A')} | {m.get('sliced_avg_latency_s', 'N/A')} |",
            f"| Avg Throughput (tok/s) | {m.get('base_avg_throughput_tps', 'N/A')} | {m.get('sliced_avg_throughput_tps', 'N/A')} |",
            "",
            "> **Note on Physical Compression:** Zero-masking reduces active participating parameters; physical tensor dimensions and serialized file sizes remain dense unless structurally compacted.",
            "",
            "## Artifact Registry (SHA-256)",
            "",
            "| Artifact | Relative Path | Size (bytes) | SHA-256 |",
            "| :--- | :--- | :---: | :--- |",
        ]
        for name, ref in manifest.artifacts.items():
            md_lines.append(f"| `{name}` | `{ref.relative_path}` | {ref.size_bytes} | `{ref.sha256[:16]}...` |")

        md_content = "\n".join(md_lines) + "\n"
        self.artifact_manager.save_text_artifact(
            manifest, "experiment_report_md", "report/experiment_report.md", md_content, artifact_type="markdown"
        )

        return {
            "json_path": str(exp_dir / "report" / "experiment_report.json"),
            "markdown_path": str(exp_dir / "report" / "experiment_report.md"),
        }
