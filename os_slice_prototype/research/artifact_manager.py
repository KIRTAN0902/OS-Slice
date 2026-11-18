"""
OS-Slice Artifact & Experiment Directory Manager
Manages per-experiment directory layout, SHA-256 checksums, manifest persistence,
and model/task-specific Fisher profile cache keys.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from .experiment_schema import ArtifactRef, ExperimentManifest, utc_now_iso


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hex digest for a file on disk."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dict_hash(data: Any) -> str:
    """Computes deterministic SHA-256 hex digest for JSON-serializable data."""
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_profile_cache_key(
    model_id: str,
    revision: str,
    architecture: str,
    operation: str,
    calibration_hash: str,
    boundary_weight: float,
    content_weight: float,
    include_eos: bool = True,
) -> str:
    """
    Constructs a strict model- and task-specific cache key so a Fisher profile
    is never reused across different model architectures, revisions, or tasks.
    """
    raw = {
        "model_id": model_id,
        "revision": revision,
        "architecture": architecture,
        "operation": operation,
        "calibration_hash": calibration_hash,
        "boundary_weight": round(float(boundary_weight), 4),
        "content_weight": round(float(content_weight), 4),
        "include_eos": bool(include_eos),
    }
    digest = compute_dict_hash(raw)[:16]
    safe_model = model_id.replace("/", "_").replace(":", "_")
    return f"{safe_model}__{operation}__{digest}"


class ArtifactManager:
    """Manages experiment directories, manifest persistence, and artifact hashing."""

    SUBDIRS = [
        "model",
        "calibration",
        "profiles",
        "slicing",
        "compensation",
        "inference",
        "inference/raw_outputs",
        "evaluation",
        "report",
    ]

    def __init__(self, experiments_root: Optional[Path] = None):
        if experiments_root is None:
            experiments_root = Path(__file__).resolve().parent.parent / "experiments"
        self.experiments_root = Path(experiments_root)
        self.experiments_root.mkdir(parents=True, exist_ok=True)

    def generate_experiment_id(self) -> str:
        """Generates a sequential OSSL-YYYY-NNNN experiment identifier."""
        year = datetime.now(timezone.utc).year
        prefix = f"OSSL-{year}-"
        existing_nums: List[int] = []
        if self.experiments_root.exists():
            for p in self.experiments_root.iterdir():
                if p.is_dir() and p.name.startswith(prefix):
                    tail = p.name[len(prefix):]
                    if tail.isdigit():
                        existing_nums.append(int(tail))
        next_num = max(existing_nums, default=0) + 1
        return f"{prefix}{next_num:04d}"

    def get_experiment_dir(self, experiment_id: str) -> Path:
        return self.experiments_root / experiment_id

    def init_experiment_dir(self, experiment_id: str) -> Path:
        exp_dir = self.get_experiment_dir(experiment_id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        for sub in self.SUBDIRS:
            (exp_dir / sub).mkdir(parents=True, exist_ok=True)
        return exp_dir

    def save_manifest(self, manifest: ExperimentManifest) -> Path:
        exp_dir = self.init_experiment_dir(manifest.experiment_id)
        manifest.updated_at = utc_now_iso()
        manifest_path = exp_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
        return manifest_path

    def load_manifest(self, experiment_id: str) -> Optional[ExperimentManifest]:
        manifest_path = self.get_experiment_dir(experiment_id) / "manifest.json"
        if not manifest_path.exists():
            return None
        with open(manifest_path, "r", encoding="utf-8") as f:
            return ExperimentManifest.model_validate_json(f.read())

    def list_manifests(self) -> List[ExperimentManifest]:
        manifests: List[ExperimentManifest] = []
        if not self.experiments_root.exists():
            return manifests
        for p in sorted(self.experiments_root.iterdir(), reverse=True):
            if p.is_dir() and (p / "manifest.json").exists():
                try:
                    m = self.load_manifest(p.name)
                    if m is not None:
                        manifests.append(m)
                except Exception:
                    continue
        return manifests

    def register_artifact(
        self,
        manifest: ExperimentManifest,
        name: str,
        file_path: Path,
        artifact_type: str = "json",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRef:
        exp_dir = self.get_experiment_dir(manifest.experiment_id)
        rel_path = str(Path(file_path).resolve().relative_to(exp_dir.resolve())).replace("\\", "/")
        ref = ArtifactRef(
            name=name,
            relative_path=rel_path,
            sha256=compute_sha256(file_path),
            size_bytes=file_path.stat().st_size,
            artifact_type=artifact_type,
            metadata=metadata or {},
        )
        manifest.artifacts[name] = ref
        self.save_manifest(manifest)
        return ref

    def save_json_artifact(
        self,
        manifest: ExperimentManifest,
        name: str,
        relative_path: str,
        payload: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRef:
        exp_dir = self.init_experiment_dir(manifest.experiment_id)
        out_path = exp_dir / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        return self.register_artifact(manifest, name, out_path, artifact_type="json", metadata=metadata)

    def save_tensor_artifact(
        self,
        manifest: ExperimentManifest,
        name: str,
        relative_path: str,
        payload: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRef:
        exp_dir = self.init_experiment_dir(manifest.experiment_id)
        out_path = exp_dir / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out_path)
        return self.register_artifact(manifest, name, out_path, artifact_type="torch_pt", metadata=metadata)

    def save_text_artifact(
        self,
        manifest: ExperimentManifest,
        name: str,
        relative_path: str,
        content: str,
        artifact_type: str = "markdown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRef:
        exp_dir = self.init_experiment_dir(manifest.experiment_id)
        out_path = exp_dir / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        return self.register_artifact(manifest, name, out_path, artifact_type=artifact_type, metadata=metadata)

    def verify_artifacts(self, manifest: ExperimentManifest) -> Dict[str, Any]:
        """Verifies existence and SHA-256 checksums for all artifacts in a manifest."""
        exp_dir = self.get_experiment_dir(manifest.experiment_id)
        results: Dict[str, Any] = {}
        all_ok = True
        for name, ref in manifest.artifacts.items():
            full_path = exp_dir / ref.relative_path
            exists = full_path.exists()
            checksum_match = False
            if exists:
                checksum_match = compute_sha256(full_path) == ref.sha256
            if not (exists and checksum_match):
                all_ok = False
            results[name] = {
                "relative_path": ref.relative_path,
                "exists": exists,
                "checksum_match": checksum_match,
                "expected_sha256": ref.sha256,
            }
        return {"all_valid": all_ok, "artifacts": results}
