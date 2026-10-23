"""
OS-Slice Model Catalog Manager
Defines supported open-weights models spanning compact SLMs (<3B) to frontier giants (120B),
along with hardware specifications, VRAM requirements, and OS-Slice research recommendations.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import os


SUPPORTED_MODELS: List[Dict[str, Any]] = [
    # --- Tier 1: SLM & Micro-Edge (< 3B) ---
    {
        "id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "name": "SmolLM2 (360M)",
        "params": "360M",
        "param_count": 360_000_000,
        "disk_size": "~750 MB",
        "min_vram": "2 GB",
        "tier": "SLM (Sub-1B)",
        "tier_category": "Tier 1: SLM & Micro-Edge (< 3B)",
        "recommended_for": "Ultra-fast prototyping & IoT micro-controllers"
    },
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "name": "Qwen 2.5 (0.5B)",
        "params": "494M",
        "param_count": 494_032_768,
        "disk_size": "~1.0 GB",
        "min_vram": "2 GB",
        "tier": "SLM (Sub-1B)",
        "tier_category": "Tier 1: SLM & Micro-Edge (< 3B)",
        "recommended_for": "Fast sanity benchmarks & edge testing"
    },
    {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "name": "Llama 3.2 (1B)",
        "params": "1.23B",
        "param_count": 1_230_000_000,
        "disk_size": "~2.5 GB",
        "min_vram": "4 GB",
        "tier": "Compact (1B-3B)",
        "tier_category": "Tier 1: SLM & Micro-Edge (< 3B)",
        "recommended_for": "Lightweight deterministic routing"
    },
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "name": "Qwen 2.5 (1.5B)",
        "params": "1.54B",
        "param_count": 1_540_000_000,
        "disk_size": "~3.1 GB",
        "min_vram": "6 GB (RTX 4050)",
        "tier": "Compact (1B-3B)",
        "tier_category": "Tier 1: SLM & Micro-Edge (< 3B)",
        "recommended_for": "Balanced edge tool-calling"
    },

    # --- Tier 2: Mid-Range Workhorses (7B - 14B) ---
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "name": "Qwen 2.5 (7B)",
        "params": "7.61B",
        "param_count": 7_610_000_000,
        "disk_size": "~15.2 GB",
        "min_vram": "16 GB (or CPU offload)",
        "tier": "Mid-Range (7B-14B)",
        "tier_category": "Tier 2: Mid-Range Workhorses (7B - 14B)",
        "recommended_for": "Aggressive 50-70% operational excision tests"
    },
    {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "name": "Llama 3.1 (8B)",
        "params": "8.03B",
        "param_count": 8_030_000_000,
        "disk_size": "~16.0 GB",
        "min_vram": "16 GB (or CPU offload)",
        "tier": "Mid-Range (7B-14B)",
        "tier_category": "Tier 2: Mid-Range Workhorses (7B - 14B)",
        "recommended_for": "Standard academic and enterprise baseline"
    },
    {
        "id": "Qwen/Qwen2.5-14B-Instruct",
        "name": "Qwen 2.5 (14B)",
        "params": "14.7B",
        "param_count": 14_700_000_000,
        "disk_size": "~29.5 GB",
        "min_vram": "32 GB",
        "tier": "Mid-Range (7B-14B)",
        "tier_category": "Tier 2: Mid-Range Workhorses (7B - 14B)",
        "recommended_for": "Complex multi-table SQL decomposition"
    },

    # --- Tier 3: High-Capacity & Frontier Giants (70B - 120B) ---
    {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "name": "Llama 3.3 (70B)",
        "params": "70.6B",
        "param_count": 70_600_000_000,
        "disk_size": "~140 GB",
        "min_vram": "140 GB (Multi-GPU / Cloud)",
        "tier": "Large (70B-120B)",
        "tier_category": "Tier 3: High-Capacity & Frontier Giants (70B - 120B)",
        "recommended_for": "Frontier teacher distillation & massive factual null-space cuts"
    },
    {
        "id": "Qwen/Qwen2.5-72B-Instruct",
        "name": "Qwen 2.5 (72B)",
        "params": "72.7B",
        "param_count": 72_700_000_000,
        "disk_size": "~145 GB",
        "min_vram": "145 GB (Multi-GPU / Cloud)",
        "tier": "Large (70B-120B)",
        "tier_category": "Tier 3: High-Capacity & Frontier Giants (70B - 120B)",
        "recommended_for": "SOTA open-weights reasoning & enterprise tool calling"
    },
    {
        "id": "Qwen/Qwen1.5-110B",
        "name": "Qwen 1.5 (110B)",
        "params": "111B",
        "param_count": 111_000_000_000,
        "disk_size": "~220 GB",
        "min_vram": "220 GB (Cloud Cluster)",
        "tier": "Large (70B-120B)",
        "tier_category": "Tier 3: High-Capacity & Frontier Giants (70B - 120B)",
        "recommended_for": "Extreme scaling studies (>100B parameter redundancy)"
    },
    {
        "id": "mlabonne/Meta-Llama-3-120B-Instruct",
        "name": "Meta-Llama-3 (120B)",
        "params": "120B",
        "param_count": 120_000_000_000,
        "disk_size": "~240 GB",
        "min_vram": "240 GB (Cloud Cluster)",
        "tier": "Large (70B-120B)",
        "tier_category": "Tier 3: High-Capacity & Frontier Giants (70B - 120B)",
        "recommended_for": "Upper boundary 120B architecture for research monograph"
    }
]


class ModelCatalogManager:
    """Utility class for querying, filtering, and checking cache status of catalog models."""

    @staticmethod
    def get_all_models() -> List[Dict[str, Any]]:
        return SUPPORTED_MODELS

    @staticmethod
    def get_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
        for m in SUPPORTED_MODELS:
            if m["id"] == model_id:
                return m
        return None

    @staticmethod
    def get_models_by_tier_category(tier_category: str) -> List[Dict[str, Any]]:
        return [m for m in SUPPORTED_MODELS if m.get("tier_category") == tier_category]

    @staticmethod
    def get_all_tier_categories() -> List[str]:
        seen = set()
        categories = []
        for m in SUPPORTED_MODELS:
            cat = m.get("tier_category", m["tier"])
            if cat not in seen:
                seen.add(cat)
                categories.append(cat)
        return categories

    @staticmethod
    def is_model_cached(model_id: str) -> bool:
        """Checks whether the model weights exist in local HuggingFace cache."""
        hf_cache_home = os.environ.get("HF_HOME")
        if not hf_cache_home:
            hf_cache_home = Path.home() / ".cache" / "huggingface" / "hub"
        else:
            hf_cache_home = Path(hf_cache_home) / "hub"

        model_folder_name = f"models--{model_id.replace('/', '--')}"
        model_dir = Path(hf_cache_home) / model_folder_name
        
        # Check if directory exists and has snapshot/blobs
        if model_dir.exists():
            snapshots_dir = model_dir / "snapshots"
            if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                return True
        return False

    @staticmethod
    def get_model_cache_status() -> Dict[str, bool]:
        return {m["id"]: ModelCatalogManager.is_model_cached(m["id"]) for m in SUPPORTED_MODELS}

    @staticmethod
    def get_local_model_path(model_id: str) -> Optional[Path]:
        """Returns the local snapshot path for a cached HuggingFace model if present."""
        hf_cache_home = os.environ.get("HF_HOME")
        if not hf_cache_home:
            hf_cache_home = Path.home() / ".cache" / "huggingface" / "hub"
        else:
            hf_cache_home = Path(hf_cache_home) / "hub"

        model_folder_name = f"models--{model_id.replace('/', '--')}"
        model_dir = Path(hf_cache_home) / model_folder_name
        
        if model_dir.exists():
            snapshots_dir = model_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = list(snapshots_dir.iterdir())
                if snapshots:
                    return snapshots[0]
        return None

    @staticmethod
    def download_model_weights(model_id: str, token: Optional[str] = None) -> Path:
        """Downloads the full model repository from HuggingFace Hub to local cache."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise RuntimeError("huggingface_hub is required to download models. Run 'pip install huggingface-hub'")

        path = snapshot_download(
            repo_id=model_id,
            token=token or os.environ.get("HF_TOKEN"),
            resume_download=True,
        )
        return Path(path)

    @staticmethod
    def get_system_hardware_profile() -> Dict[str, Any]:
        """Returns available system CPU, RAM, and GPU/MPS devices."""
        import torch
        import psutil

        ram_total_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        ram_avail_gb = round(psutil.virtual_memory().available / (1024 ** 3), 1)

        has_cuda = torch.cuda.is_available()
        cuda_device_name = torch.cuda.get_device_name(0) if has_cuda else None
        vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1) if has_cuda else 0.0

        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        preferred_device = "cuda" if has_cuda else ("mps" if has_mps else "cpu")

        return {
            "ram_total_gb": ram_total_gb,
            "ram_avail_gb": ram_avail_gb,
            "has_cuda": has_cuda,
            "cuda_device_name": cuda_device_name,
            "vram_total_gb": vram_total_gb,
            "has_mps": has_mps,
            "preferred_device": preferred_device,
        }

    @staticmethod
    def check_model_feasibility(model_id: str) -> Dict[str, Any]:
        """Audits hardware feasibility for running or slicing the given model."""
        hw = ModelCatalogManager.get_system_hardware_profile()
        model = ModelCatalogManager.get_model_by_id(model_id)
        if not model:
            return {"feasible": False, "reason": f"Model {model_id} not found in catalog"}

        param_count = model.get("param_count", 0)
        fp16_req_gb = (param_count * 2) / (1024 ** 3)
        int4_req_gb = (param_count * 0.5) / (1024 ** 3)

        # Check VRAM or RAM
        if hw["has_cuda"]:
            fits_vram_fp16 = hw["vram_total_gb"] >= fp16_req_gb
            fits_vram_int4 = hw["vram_total_gb"] >= int4_req_gb
        else:
            fits_vram_fp16 = False
            fits_vram_int4 = False

        fits_ram_fp16 = hw["ram_avail_gb"] >= fp16_req_gb
        fits_ram_int4 = hw["ram_avail_gb"] >= int4_req_gb

        return {
            "model_id": model_id,
            "name": model["name"],
            "fp16_req_gb": round(fp16_req_gb, 1),
            "int4_req_gb": round(int4_req_gb, 1),
            "fits_vram_fp16": fits_vram_fp16,
            "fits_vram_int4": fits_vram_int4,
            "fits_ram_fp16": fits_ram_fp16,
            "fits_ram_int4": fits_ram_int4,
            "hardware": hw,
        }

    @staticmethod
    def calculate_slicing_projections(model_id: str, slicing_ratio: float) -> Dict[str, Any]:
        """Calculates theoretical active parameters and memory footprint after slicing."""
        model = ModelCatalogManager.get_model_by_id(model_id)
        if not model:
            return {}

        total_params = model.get("param_count", 0)
        excised_params = total_params * slicing_ratio
        active_params = total_params - excised_params

        # FP16 memory: 2 bytes per parameter
        orig_fp16_gb = (total_params * 2) / (1024 ** 3)
        sliced_fp16_gb = (active_params * 2) / (1024 ** 3)
        saved_gb = orig_fp16_gb - sliced_fp16_gb

        return {
            "model_id": model_id,
            "name": model["name"],
            "total_params": total_params,
            "slicing_ratio": slicing_ratio,
            "active_params": active_params,
            "excised_params": excised_params,
            "orig_fp16_gb": round(orig_fp16_gb, 1),
            "sliced_fp16_gb": round(sliced_fp16_gb, 1),
            "saved_fp16_gb": round(saved_gb, 1),
        }
