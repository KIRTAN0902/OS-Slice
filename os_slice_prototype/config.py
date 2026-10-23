"""
OS-Slice Configuration Module
Defines runtime parameters, model identifiers, slicing ratios, and paths.
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class OSSliceConfig:
    # Model specification
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Target Operational Primitive
    operational_primitive: str = "Structured JSON Schema Extraction"
    
    # Compute environment
    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available() 
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() 
              else "cpu")
    ))
    
    # Data type
    @property
    def dtype(self) -> torch.dtype:
        if self.device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    # Slicing & Sparsity Budgets
    default_slicing_ratio: float = 0.20  # 20% structural parameter reduction by default
    min_slicing_ratio: float = 0.05
    max_slicing_ratio: float = 0.65
    
    # Operational boundary weighting
    boundary_token_weight: float = 6.0   # Emphasis on JSON delimiters, braces, keys
    content_token_weight: float = 1.0    # Emphasis on extracted values
    
    # Calibration & Saliency Profiling
    num_calibration_samples: int = 24
    profiling_batch_size: int = 2
    max_seq_len: int = 512
    
    # Localized Error Compensation
    compensation_steps: int = 15
    compensation_lr: float = 4e-4
    compensation_weight_decay: float = 1e-4
    
    # Evaluation Benchmark
    num_eval_samples: int = 12
    eval_max_new_tokens: int = 180
    eval_temperature: float = 0.0  # Deterministic decoding for benchmarking
    
    # Directories
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.resolve())
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.resolve() / "data")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).parent.resolve() / "cache")
    profiles_dir: Path = field(default_factory=lambda: Path(__file__).parent.resolve() / "cache" / "profiles")
    benchmark_dir: Path = field(default_factory=lambda: Path(__file__).parent.resolve() / "cache" / "benchmarks")

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)

    def summary(self) -> dict:
        return {
            "model_id": self.model_id,
            "operational_primitive": self.operational_primitive,
            "device": self.device,
            "dtype": str(self.dtype),
            "default_slicing_ratio": f"{self.default_slicing_ratio * 100:.1f}%",
            "boundary_token_weight": self.boundary_token_weight,
            "calibration_samples": self.num_calibration_samples,
            "compensation_steps": self.compensation_steps,
        }


# Default global instance
default_config = OSSliceConfig()
