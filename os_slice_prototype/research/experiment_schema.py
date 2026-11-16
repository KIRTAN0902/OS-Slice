"""
OS-Slice Pydantic Experiment, Event, and Benchmark Schemas.
Enforces strict data contracts, schema versioning, and lineage tracking
across CLI and Research Lab executions.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


SCHEMA_VERSION = "1.0.0"


def utc_now_iso() -> str:
    """Returns current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class ExperimentStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageName(str, Enum):
    CALIBRATION = "calibration"
    PROFILING = "profiling"
    SLICING = "slicing"
    COMPENSATION = "compensation"
    EVALUATION = "evaluation"
    ARENA = "arena"
    ABLATION = "ablation"
    SWEEP = "sweep"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskReadiness(str, Enum):
    IMPLEMENTED = "implemented"
    CATALOG_TARGET_ONLY = "catalog_target_only"


class CardState(str, Enum):
    VALIDATED_BENCHMARK = "Validated benchmark"
    READY_TO_EXECUTE = "Ready to execute"
    CLOUD_EXECUTION_REQUIRED = "Cloud execution required"
    NOT_EVALUATED = "Not evaluated"
    FAILED_INCOMPLETE = "Failed / incomplete"


class ModelIdentity(BaseModel):
    model_id: str
    revision: str = "main"
    architecture: str = "unknown"
    tokenizer_id: str = ""
    dtype: str = "float32"
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    total_params: Optional[int] = None
    physical_size_bytes: Optional[int] = None


class DatasetRef(BaseModel):
    dataset_id: str
    dataset_version: str = "1.0.0"
    dataset_hash: str
    sample_count: int = 0
    sample_ids: List[str] = Field(default_factory=list)


class SlicingConfig(BaseModel):
    ratio: float = Field(ge=0.0, le=0.95, default=0.20)
    protected_layers: List[Any] = Field(default_factory=lambda: [0, 1, "L-2", "L-1"])
    variance_preserving_scaling: bool = True


class CompensationConfig(BaseModel):
    enabled: bool = True
    method: str = "diagonal_ridge_damping"
    ridge_lambda: float = 1e-3


class RuntimeInfo(BaseModel):
    device: str = "cpu"
    software_versions: Dict[str, str] = Field(default_factory=dict)
    seed: int = 123
    source_interface: str = "research_lab"  # "cli" or "research_lab" or "api"


class ArtifactRef(BaseModel):
    name: str
    relative_path: str
    sha256: str
    size_bytes: int
    artifact_type: str = "json"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StageRecord(BaseModel):
    stage: StageName
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_s: Optional[float] = None
    summary: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class EventRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    experiment_id: str
    event_id: str
    stage: str
    event_name: str  # stage_started, progress, stage_completed, stage_failed
    timestamp: str = Field(default_factory=utc_now_iso)
    status: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentManifest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    experiment_id: str
    status: ExperimentStatus = ExperimentStatus.CREATED
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    label: str = ""
    is_historical_reference: bool = False
    model: ModelIdentity
    operation: str = "structured_json_extraction"
    slice_id: str = ""
    calibration: Optional[DatasetRef] = None
    evaluation: Optional[DatasetRef] = None
    slicing: SlicingConfig = Field(default_factory=SlicingConfig)
    compensation: CompensationConfig = Field(default_factory=CompensationConfig)
    runtime: RuntimeInfo = Field(default_factory=RuntimeInfo)
    stages: Dict[str, StageRecord] = Field(default_factory=dict)
    artifacts: Dict[str, ArtifactRef] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class PairedInferenceResult(BaseModel):
    role: str  # "base" or "sliced"
    model_id: str
    raw_output: str
    is_valid: bool
    validator_name: str
    validation_error: Optional[str] = None
    normalized_valid: Optional[bool] = None
    normalized_status: Optional[str] = None
    task_score: float = 0.0
    schema_adherence: float = 0.0
    field_exact_match: Optional[float] = None
    has_trailing_text: bool = False
    first_token_latency_s: Optional[float] = None
    total_latency_s: float = 0.0
    generated_tokens: int = 0
    throughput_tps: float = 0.0
    run_status: str = "Success"  # Success, Model Error, Timeout, Invalid Output, Cancelled
    error_message: Optional[str] = None
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)


class PairedComparisonRecord(BaseModel):
    run_id: str
    experiment_id: Optional[str] = None
    timestamp: str = Field(default_factory=utc_now_iso)
    model_id: str
    slice_id: str
    task_id: str
    prompt: str
    schema_instruction: str = ""
    ground_truth: Optional[Dict[str, Any]] = None
    decoding_config: Dict[str, Any] = Field(default_factory=dict)
    base: PairedInferenceResult
    sliced: PairedInferenceResult
    pair_lineage_verified: bool = True
    runtime_differences: List[str] = Field(default_factory=list)


class BenchmarkSummary(BaseModel):
    experiment_id: str
    model_id: str
    operation: str
    slice_id: str
    slicing_ratio: float
    sample_count: int
    base_strict_valid_rate: float
    sliced_strict_valid_rate: float
    base_normalized_valid_rate: float
    sliced_normalized_valid_rate: float
    base_schema_adherence: float
    sliced_schema_adherence: float
    base_field_exact_match: Optional[float] = None
    sliced_field_exact_match: Optional[float] = None
    base_avg_latency_s: float
    sliced_avg_latency_s: float
    base_avg_first_token_s: Optional[float] = None
    sliced_avg_first_token_s: Optional[float] = None
    base_avg_throughput_tps: float
    sliced_avg_throughput_tps: float
    base_total_params: int
    sliced_active_params: int
    active_param_reduction_pct: float
    base_physical_size_bytes: Optional[int] = None
    sliced_physical_size_bytes: Optional[int] = None
    peak_ram_mb: Optional[float] = None
    device: str = "cpu"
    dtype: str = "float32"
    warmup_included: bool = False
    timing_scope: str = "generation_wall_time_excluding_load_and_validation"


class AblationRunRecord(BaseModel):
    config_id: str  # E0, E1, E2, E3, E4
    config_name: str
    research_question: str
    experiment_id: Optional[str] = None
    status: str = "completed"
    active_params: int
    total_params: int
    active_reduction_pct: float
    strict_valid_rate: float
    normalized_valid_rate: float
    schema_adherence: float
    avg_latency_s: float
    throughput_tps: float
    notes: str = ""


class CompressionSweepPoint(BaseModel):
    target_ratio: float
    experiment_id: Optional[str] = None
    status: str = "completed"
    feasible: bool = True
    active_params: int
    total_params: int
    active_reduction_pct: float
    strict_valid_rate: float
    normalized_valid_rate: float
    schema_adherence: float
    avg_latency_s: float
    throughput_tps: float
    physical_size_bytes: Optional[int] = None
    peak_ram_mb: Optional[float] = None
    is_pareto_efficient: bool = False
    error: Optional[str] = None
