"""
OS-Slice Shared Research Services Package
Provides unified experiment lifecycle, Pydantic schemas, event logging,
artifact management, metrics collection, task registry, model arena,
and benchmark services shared by both the Research Lab UI and CLI Workbench.
"""

from .experiment_schema import (
    SCHEMA_VERSION,
    ExperimentStatus,
    StageName,
    StageStatus,
    TaskReadiness,
    CardState,
    ModelIdentity,
    DatasetRef,
    SlicingConfig,
    CompensationConfig,
    RuntimeInfo,
    ArtifactRef,
    EventRecord,
    ExperimentManifest,
    PairedInferenceResult,
    PairedComparisonRecord,
    BenchmarkSummary,
    AblationRunRecord,
    CompressionSweepPoint,
)

__all__ = [
    "SCHEMA_VERSION",
    "ExperimentStatus",
    "StageName",
    "StageStatus",
    "TaskReadiness",
    "CardState",
    "ModelIdentity",
    "DatasetRef",
    "SlicingConfig",
    "CompensationConfig",
    "RuntimeInfo",
    "ArtifactRef",
    "EventRecord",
    "ExperimentManifest",
    "PairedInferenceResult",
    "PairedComparisonRecord",
    "BenchmarkSummary",
    "AblationRunRecord",
    "CompressionSweepPoint",
]
