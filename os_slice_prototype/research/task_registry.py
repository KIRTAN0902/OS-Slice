"""
OS-Slice Operational Task Registry
Defines input/output schemas, validators, metric functions, dataset versions,
and readiness status for each operational task. Prevents unimplemented target tasks
from being misrepresented as validated slices.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .experiment_schema import TaskReadiness
from .evaluation_metrics.json_metrics import evaluate_json_output
from .evaluation_metrics.sql_metrics import evaluate_sql_output
from .evaluation_metrics.tool_call_metrics import evaluate_tool_call_output
from .evaluation_metrics.classification_metrics import evaluate_classification_output


@dataclass
class TaskSpec:
    task_id: str
    display_name: str
    readiness: TaskReadiness
    description: str
    input_schema: Dict[str, str]
    output_schema: Dict[str, str]
    validator_fn: Callable[[str, Optional[Dict[str, Any]]], Dict[str, Any]]
    metric_names: List[str]
    ground_truth_required: bool
    dataset_version: str
    supported_slice_ids: List[str] = field(default_factory=list)
    status_note: str = ""


TASK_REGISTRY: Dict[str, TaskSpec] = {
    "structured_json_extraction": TaskSpec(
        task_id="structured_json_extraction",
        display_name="Structured JSON Extraction",
        readiness=TaskReadiness.IMPLEMENTED,
        description=(
            "Extract structured enterprise entities from unstructured context into strict JSON "
            "conforming to a requested schema."
        ),
        input_schema={"context": "str", "schema_instruction": "str"},
        output_schema={"json_object": "dict[str, Any]"},
        validator_fn=evaluate_json_output,
        metric_names=[
            "strict_valid_rate",
            "normalized_valid_rate",
            "schema_adherence",
            "field_exact_match",
        ],
        ground_truth_required=False,
        dataset_version="json-enterprise-v1.1",
        supported_slice_ids=["qwen2.5-0.5b-json-slice"],
        status_note="Implemented and validated benchmark task.",
    ),
    "sql_synthesis": TaskSpec(
        task_id="sql_synthesis",
        display_name="SQL Synthesis",
        readiness=TaskReadiness.CATALOG_TARGET_ONLY,
        description="Deterministic text-to-SQL generation with relational schema binding.",
        input_schema={"question": "str", "ddl_schema": "str"},
        output_schema={"sql_query": "str"},
        validator_fn=evaluate_sql_output,
        metric_names=["sql_syntax_valid", "execution_accuracy"],
        ground_truth_required=True,
        dataset_version="unimplemented",
        supported_slice_ids=[],
        status_note="Catalog target profile only; requires SQL parser, fixed DB, and validated slice.",
    ),
    "api_tool_calling": TaskSpec(
        task_id="api_tool_calling",
        display_name="API / Tool Calling",
        readiness=TaskReadiness.CATALOG_TARGET_ONLY,
        description="Deterministic function selection and argument schema coercion.",
        input_schema={"user_query": "str", "tools_schema": "list[dict]"},
        output_schema={"tool_name": "str", "arguments": "dict"},
        validator_fn=evaluate_tool_call_output,
        metric_names=["tool_name_accuracy", "argument_schema_valid"],
        ground_truth_required=True,
        dataset_version="unimplemented",
        supported_slice_ids=[],
        status_note="Catalog target profile only; requires tool-call validator and validated slice.",
    ),
    "ner_pii_extraction": TaskSpec(
        task_id="ner_pii_extraction",
        display_name="NER / PII Extraction",
        readiness=TaskReadiness.CATALOG_TARGET_ONLY,
        description="Deterministic PII/HIPAA entity detection and redaction.",
        input_schema={"document": "str", "entity_types": "list[str]"},
        output_schema={"entities": "list[dict]"},
        validator_fn=evaluate_classification_output,
        metric_names=["entity_precision", "entity_recall", "entity_f1"],
        ground_truth_required=True,
        dataset_version="unimplemented",
        supported_slice_ids=[],
        status_note="Catalog target profile only; requires synthetic dataset and F1 evaluator.",
    ),
    "operational_action_planning": TaskSpec(
        task_id="operational_action_planning",
        display_name="Operational Action Planning",
        readiness=TaskReadiness.CATALOG_TARGET_ONLY,
        description="Multi-hop structured action state machine planning.",
        input_schema={"state": "str", "available_actions": "list[str]"},
        output_schema={"action_plan": "list[dict]"},
        validator_fn=evaluate_classification_output,
        metric_names=["action_accuracy", "constraint_satisfaction"],
        ground_truth_required=True,
        dataset_version="unimplemented",
        supported_slice_ids=[],
        status_note="Catalog target profile only; requires fixed test set and constraint checker.",
    ),
}


class TaskRegistry:
    """Provides read access and readiness checks over TASK_REGISTRY."""

    @staticmethod
    def list_tasks() -> List[TaskSpec]:
        return list(TASK_REGISTRY.values())

    @staticmethod
    def get_task(task_id: str) -> Optional[TaskSpec]:
        return TASK_REGISTRY.get(task_id)

    @staticmethod
    def is_task_implemented(task_id: str) -> bool:
        spec = TASK_REGISTRY.get(task_id)
        return spec is not None and spec.readiness == TaskReadiness.IMPLEMENTED

    @staticmethod
    def validate_output(
        task_id: str,
        raw_output: str,
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        spec = TASK_REGISTRY.get(task_id)
        if spec is None:
            return {
                "validator_name": "unknown_task",
                "strict_valid": False,
                "strict_error": f"Unknown task_id: {task_id}",
                "normalized_valid": False,
                "normalized_status": "Unknown task",
                "has_trailing_text": False,
                "schema_adherence": 0.0,
                "field_exact_match": None,
                "task_score": 0.0,
            }
        return spec.validator_fn(raw_output, ground_truth)
