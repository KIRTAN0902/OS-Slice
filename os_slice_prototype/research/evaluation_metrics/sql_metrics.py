"""
SQL Synthesis Evaluator (Target Profile Specification)
Note: SQL synthesis is a catalog target profile only and is not yet a validated slice.
"""

from typing import Any, Dict, Optional


def evaluate_sql_output(raw_output: str, ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "validator_name": "sql_target_validator@0.1.0-unimplemented",
        "strict_valid": False,
        "strict_error": "SQL synthesis evaluator and task slice are not implemented in this milestone.",
        "normalized_valid": False,
        "normalized_status": "Not implemented",
        "has_trailing_text": False,
        "schema_adherence": 0.0,
        "field_exact_match": None,
        "task_score": 0.0,
    }
