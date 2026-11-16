"""
Structured JSON Extraction Evaluator (v1.0.0)
Strictly separates raw JSON parseability from normalized/cleaned extraction,
detects trailing generation tails, and computes schema adherence and field exact match.
"""

import json
import re
from typing import Any, Dict, Optional, Tuple


VALIDATOR_NAME = "json_strict_validator"
VALIDATOR_VERSION = "1.0.0"


def strict_parse_json(raw_output: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Strictly parses raw model output using json.loads without any fence stripping
    or substring extraction.
    """
    stripped = raw_output.strip()
    if not stripped:
        return False, None, "Empty output"
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return True, parsed, "Valid raw JSON object"
        return False, None, f"Parsed JSON is {type(parsed).__name__}, expected object"
    except json.JSONDecodeError as e:
        return False, None, f"Strict JSONDecodeError: {e.msg} (line {e.lineno}, col {e.colno})"


def normalized_parse_json(raw_output: str) -> Tuple[bool, Optional[Dict[str, Any]], str, bool]:
    """
    Secondary diagnostic parser: attempts markdown fence stripping or balanced-brace extraction.
    Returns (is_valid, parsed_dict, status_msg, has_trailing_text).
    """
    clean_text = raw_output.strip()
    strict_ok, strict_obj, strict_msg = strict_parse_json(clean_text)
    if strict_ok:
        return True, strict_obj, "Strict JSON Valid (Raw)", False

    # 1. Markdown fenced block
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
    if md_match:
        try:
            parsed = json.loads(md_match.group(1))
            if isinstance(parsed, dict):
                after = clean_text[md_match.end():].strip()
                before = clean_text[:md_match.start()].strip()
                has_trail = bool(after or before)
                return True, parsed, "Normalized JSON Valid (Markdown Block)", has_trail
        except Exception:
            pass

    # 2. Balanced brace extraction
    start_idx = clean_text.find("{")
    if start_idx != -1:
        bracket_count = 0
        end_idx = -1
        in_string = False
        escape_next = False
        for i in range(start_idx, len(clean_text)):
            ch = clean_text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == "{":
                    bracket_count += 1
                elif ch == "}":
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
        if end_idx != -1:
            candidate = clean_text[start_idx:end_idx]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    has_trail = bool(clean_text[end_idx:].strip() or clean_text[:start_idx].strip())
                    return True, parsed, "Normalized JSON Valid (Extracted Object)", has_trail
            except Exception as e:
                return False, None, f"Broken syntax in extracted object: {str(e)[:60]}", False

    return False, None, strict_msg, False


def evaluate_schema_adherence(
    predicted: Optional[Dict[str, Any]],
    expected: Optional[Dict[str, Any]],
) -> float:
    """Calculates key-level schema adherence score in [0.0, 1.0]."""
    if predicted is None or not isinstance(predicted, dict):
        return 0.0
    if not expected:
        return 1.0
    expected_keys = set(expected.keys())
    predicted_keys = set(predicted.keys())
    if not expected_keys:
        return 1.0
    return len(expected_keys.intersection(predicted_keys)) / len(expected_keys)


def evaluate_field_exact_match(
    predicted: Optional[Dict[str, Any]],
    expected: Optional[Dict[str, Any]],
) -> Optional[float]:
    """Calculates exact value match fraction across expected schema keys."""
    if expected is None:
        return None
    if predicted is None or not isinstance(predicted, dict):
        return 0.0
    if not expected:
        return 1.0
    matches = 0
    for k, exp_v in expected.items():
        if k in predicted:
            pred_v = predicted[k]
            if isinstance(exp_v, float) and isinstance(pred_v, (int, float)):
                if abs(float(pred_v) - exp_v) < 1e-6:
                    matches += 1
            elif pred_v == exp_v:
                matches += 1
    return matches / len(expected)


def evaluate_json_output(
    raw_output: str,
    ground_truth: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Runs strict raw JSON validation first, then records normalized validation
    as a separate diagnostic without mutating raw_output.
    """
    strict_ok, strict_obj, strict_msg = strict_parse_json(raw_output)
    norm_ok, norm_obj, norm_msg, has_trailing = normalized_parse_json(raw_output)

    # Strict adherence uses strict_obj if strictly valid; normalized adherence uses norm_obj
    scoring_obj = strict_obj if strict_ok else norm_obj
    adherence = evaluate_schema_adherence(scoring_obj, ground_truth) if ground_truth is not None else (1.0 if strict_ok else 0.0)
    exact_match = evaluate_field_exact_match(scoring_obj, ground_truth) if ground_truth is not None else None

    return {
        "validator_name": f"{VALIDATOR_NAME}@{VALIDATOR_VERSION}",
        "strict_valid": strict_ok,
        "strict_error": None if strict_ok else strict_msg,
        "normalized_valid": norm_ok,
        "normalized_status": norm_msg,
        "has_trailing_text": has_trailing,
        "schema_adherence": adherence,
        "field_exact_match": exact_match,
        "parsed_object": scoring_obj,
        "task_score": exact_match if exact_match is not None else adherence,
    }
