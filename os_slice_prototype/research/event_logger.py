"""
OS-Slice Structured Event Logger
Persists truthful stage lifecycle events (stage_started, progress, stage_completed, stage_failed)
to per-experiment JSONL logs without fabricating progress or exposing secrets.
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .experiment_schema import EventRecord, utc_now_iso


_SENSITIVE_KEYS = {"token", "api_key", "secret", "password", "hf_token", "authorization"}


def _sanitize_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """Removes potential credential/secret keys from event metadata."""
    cleaned: Dict[str, Any] = {}
    for k, v in data.items():
        if any(sk in str(k).lower() for sk in _SENSITIVE_KEYS):
            cleaned[k] = "[REDACTED]"
        elif isinstance(v, dict):
            cleaned[k] = _sanitize_metadata(v)
        else:
            cleaned[k] = v
    return cleaned


class EventLogger:
    """Appends and reads structured EventRecord lines for an experiment."""

    def __init__(self, experiment_id: str, events_path: Path):
        self.experiment_id = experiment_id
        self.events_path = Path(events_path)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        stage: str,
        event_name: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EventRecord:
        record = EventRecord(
            experiment_id=self.experiment_id,
            event_id=f"evt-{uuid.uuid4().hex[:12]}",
            stage=stage,
            event_name=event_name,
            timestamp=utc_now_iso(),
            status=status,
            metadata=_sanitize_metadata(metadata or {}),
        )
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
        return record

    def stage_started(self, stage: str, metadata: Optional[Dict[str, Any]] = None) -> EventRecord:
        return self.log(stage=stage, event_name="stage_started", status="running", metadata=metadata)

    def progress(self, stage: str, metadata: Optional[Dict[str, Any]] = None) -> EventRecord:
        return self.log(stage=stage, event_name="progress", status="running", metadata=metadata)

    def stage_completed(
        self,
        stage: str,
        duration_s: float,
        artifacts: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EventRecord:
        meta = dict(metadata or {})
        meta["duration_s"] = round(duration_s, 4)
        if artifacts:
            meta["artifacts"] = artifacts
        return self.log(stage=stage, event_name="stage_completed", status="completed", metadata=meta)

    def stage_failed(
        self,
        stage: str,
        error: str,
        duration_s: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EventRecord:
        meta = dict(metadata or {})
        meta["error"] = str(error)[:1000]
        if duration_s is not None:
            meta["duration_s"] = round(duration_s, 4)
        return self.log(stage=stage, event_name="stage_failed", status="failed", metadata=meta)

    def read_events(self) -> List[EventRecord]:
        if not self.events_path.exists():
            return []
        events: List[EventRecord] = []
        with open(self.events_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(EventRecord.model_validate_json(line))
                except Exception:
                    continue
        return events
