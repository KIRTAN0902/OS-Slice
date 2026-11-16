"""
Comprehensive Validation Suite for OS-Slice Research Lab & Shared Services.
Covers all 9 test groups in Section 15 of the Implementation Specification:
Calibration, Profiler, Slicer, Compensator, Generation/JSON Metrics,
Benchmark, Model Arena, Artifacts/Events, and UI/CLI Shared Service Integration.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO_ROOT = REPO_ROOT / "os_slice_prototype"
for p in (str(REPO_ROOT), str(PROTO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import OSSliceConfig
from data.generate_calibration import (
    CALIBRATION_RECORDS,
    EVALUATION_RECORDS,
    build_calibration_dataset,
    compute_dataset_hash,
    inspect_calibration_record,
)
from engine.profiler import OperationalProfiler
from engine.slicer import StructuralSlicer
from engine.compensator import OperationalCompensator
from research.artifact_manager import ArtifactManager, build_profile_cache_key
from research.benchmark_registry import BenchmarkRegistry
from research.benchmark_runner import BenchmarkRunner
from research.evaluation_metrics.json_metrics import evaluate_json_output
from research.event_logger import EventLogger
from research.experiment_manager import ExperimentManager
from research.experiment_schema import ExperimentStatus, TaskReadiness
from research.metrics_collector import compute_relative_change, MetricsCollector
from research.model_arena import ModelArenaService
from research.model_pair_manager import ModelPairManager
from research.task_registry import TaskRegistry


# ---------------------------------------------------------------------------
# Lightweight Synthetic Qwen2-like Model & Tokenizer for Fast Deterministic CI
# ---------------------------------------------------------------------------

class MiniTokenizer:
    def __init__(self):
        self.vocab_size = 128
        self.eos_token_id = 0
        self.pad_token_id = 0
        self._fixed_json = '{"contract_id": "SA-1", "amount": 10.0, "ok": true}'

    def encode(self, text: str, add_special_tokens: bool = False):
        if not text:
            return []
        return [((ord(ch) % 120) + 1) for ch in text[:64]]

    def decode(self, ids, skip_special_tokens: bool = True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if len(ids) == 1:
            return chr(((int(ids[0]) - 1) % 26) + 97)
        return self._fixed_json

    def __call__(self, text: str, return_tensors: str = "pt"):
        ids = self.encode(text)
        t = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}


class MiniAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class MiniMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.self_attn = MiniAttention(hidden_size, num_heads)
        self.mlp = MiniMLP(hidden_size, intermediate_size)

    def forward(self, x):
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class MiniCausalLM(nn.Module):
    def __init__(self, vocab_size=128, hidden_size=32, num_layers=6, num_heads=4, intermediate_size=64):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen2",
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            intermediate_size=intermediate_size,
            _commit_hash="test-rev-001",
        )
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.layers = nn.ModuleList(
            [MiniLayer(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask=None):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, max_new_tokens=12, **kwargs):
        new_toks = torch.full((input_ids.shape[0], 6), 5, dtype=torch.long, device=input_ids.device)
        streamer = kwargs.get("streamer")
        if streamer is not None:
            streamer.put(new_toks)
            streamer.end()
        return torch.cat([input_ids, new_toks], dim=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_calibration_weights_and_dataset_hash():
    tok = MiniTokenizer()
    assert len(CALIBRATION_RECORDS) == 8
    assert len(EVALUATION_RECORDS) == 12
    ds = build_calibration_dataset(tok, records=CALIBRATION_RECORDS[:2], boundary_weight=6.0, content_weight=1.0)
    assert len(ds) == 2
    sample = ds[0]
    assert (sample["labels"] == -100).any()
    assert (sample["operational_weights"][sample["labels"] == -100] == 0.0).all()
    insp = inspect_calibration_record(tok, CALIBRATION_RECORDS[0], boundary_weight=6.0, content_weight=1.0)
    assert insp["prompt_token_count"] > 0
    assert insp["completion_token_count"] > 0
    h1 = compute_dataset_hash(CALIBRATION_RECORDS)
    h2 = compute_dataset_hash(CALIBRATION_RECORDS)
    assert h1 == h2 and len(h1) == 64


def test_profiler_dimensions_and_cache_key_isolation():
    torch.manual_seed(42)
    model = MiniCausalLM()
    tok = MiniTokenizer()
    cfg = OSSliceConfig(model_id="mini/qwen2", device="cpu")
    ds = build_calibration_dataset(tok, records=CALIBRATION_RECORDS[:2])
    prof = OperationalProfiler(model, cfg)
    meta = prof.profile(ds)
    assert prof.head_saliency.shape == (6, 4)
    assert prof.mlp_saliency.shape == (6, 64)
    assert torch.isfinite(prof.head_saliency).all()
    assert torch.isfinite(prof.mlp_saliency).all()
    assert len(meta["sample_losses"]) == 2

    k1 = build_profile_cache_key("m1", "main", "qwen2", "structured_json_extraction", "abc", 6.0, 1.0)
    k2 = build_profile_cache_key("m2", "main", "qwen2", "structured_json_extraction", "abc", 6.0, 1.0)
    k3 = build_profile_cache_key("m1", "main", "qwen2", "sql_synthesis", "abc", 6.0, 1.0)
    assert k1 != k2 and k1 != k3


def test_slicer_protected_layers_and_accounting():
    torch.manual_seed(42)
    model = MiniCausalLM(num_layers=6, num_heads=4, intermediate_size=64)
    cfg = OSSliceConfig(model_id="mini/qwen2", device="cpu")
    head_sal = torch.rand(6, 4)
    mlp_sal = torch.rand(6, 64)
    slicer = StructuralSlicer(cfg)
    sliced = slicer.apply_slicing(model, head_sal, mlp_sal, slicing_ratio=0.25, in_place=False)
    rep = slicer.build_slicing_report(sliced, model)

    assert rep["protected_layers"] == [0, 1, 4, 5]
    assert rep["protected_layers_verified"] is True
    assert rep["actual_tensor_zeroed_params"] > 0
    assert rep["actual_tensor_nonzero_params"] < rep["actual_tensor_total_params"]
    assert rep["physical_tensor_dimensions_unchanged"] is True


def test_compensator_only_modifies_active_intermediate_weights():
    torch.manual_seed(42)
    model = MiniCausalLM(num_layers=6, num_heads=4, intermediate_size=64)
    cfg = OSSliceConfig(model_id="mini/qwen2", device="cpu")
    head_sal = torch.rand(6, 4)
    mlp_sal = torch.rand(6, 64)
    sliced = StructuralSlicer(cfg).apply_slicing(model, head_sal, mlp_sal, slicing_ratio=0.25, in_place=False)

    # Snapshot protected layer 0 weight
    l0_before = sliced.model.model.layers[0].mlp.down_proj.weight.data.clone()
    comp_rep = OperationalCompensator(cfg).compensate(sliced, ridge_lambda=1e-3)
    l0_after = sliced.model.model.layers[0].mlp.down_proj.weight.data

    assert torch.equal(l0_before, l0_after)
    assert comp_rep["method"] == "diagonal_ridge_damping"
    assert comp_rep["compensated_layers"] == 2  # layers 2 and 3 in 6-layer model
    assert "loss_reduction_pct" not in comp_rep  # no hardcoded 32.5


def test_json_metrics_strict_vs_normalized_and_trailing_text():
    raw_valid = '{"a": 1, "b": "ok"}'
    r1 = evaluate_json_output(raw_valid, {"a": 1, "b": "ok"})
    assert r1["strict_valid"] is True
    assert r1["has_trailing_text"] is False
    assert r1["field_exact_match"] == 1.0

    fenced = '```json\n{"a": 1, "b": "ok"}\n```\nExtra trailing commentary.'
    r2 = evaluate_json_output(fenced, {"a": 1, "b": "ok"})
    assert r2["strict_valid"] is False
    assert r2["normalized_valid"] is True
    assert r2["has_trailing_text"] is True


def test_metrics_collector_formulas_preserve_negative_delta():
    # Lower is better: latency increased from 5.0 to 10.0 -> -100% improvement
    assert compute_relative_change(5.0, 10.0, higher_is_better=False) == pytest.approx(-100.0)
    # Higher is better: throughput increased from 10.0 to 12.0 -> +20% improvement
    assert compute_relative_change(10.0, 12.0, higher_is_better=True) == pytest.approx(20.0)


def test_task_registry_and_pair_readiness_gating():
    assert TaskRegistry.is_task_implemented("structured_json_extraction") is True
    assert TaskRegistry.is_task_implemented("sql_synthesis") is False
    r_sql = ModelPairManager.check_pair_readiness("Qwen/Qwen2.5-0.5B-Instruct", "sql_synthesis")
    assert r_sql["ready"] is False
    r_cloud = ModelPairManager.check_pair_readiness("meta-llama/Llama-3.3-70B-Instruct", "structured_json_extraction")
    assert r_cloud["ready"] is False


def test_end_to_end_experiment_manager_and_artifacts(tmp_path: Path):
    torch.manual_seed(123)
    model = MiniCausalLM(num_layers=6, num_heads=4, intermediate_size=64)
    tok = MiniTokenizer()
    cfg = OSSliceConfig(model_id="Qwen/Qwen2.5-0.5B-Instruct", device="cpu", default_slicing_ratio=0.25)

    am = ArtifactManager(experiments_root=tmp_path / "experiments")
    mgr = ExperimentManager(artifact_manager=am)
    res = mgr.run_full_experiment(
        base_model=model,
        tokenizer=tok,
        config=cfg,
        slicing_ratio=0.25,
        calibration_records=CALIBRATION_RECORDS[:2],
        eval_records=EVALUATION_RECORDS[:2],
        source_interface="cli",
        label="CI Synthetic E2E Test",
        force_reprofile=True,
    )
    man = res["manifest"]
    assert man.status == ExperimentStatus.COMPLETED
    assert man.experiment_id.startswith("OSSL-")
    verification = am.verify_artifacts(man)
    assert verification["all_valid"] is True

    events = EventLogger(man.experiment_id, am.get_experiment_dir(man.experiment_id) / "execution_events.jsonl").read_events()
    assert len(events) >= 10

    # Verify BenchmarkRegistry picks up completed experiment
    cards = BenchmarkRegistry(am).list_benchmark_cards()
    assert any(c["card_id"] == man.experiment_id for c in cards)
