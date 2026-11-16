# OS-Slice: Complete Knowledge Transfer (KT) Document

**Project:** Operation-Specific Behavioral Slicing (`OS-Slice`)  
**Repository:** `https://github.com/KIRTAN0902/OS-Slice` (Branch: `main`)  
**Local Workspace:** `C:\kirtan\college\OS-SLICE`  
**Version:** `v1.0.0` (Research Framework, Prototype Engine, CLI Workbench & Web Console)  
**Document Purpose:** Complete technical, mathematical, architectural, and operational handover for incoming researchers, engineers, and project maintainers.

---

## Table of Contents
1. [Executive Summary & Core Thesis](#1-executive-summary--core-thesis)
2. [Research Foundations & Literature Lineage](#2-research-foundations--literature-lineage)
3. [Repository Map & File Inventory](#3-repository-map--file-inventory)
4. [Core Engine: Code-to-Math Implementation Guide](#4-core-engine-code-to-math-implementation-guide)
5. [Data & Calibration Pipeline](#5-data--calibration-pipeline)
6. [Model Catalog & Pre-Sliced Registry](#6-model-catalog--pre-sliced-registry)
7. [Configuration, Caching & Artifact Schemas](#7-configuration-caching--artifact-schemas)
8. [User Interfaces & Operational Entry Points](#8-user-interfaces--operational-entry-points)
9. [Empirical Benchmarks & Observed Results](#9-empirical-benchmarks--observed-results)
10. [Installation, Environment & Global CLI Setup](#10-installation-environment--global-cli-setup)
11. [Known Limitations, Technical Gotchas & Debt](#11-known-limitations-technical-gotchas--debt)
12. [Video & Presentation Assets](#12-video--presentation-assets)
13. [Maintainer Roadmap & Next Engineering Steps](#13-maintainer-roadmap--next-engineering-steps)

---

## 1. Executive Summary & Core Thesis

### The Problem: The Over-Provisioning Pathology
Enterprise AI systems routinely deploy dense, general-purpose Large Language Models (7B–120B+ parameters) to execute narrow, deterministic procedural tasks—such as Structured JSON Schema Extraction, Text-to-SQL synthesis, API tool calling, and PII redaction. Monolithic generalists waste the majority of their memory bandwidth on encyclopedic factual recall and conversational fluency while remaining vulnerable to **stochastic syntax drift** (malformed JSON, extra prose, dropped schema keys).

### The Solution: Operation-Specific Behavioral Slicing (`OS-Slice`)
**OS-Slice** isolates and preserves only the parameter sub-networks responsible for deterministic operational invariants, while zero-masking non-operational generalist weights through a 4-stage pipeline:
1. **Boundary-Conditioned Operational Token Masking** (`data/generate_calibration.py`)
2. **Empirical Fisher Information Saliency Profiling** (`engine/profiler.py`)
3. **Layer-Aware Structural Parameter Slicing** (`engine/slicer.py`)
4. **Second-Order Manifold Compensation** (`engine/compensator.py`)

This yields an operational micro-model (`OSSliceModel`) verified end-to-end on `Qwen/Qwen2.5-0.5B-Instruct` with **100.0% Strict JSON Parseability**, **100.0% Schema Key Adherence**, and **11.51% total parameter reduction** (56.8M parameters excised at a 20% intermediate-layer slicing ratio).

---

## 2. Research Foundations & Literature Lineage

The framework is grounded in a critical review of **18 foundational papers** compiled in:
- [`docs/papers.txt`](docs/papers.txt) — Bibliography across 4 research pillars
- [`docs/os_slice_literature_review.md`](docs/os_slice_literature_review.md) — Exhaustive peer-reviewer critique of all 18 papers
- [`docs/OS_Slice_Literature_Review_and_Synthesis.pdf`](docs/OS_Slice_Literature_Review_and_Synthesis.pdf) — Publication-ready white-paper synthesis

| Research Pillar | Representative Papers Analyzed | Key Architectural Influence on OS-Slice |
| :--- | :--- | :--- |
| **1. Circuit Discovery & Mechanistic Slicing** | ACDC (Conmy et al.), Attribution-Guided Pruning (Yao et al.), IOI (Wang et al.), ROME (Meng et al.) | Motivated boundary-conditioned attribution rather than coarse layer-averaged importance. |
| **2. Structured Pruning & Weight Masking** | LLM-Pruner (Ma et al.), Sheared LLaMA (Xia et al.), Wanda (Sun et al.), SparseGPT (Frantar & Alistarh), Gromov et al. | Adopted group-level attention head + SwiGLU MLP channel slicing, outer layer protection, and second-order error compensation concepts. |
| **3. Knowledge Distillation & Compact SLMs** | Distilling Step-by-Step (Hsieh et al.), MiniLLM (Gu et al.), TinyLlama, SmolLM2, MobileLLM, BabyLLaMA | Informed sub-3B SLM edge targets (`SmolLM2-360M`, `Qwen2.5-0.5B`, `Llama-3.2-1B`). |
| **4. Enterprise Tooling & Schema Compliance** | Gorilla (Patil et al.), SynCode (Ugare et al.), BIRD (Li et al.) | Motivated strict JSON parseability and key adherence metrics over generic perplexity benchmarks. |

---

## 3. Repository Map & File Inventory

```text
C:\kirtan\college\OS-SLICE\
├── README.md                          # Primary GitHub documentation & badges
├── KNOWLEDGE_TRANSFER.md              # This complete KT handover guide
├── architect.md                       # System & mathematical architecture spec
├── Block diagram.md                   # Mermaid & ASCII block diagrams
├── workflow.md                        # 6-phase operational lifecycle spec
├── install.ps1                        # Windows 1-line installer & PATH/PROFILE setup
├── oslice.cmd / oslice.ps1            # Root CLI launch wrappers
├── oslice_cli.py                      # Claude Code-style interactive Rich TUI (939 lines)
├── ui.py                              # Standalone Tiered Model Catalog Streamlit UI
├── run_benchmark.ps1 / run_ui.ps1     # Quick PowerShell launchers
├── bin/
│   ├── oslice.cmd                     # Global PATH CMD wrapper
│   └── oslice.ps1                     # Global PATH PowerShell wrapper
├── catalog/
│   ├── __init__.py                    # Exports SUPPORTED_MODELS, SLICED_MODELS, Managers
│   ├── manager.py                     # 11-model catalog (360M-120B), HF cache & downloader
│   └── sliced_models.py               # Registry of 6 pre-sliced operational models
├── docs/
│   ├── papers.txt                     # 18-paper bibliography
│   ├── os_slice_literature_review.md  # Full 18-paper critical audit
│   ├── OS_Slice_Literature_Review_and_Synthesis.pdf
│   ├── architect.md                   # Mirror of root architect.md
│   ├── block_diagram.md               # Mirror of root Block diagram.md
│   └── workflow.md                    # Mirror of root workflow.md
├── os_slice_prototype/
│   ├── README.md                      # Prototype-specific docs
│   ├── requirements.txt               # Python dependencies
│   ├── config.py                      # OSSliceConfig dataclass & paths
│   ├── app.py                         # Full 4-tab Streamlit evaluation console
│   ├── benchmark.py                   # Standalone head-to-head benchmarking pipeline
│   ├── ui.py                          # Prototype mirror of root ui.py
│   ├── catalog/                       # Prototype mirror of root catalog/
│   ├── data/
│   │   └── generate_calibration.py    # 8 calibration + 4 unseen eval records & token masking
│   ├── engine/
│   │   ├── __init__.py                # Exports Profiler, Slicer, OSSliceModel, Compensator
│   │   ├── profiler.py                # Empirical Fisher saliency computation & .pt cache
│   │   ├── slicer.py                  # StructuralSlicer & OSSliceModel wrapper
│   │   └── compensator.py             # OperationalCompensator (OBS-style ridge damping)
│   └── cache/
│       ├── profiles/
│       │   └── operational_fisher_profile.pt   # Precomputed 471 KB Fisher profile (Qwen 0.5B)
│       └── benchmarks/
│           └── latest_benchmark_results.json   # Persisted empirical benchmark artifact
└── Media / Recording Scripts:
    ├── record_cli_screen.py           # Playwright + FFmpeg CLI screen recorder
    ├── cli_recorder.html              # Standalone animated terminal player
    ├── OS_Slice_CLI_Screen_Recording.mp4 / .gif / _Thumbnail.png
    ├── record_full_demo.py            # Playwright + FFmpeg Streamlit UI recorder
    ├── terminal_showcase.html         # HTML terminal overview slide
    └── OS_Slice_UI_Demonstration.mp4 / .gif / _Thumbnail.png
```

> **Maintainer Warning on Mirrored Files:** `catalog/` and `ui.py` exist both at the repository root and inside `os_slice_prototype/`. Always sync changes between both locations so `oslice_cli.py` (which imports from root `catalog/`) and `os_slice_prototype/app.py` (which imports from `os_slice_prototype/catalog/` and `os_slice_prototype/ui.py`) stay consistent.

---

## 4. Core Engine: Code-to-Math Implementation Guide

All core slicing logic resides in `os_slice_prototype/engine/`.

### Stage 1: Operational Loss & Boundary Token Weighting (`engine/profiler.py` & `data/generate_calibration.py`)
Rather than treating all tokens equally during gradient attribution, `identify_operational_boundary_tokens()` assigns token-level importance weights $w_t$:
- **Prompt tokens:** `labels = -100`, `operational_weights = 0.0` (excluded from loss).
- **Structural delimiters** (`{`, `}`, `[`, `]`, `:`, `,`, `"`, whitespace/newlines) and **schema keys / JSON boolean/null literals** (`true`, `false`, `null`): weighted **`boundary_weight = 6.0`**.
- **Numeric tokens** (`^[0-9.]+$`): weighted **`boundary_weight * 0.7 = 4.2`**.
- **Semantic payload content** (entity strings such as `"Apex Logistics LLC"`): weighted **`content_weight = 1.0`**.

In `OperationalProfiler.compute_operational_loss()`, autoregressive shift logits and shift labels compute masked weighted Cross-Entropy:
$$\mathcal{L}_{op} = \frac{\sum_t (w_t \cdot \mathbb{I}[y_t \neq -100]) \cdot \ell_{CE}(\hat{y}_t, y_t)}{\sum_t w_t \cdot \mathbb{I}[y_t \neq -100]}$$

### Stage 2: Empirical Fisher Saliency Profiling (`engine/profiler.py`)
`OperationalProfiler` inspects `model.config` (for Qwen/LLaMA-style models, locating layers via `model.layers`, `transformer.h`, or `layers`) and initializes:
- `head_saliency` of shape `[num_layers, num_heads]`
- `mlp_saliency` of shape `[num_layers, intermediate_size]`

For each calibration sample, it runs `loss.backward()` and computes the weight-scaled empirical Fisher diagonal:
- **Attention Heads (`q_proj` and `o_proj`):**
  $$S_{head}^{(l, h)} = \sum_{w \in (W_{Q,h}^{(l)}, W_{O,h}^{(l)})} \left(\frac{\partial \mathcal{L}_{op}}{\partial w}\right)^2 \cdot w^2$$
- **SwiGLU MLP Channels (`gate_proj` and `down_proj`):**
  $$S_{mlp}^{(l, m)} = \sum_{w \in (W_{gate,m,:}^{(l)}, W_{down,:,m}^{(l)})} \left(\frac{\partial \mathcal{L}_{op}}{\partial w}\right)^2 \cdot w^2$$

Scores are averaged across calibration samples and persisted via `torch.save()` to `operational_fisher_profile.pt` along with layer-wise mean importance metadata.

### Stage 3: Layer-Aware Structural Slicing (`engine/slicer.py`)
`StructuralSlicer.compute_structural_masks()` enforces **Interface Layer Protection**:
- **Protected Layers:** Layers `0`, `1`, `L-2`, and `L-1` retain **100%** of heads and MLP channels (`mask = 1.0`), preventing token embedding/unembedding manifold collapse.
- **Sliced Layers (`range(2, num_layers - 2)`):** For each intermediate layer, `torch.topk(..., k, largest=False)` zeros out the lowest-saliency `slicing_ratio` fraction of attention heads (`k_h = max(int(num_heads * slicing_ratio), 1)`) and MLP channels (`k_m = int(intermediate_size * slicing_ratio)`).

`StructuralSlicer.apply_slicing()` clones the model (unless `in_place=True`), zeroes:
- Attention: `q_proj.weight[q_start:q_end, :]`, `q_proj.bias[q_start:q_end]` (if present), and `o_proj.weight[:, q_start:q_end]`
- MLP: `gate_proj.weight[pruned_m, :]`, `up_proj.weight[pruned_m, :]`, and `down_proj.weight[:, pruned_m]`

**Variance-Preserving Scaling:** Surviving parameters are rescaled to preserve output activation variance:
- Heads: `o_proj.weight[:, active_h_slice] *= (num_heads / len(active_h)) ** 0.5`
- MLPs: `down_proj.weight[:, active_m] *= (intermediate_size / len(active_m)) ** 0.5`

The result is wrapped in `OSSliceModel(nn.Module)`, which tracks `total_params`, exact `active_params`, `sliced_params`, and `effective_sparsity`, and delegates `forward()` and `generate()` (defaulting `repetition_penalty=1.05`).

### Stage 4: Localized Error Compensation (`engine/compensator.py`)
`OperationalCompensator.compensate()` applies second-order OBS-inspired ridge damping across surviving active channels in layers `2` through `num_layers - 3` (`ridge_lambda = 1e-3`):
- **MLP `down_proj`:** `damping = 1.0 + ridge_lambda * (len(pruned_m) / intermediate_size)`, scaling `mlp.down_proj.weight[:, active_m] *= (1.0 / damping)`
- **Attention `o_proj`:** `damping_h = 1.0 + ridge_lambda * (len(pruned_h) / num_heads)`, scaling `self_attn.o_proj.weight[:, q_start:q_end] *= (1.0 / damping_h)`

> **Implementation Note for Maintainers:** While the class docstring and architectural docs describe the full closed-form ridge equation $\Delta W = E^T Z (Z^T Z + \lambda I)^{-1}$, the current prototype implementation in `compensator.py` executes fast diagonal ridge damping (`1.0 / damping`) in `torch.no_grad()` without caching full calibration covariance matrices $Z^T Z$. Implementing full layer-wise Hessian/covariance inversion is an explicit next-step upgrade (see Section 13).

---

## 5. Data & Calibration Pipeline

Located in `os_slice_prototype/data/generate_calibration.py`:
- **`CALIBRATION_RECORDS` (8 Enterprise Domains):**
  1. Financial Transactions (`WTRX-90142`)
  2. DevOps Incident Response (`auth-gateway` 504)
  3. Clinical Assessment (`MRN-44910` asthma ICD-10)
  4. E-Commerce Logistics (`PKG-78234`)
  5. Customer Support Dispatch (`TCK-5521`)
  6. Cybersecurity Audit (SSH brute-force log)
  7. HR Employee Records (`EMP-1093`)
  8. Supply Chain Inventory (`WH-CHICAGO-2`)
- **`EVALUATION_RECORDS` (4 Strictly Unseen Zero-Shot Domains):**
  1. Legal Contract Extraction (`SA-2026-99`)
  2. Cloud Infrastructure Billing (`INV-88219`)
  3. IoT Environmental Monitoring (`SENSOR-ENV-104`)
  4. Database Migration Job (`ETL-MIG-771`)
- **Prompt Template (`format_prompt`):** Uses Qwen/ChatML special tokens (`<|im_start|>system`, `<|im_start|>user`, `<|im_start|>assistant`) instructing strict JSON output without markdown fences or prose.

---

## 6. Model Catalog & Pre-Sliced Registry

### A. Foundational Model Catalog (`catalog/manager.py` — `SUPPORTED_MODELS`)
Covers **11 open-weights models** spanning 3 architectural tiers:

| Tier | Model Name | Hugging Face ID | Params | Disk Size | Min VRAM |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Tier 1: SLM & Micro-Edge (< 3B)** | SmolLM2 (360M) | `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M | ~750 MB | 2 GB |
| **Tier 1: SLM & Micro-Edge (< 3B)** | Qwen 2.5 (0.5B) | `Qwen/Qwen2.5-0.5B-Instruct` | 494M | ~1.0 GB | 2 GB |
| **Tier 1: SLM & Micro-Edge (< 3B)** | Llama 3.2 (1B) | `meta-llama/Llama-3.2-1B-Instruct` | 1.23B | ~2.5 GB | 4 GB |
| **Tier 1: SLM & Micro-Edge (< 3B)** | Qwen 2.5 (1.5B) | `Qwen/Qwen2.5-1.5B-Instruct` | 1.54B | ~3.1 GB | 6 GB (RTX 4050) |
| **Tier 2: Mid-Range (7B - 14B)** | Qwen 2.5 (7B) | `Qwen/Qwen2.5-7B-Instruct` | 7.61B | ~15.2 GB | 16 GB |
| **Tier 2: Mid-Range (7B - 14B)** | Llama 3.1 (8B) | `meta-llama/Llama-3.1-8B-Instruct` | 8.03B | ~16.0 GB | 16 GB |
| **Tier 2: Mid-Range (7B - 14B)** | Qwen 2.5 (14B) | `Qwen/Qwen2.5-14B-Instruct` | 14.7B | ~29.5 GB | 32 GB |
| **Tier 3: Frontier (70B - 120B)** | Llama 3.3 (70B) | `meta-llama/Llama-3.3-70B-Instruct` | 70.6B | ~140 GB | 140 GB |
| **Tier 3: Frontier (70B - 120B)** | Qwen 2.5 (72B) | `Qwen/Qwen2.5-72B-Instruct` | 72.7B | ~145 GB | 145 GB |
| **Tier 3: Frontier (70B - 120B)** | Qwen 1.5 (110B) | `Qwen/Qwen1.5-110B` | 111B | ~220 GB | 220 GB |
| **Tier 3: Frontier (70B - 120B)** | Meta-Llama-3 (120B) | `mlabonne/Meta-Llama-3-120B-Instruct` | 120B | ~240 GB | 240 GB |

`ModelCatalogManager` provides:
- `is_model_cached()` / `get_local_model_path()`: Inspects `$HF_HOME/hub` or `~/.cache/huggingface/hub/models--{org}--{repo}/snapshots`.
- `download_model_weights()`: Calls `huggingface_hub.snapshot_download(..., resume_download=True)`.
- `get_system_hardware_profile()` & `check_model_feasibility()`: Inspects RAM (`psutil`), CUDA VRAM, and Apple MPS, computing FP16 (`param_count * 2`) and INT4 (`param_count * 0.5`) footprints.
- `calculate_slicing_projections()`: Projects post-slicing active parameters and FP16 GB savings.

### B. Pre-Sliced Models Registry (`catalog/sliced_models.py` — `SLICED_MODELS`)
Defines **6 operational slice profiles**:
1. **`qwen2.5-0.5b-json-slice`** (`LOCAL_ACTIVE`) — Backed by precomputed `operational_fisher_profile.pt` and real `latest_benchmark_results.json` metrics (`437.2M / 494.0M` params, `-11.51%` reduction, `100.0%` parseability).
2. **`smollm2-360m-sql-slice`** (`READY_TO_SLICE`) — Target: Deterministic SQL Query Synthesis (`309.6M / 360.0M`, `-14.0%`).
3. **`llama-3.2-1b-tool-slice`** (`READY_TO_SLICE`) — Target: API Tool Calling (`980.3M / 1.23B`, `-20.3%`).
4. **`qwen2.5-1.5b-entity-slice`** (`READY_TO_SLICE`) — Target: PII Redaction & NER (`1.21B / 1.54B`, `-21.43%`).
5. **`qwen2.5-7b-agent-slice`** (`CLOUD_SLICE_READY`) — Target: ReAct Action Planning (`5.30B / 7.61B`, `-30.3%`).
6. **`llama-3.1-8b-contract-slice`** (`CLOUD_SLICE_READY`) — Target: Legal Contract Clause Auditing (`5.82B / 8.03B`, `-27.52%`).

*(Note: Entries 2–6 are catalog target profiles without `.pt` caches in the repo yet; activating them triggers calibration/slicing or requires local/cloud execution.)*

---

## 7. Configuration, Caching & Artifact Schemas

### Runtime Configuration (`os_slice_prototype/config.py`)
`OSSliceConfig` defaults:
- `model_id`: `"Qwen/Qwen2.5-0.5B-Instruct"`
- `operational_primitive`: `"Structured JSON Schema Extraction"`
- `device`: Auto-detected (`cuda` -> `mps` -> `cpu`)
- `dtype`: `bfloat16`/`float16` on CUDA, `float32` on CPU
- `default_slicing_ratio`: `0.20` (bounds: `0.05` to `0.65`)
- `boundary_token_weight`: `6.0`, `content_token_weight`: `1.0`
- `num_calibration_samples`: `24`, `profiling_batch_size`: `2`, `max_seq_len`: `512`
- `compensation_steps`: `15`, `compensation_lr`: `4e-4`, `compensation_weight_decay`: `1e-4`
- `eval_max_new_tokens`: `180` (note: `generate_prediction()` defaults to `160`, `run_benchmark()` passes `config.eval_max_new_tokens`), `eval_temperature`: `0.0`

### Persisted Artifacts
1. **`os_slice_prototype/cache/profiles/operational_fisher_profile.pt`** (~471 KB):
   PyTorch dict with keys: `"head_saliency"` (`Tensor [24, 14]`), `"mlp_saliency"` (`Tensor [24, 4864]`), `"metadata"` (mean/max saliency & per-layer importance lists).
2. **`os_slice_prototype/cache/benchmarks/latest_benchmark_results.json`** (~5.7 KB):
   JSON object with `"metrics"`, `"base_samples"`, and `"sliced_samples"`.

---

## 8. User Interfaces & Operational Entry Points

| Interface | Entry Command | Description |
| :--- | :--- | :--- |
| **Claude Code-Style Terminal CLI** | `oslice` (or `python oslice_cli.py`) | Rich-based TUI with 7 modules: `[1]` Pre-Sliced Hub, `[2]` HF Library (up to 120B + downloader), `[3]` Benchmarks & Deep Analysis (reads `latest_benchmark_results.json` + can invoke `benchmark.py` live), `[4]` Create Slice Wizard, `[5]` Inference Playground, `[6]` Launch Web UI, `[7]` System Diagnosis. |
| **Full Streamlit Evaluation Console** | `streamlit run os_slice_prototype/app.py` (or `.\run_ui.ps1`) | Loads base & sliced models via `@st.cache_resource`, exposes 4 tabs: Side-by-Side Live Generation, Behavioral Circuit Saliency (Altair heatmaps), Empirical Benchmarks, and Model Catalog (360M–120B). |
| **Standalone Catalog Web UI** | `streamlit run ui.py` | Lightweight catalog + VRAM savings calculator without loading PyTorch model weights. |
| **Standalone CLI Benchmark** | `python os_slice_prototype/benchmark.py` (or `.\run_benchmark.ps1`) | End-to-end execution: loads model, loads/computes Fisher profile, slices, compensates, runs greedy decoding (`do_sample=False`) on `EVALUATION_RECORDS`, writes `latest_benchmark_results.json`. |

---

## 9. Empirical Benchmarks & Observed Results

Measured from `os_slice_prototype/cache/benchmarks/latest_benchmark_results.json` (`Qwen/Qwen2.5-0.5B-Instruct`, CPU execution, `default_slicing_ratio = 0.20` on intermediate layers 2–21):

| Evaluation Metric | Base Generalist Model | OS-Slice Micro-Model | Observed Delta |
| :--- | :---: | :---: | :--- |
| **Active Parameters** | 494,032,768 (100.0%) | **437,190,528 (88.49%)** | **-56,842,240 (-11.51%)** |
| **Strict JSON Parseability** | 100.0% (4 / 4) | **100.0% (4 / 4)** | Parity (0% syntax corruption) |
| **Schema Key Adherence** | 100.0% | **100.0%** | Parity (all 6 keys matched/domain) |
| **Average Throughput** | 15.30 tok/s | **15.62 tok/s** | **+2.05% token generation rate** |
| **Average Latency / Sample** | 5.02 s | **10.64 s** | Higher wall-time due to post-JSON tokens (see §11.2) |

Per-domain validation (`clean_and_parse_json()` returned `"Strict JSON Valid (Markdown Block)"` and `adherence = 1.0` on all 4):
1. **Legal Contract Extraction:** Base `4.49s` (`16.24 tok/s`) vs Sliced `9.88s` (`18.23 tok/s`)
2. **Cloud Infrastructure Billing:** Base `4.29s` (`17.97 tok/s`) vs Sliced `10.30s` (`17.48 tok/s`)
3. **IoT Environmental Monitoring:** Base `6.96s` (`10.64 tok/s`) vs Sliced `18.19s` (`9.90 tok/s`)
4. **Database Migration Job:** Base `4.34s` (`16.37 tok/s`) vs Sliced `4.21s` (`16.87 tok/s`)

---

## 10. Installation, Environment & Global CLI Setup

### One-Line Automated Setup (Windows PowerShell)
```powershell
irm https://raw.githubusercontent.com/KIRTAN0902/OS-Slice/main/install.ps1 | iex
```

What `install.ps1` does:
1. Clones `https://github.com/KIRTAN0902/OS-Slice.git` (or downloads zip archive if `git` is unavailable).
2. Creates `.venv` using `uv` (if installed) or `python -m venv .venv`.
3. Installs CPU PyTorch (`https://download.pytorch.org/whl/cpu`), `os_slice_prototype/requirements.txt`, `huggingface-hub`, and `rich`.
4. Writes `bin/oslice.cmd` & `bin/oslice.ps1`, appends `bin/` to User `PATH`, and adds `function oslice` to PowerShell `$PROFILE`.
5. Presents a 4-option launch menu defaulting to `oslice_cli.py`.

### Runtime Environment Details
- Python `3.12+` inside `.venv` (`torch==2.14.1+cpu`, `transformers==5.20.2`, `streamlit==1.66.1`, `rich==15.0.0`, `huggingface-hub==2.2.0`, `altair==6.3.0`).
- Local HF cache detected on dev workstation: `C:\Users\shahk\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct`.

---

## 11. Known Limitations, Technical Gotchas & Debt

Incoming engineers should be aware of the following exact system behaviors before modifying or presenting the codebase:

1. **Zero-Masking vs. Physical Matrix Compaction (`engine/slicer.py`):**
   `StructuralSlicer` physically sets pruned head/channel weights to `0.0` inside dense tensors (`OSSliceModel`) rather than slicing out rows/columns into smaller `nn.Linear` modules. Thus, while `active_params` drops by 11.51% (proving behavioral isolation), physical RAM/VRAM footprint remains identical to the dense model until physical structural compaction (`torch.nn.Linear` resizing + GQA head indexing) is implemented.
2. **Post-JSON Repetition / Latency Tail (`latest_benchmark_results.json`):**
   On 3 of 4 unseen samples (Legal, Billing, IoT), the sliced model produced valid JSON first (`100%` valid & `1.0` adherence) and then continued generating trailing explanation text or a second JSON block until `max_new_tokens=180`, causing `sliced_avg_latency_s` (`10.64s`) to exceed `base_avg_latency_s` (`5.02s`) despite higher `tokens_per_sec` (`15.62` vs `15.30`). Adding `<|im_end|>` / closing-fence stop criteria or weighting EOS tokens in `generate_calibration.py` will eliminate this tail.
3. **Simulated Interactive Paths in `oslice_cli.py`:**
   - Modules `1` (Dossier), `2` (HF Catalog & `snapshot_download`), `3` (cached JSON scorecard + live `benchmark.py` subprocess), and `7` (Hardware Audit) read live system/disk state.
   - However, `run_slice_pipeline_for_model()` (Module 4) and `run_interactive_inference_playground()` (Module 5) in `oslice_cli.py` are currently fast UI demonstrations (`time.sleep` + preset JSON responses) rather than invoking `OperationalProfiler`/`OSSliceModel` in-process. Real live PyTorch slicing and inference run in `os_slice_prototype/benchmark.py` and `os_slice_prototype/app.py`.
4. **Windows Console UTF-8 Requirement:**
   Without `sys.stdout.reconfigure(encoding="utf-8")`, Windows PowerShell (`cp1252`) throws `UnicodeEncodeError` on Rich emojis and box characters. Always include the UTF-8 guard (already present at top of `oslice_cli.py`).
5. **Streamlit File-Watcher Warning with `transformers`:**
   Running `streamlit run os_slice_prototype/app.py` logs non-fatal `local_sources_watcher` tracebacks (`ModuleNotFoundError: No module named 'torchvision'`) while scanning lazy vision modules inside `transformers`. Launching with `--server.fileWatcherType none` suppresses this noise.
6. **Gated Models Require `HF_TOKEN`:**
   Llama 3.x models in `SUPPORTED_MODELS` (`Llama-3.2-1B`, `Llama-3.1-8B`, `Llama-3.3-70B`) require Hugging Face license approval and an `HF_TOKEN` environment variable (prompted in `oslice_cli.py`).
7. **Saliency Profile Coupling:**
   `operational_fisher_profile.pt` is shaped specifically for `Qwen/Qwen2.5-0.5B-Instruct` (`[24, 14]` heads, `[24, 4864]` MLP channels). Switching `selected_model_id` in `app.py` while keeping the same `operational_fisher_profile.pt` filename on disk will trigger a shape mismatch unless profile filenames are keyed by `model_id`.

---

## 12. Video & Presentation Assets

Pre-recorded 1080p demonstration artifacts are committed in the repo root for mentor/stakeholder reviews:
- **Exact Terminal CLI Recording:**
  - [`OS_Slice_CLI_Screen_Recording.mp4`](OS_Slice_CLI_Screen_Recording.mp4) (2.60 MB, 1080p, 30fps, 01:01 runtime)
  - [`OS_Slice_CLI_Screen_Recording.gif`](OS_Slice_CLI_Screen_Recording.gif) & [`OS_Slice_CLI_Thumbnail.png`](OS_Slice_CLI_Thumbnail.png)
  - Regenerate via `python record_cli_screen.py` (uses `cli_recorder.html` + Playwright + FFmpeg).
- **Full Console + Streamlit UI Recording:**
  - [`OS_Slice_UI_Demonstration.mp4`](OS_Slice_UI_Demonstration.mp4) (3.23 MB, 1080p, 30fps, 01:36 runtime)
  - [`OS_Slice_UI_Demonstration.gif`](OS_Slice_UI_Demonstration.gif) & [`OS_Slice_UI_Thumbnail.png`](OS_Slice_UI_Thumbnail.png)
  - Regenerate via `python record_full_demo.py` (requires Streamlit running on `http://localhost:8501`).

---

## 13. Maintainer Roadmap & Next Engineering Steps

1. **Physical Weight Compaction (`engine/slicer.py`):**
   Replace zero-masking with true structural tensor resizing (slicing out pruned `q_proj` rows / `o_proj` columns and `gate_proj`/`up_proj`/`down_proj` channels, updating `config.intermediate_size` / per-layer sizes) so RAM, VRAM, and disk size shrink proportionally.
2. **Full Second-Order Covariance Inversion (`engine/compensator.py`):**
   Cache layer input activations $Z_{active}$ across `CALIBRATION_RECORDS` and solve the exact closed-form ridge update $\Delta W_{active} = E^T Z_{active} (Z_{active}^T Z_{active} + \lambda I)^{-1}$ for `down_proj` and `o_proj`.
3. **Model-Keyed Profile Caching (`config.py` / `app.py` / `benchmark.py`):**
   Save profiles as `cache/profiles/{safe_model_id}_{operation}_fisher.pt` to support multi-model benchmarking without cache collisions.
4. **EOS Boundary Weighting (`data/generate_calibration.py`):**
   Include `<|im_end|>` / `eos_token_id` in `completion_ids` with `boundary_weight` so sliced models terminate immediately after closing `}`.
5. **Wire Live Engine into `oslice_cli.py`:**
   Connect Module `[4]` (Create Slice) and Module `[5]` (Playground) to `OperationalProfiler`, `StructuralSlicer`, and `generate_prediction()`, deduplicating `catalog/` and `os_slice_prototype/catalog/`.
6. **Automated Test Suite:**
   Add `pytest` unit tests covering `identify_operational_boundary_tokens()`, mask invariants on layers `0, 1, L-2, L-1`, `clean_and_parse_json()`, and `ModelCatalogManager.calculate_slicing_projections()`.
