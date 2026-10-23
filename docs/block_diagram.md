# OS-Slice: Block Diagram & Architectural Schematics
**Operation-Specific Behavioral Slicing (OS-Slice)**
*System Block Diagrams, Component Schematics, and Tensor Dataflow Architectures*

---

## 1. High-Level End-to-End System Block Diagram

This block diagram illustrates the overall progression from a dense, general-purpose open-weights foundation model to an operational, zero-drift micro-model.

```mermaid
flowchart TD
    subgraph Inputs["1. System Inputs"]
        M0["Dense Generalist Model<br/>(e.g., Qwen 2.5 0.5B - 494.0M Params)"]
        D_op["Operational Calibration Corpus<br/>(Task-Specific Boundary Invariants)"]
    end

    subgraph Stage1["Stage 1: Boundary Token Masking"]
        TM["Token Classifier & Mask Generator<br/>data/generate_calibration.py"]
        W_struct["Syntax Delimiters & Keys<br/>(Weight = 6.0)"]
        W_payload["Semantic Payload<br/>(Weight = 1.0)"]
        L_op["Operational Loss Formulation:<br/>L_op = sum(w_t * CrossEntropy)"]
        
        D_op --> TM
        TM --> W_struct
        TM --> W_payload
        W_struct --> L_op
        W_payload --> L_op
    end

    subgraph Stage2["Stage 2: Fisher Saliency Profiling"]
        FP["Empirical Fisher Profiler<br/>engine/profiler.py"]
        Grad["Forward-Backward Gradient Hooks"]
        FIM["FIM Diagonal Computation:<br/>S_G = sum (grad_w L_op)^2 * w^2"]
        HeadSal["Attention Head Saliency<br/>[Layers x Heads]"]
        MLPSal["SwiGLU MLP Saliency<br/>[Layers x Intermediate]"]
        
        M0 --> Grad
        L_op --> Grad
        Grad --> FP
        FP --> FIM
        FIM --> HeadSal
        FIM --> MLPSal
    end

    subgraph Stage3["Stage 3: Layer-Aware Structural Slicing"]
        SS["Structural Slicer<br/>engine/slicer.py"]
        Protect["Interface Layer Protection<br/>(Layers 0-1 & L-2, L-1: 100% Retained)"]
        MaskGen["Binary Structural Mask Generation<br/>(Threshold Cut on Intermediate Layers)"]
        VarScale["Variance Preserving Scaling:<br/>scale = sqrt(1 / (1 - sparsity))"]
        
        HeadSal --> SS
        MLPSal --> SS
        Protect --> SS
        SS --> MaskGen
        MaskGen --> VarScale
    end

    subgraph Stage4["Stage 4: Second-Order Manifold Compensation"]
        OBS["Optimal Brain Surgeon Compensator<br/>engine/compensator.py"]
        ActCache["Active Activation Caching<br/>X_active from Calibration"]
        RidgeSolve["Closed-Form Ridge Reconstruction:<br/>Delta W = E^T * Z * (Z^T * Z + lambda*I)^-1"]
        
        VarScale --> OBS
        ActCache --> OBS
        OBS --> RidgeSolve
    end

    subgraph Output["5. Deployment & Execution"]
        MicroModel["OS-Slice Micro-Model<br/>(437.2M Active Params / -11.51% Bloat)"]
        Bench["Head-to-Head Empirical Evaluator<br/>(100% Strict JSON Parseability)"]
        TUI["oslice CLI Interactive Workbench<br/>(Claude Code Style TUI)"]
        
        RidgeSolve --> MicroModel
        MicroModel --> Bench
        MicroModel --> TUI
    end

    style Inputs fill:#2d3436,stroke:#dfe6e9,color:#fff
    style Stage1 fill:#0984e3,stroke:#74b9ff,color:#fff
    style Stage2 fill:#6c5ce7,stroke:#a29bfe,color:#fff
    style Stage3 fill:#d63031,stroke:#ff7675,color:#fff
    style Stage4 fill:#00b894,stroke:#55efc4,color:#fff
    style Output fill:#e17055,stroke:#fab1a0,color:#fff
```

---

## 2. Subsystem Interaction & Dataflow Block Diagram

```mermaid
graph LR
    subgraph UserInterface["User Presentation Layer"]
        CLI["oslice CLI<br/>(Terminal TUI)"]
        WebUI["Streamlit Dashboard<br/>(ui.py)"]
    end

    subgraph CatalogManager["Catalog & Storage Subsystem"]
        CatMgr["ModelCatalogManager<br/>(catalog/manager.py)"]
        HFHub["HuggingFace Hub Cache<br/>(~/.cache/huggingface/hub)"]
        SliceMgr["SlicedModelManager<br/>(catalog/sliced_models.py)"]
        ProfileCache["Profile & Saliency Cache<br/>(cache/profiles/*.pt)"]
    end

    subgraph Engine["OS-Slice Algorithmic Engine"]
        Profiler["OperationalProfiler"]
        Slicer["OperationalSlicer"]
        Compensator["OperationalCompensator"]
    end

    subgraph Evaluation["Benchmark Subsystem"]
        Benchmarker["BenchmarkEvaluator<br/>(benchmark.py)"]
        TestCorpus["Unseen Domain Tests<br/>(Legal, Billing, IoT, ETL)"]
    end

    CLI <--> CatMgr
    CLI <--> SliceMgr
    CLI <--> Benchmarker
    WebUI <--> CatMgr

    CatMgr <--> HFHub
    SliceMgr <--> ProfileCache

    CatMgr --> Engine
    ProfileCache --> Slicer
    Engine --> ProfileCache
    Engine --> Benchmarker
    TestCorpus --> Benchmarker
```

---

## 3. Transformer Block Structural Excision Block Diagram

This schematic illustrates how an individual Transformer layer is structurally sliced, showing the exact components zero-masked versus preserved.

```text
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                    TRANSFORMER BLOCK l (Intermediate Layer: 2 <= l <= 21)              │
 │                                                                                        │
 │  Input Tensor: X_in [Batch, SeqLen, HiddenSize]                                        │
 │     │                                                                                  │
 │     ├─── [Residual Stream 1] ──────────────────────────────────────────────────┐       │
 │     ▼                                                                          │       │
 │  ┌──────────────────────────────────────────────────────────────────────────┐  │       │
 │  │ RMSNorm (LayerNorm)                                                      │  │       │
 │  └──────────────────────────────┬───────────────────────────────────────────┘  │       │
 │                                 ▼                                              │       │
 │  ┌──────────────────────────────────────────────────────────────────────────┐  │       │
 │  │ Multi-Head Self-Attention Block                                          │  │       │
 │  │                                                                          │  │       │
 │  │  Query Projections (W_Q):                                                │  │       │
 │  │    Head 0: [PRESERVED - High Saliency]  ────┐                            │  │       │
 │  │    Head 1: [EXCISED   - Low Saliency]   ─X  │                            │  │       │
 │  │    Head 2: [PRESERVED - High Saliency]  ────┼───► Scaled Dot-Product     │  │       │
 │  │    ...                                      │     Attention Engine       │  │       │
 │  │    Head 13: [PRESERVED - Pointer Mech]  ────┘                            │  │       │
 │  │                                                                          │  │       │
 │  │  Key & Value Projections (W_K, W_V):                                     │  │       │
 │  │    Shared across GQA Heads (Key/Value geometry preserved)                │  │       │
 │  │                                                                          │  │       │
 │  │  Output Projection (W_O):                                                │  │       │
 │  │    Active Head Projections compensated via OBS Ridge update              │  │       │
 │  └──────────────────────────────┬───────────────────────────────────────────┘  │       │
 │                                 ▼                                              │       │
 │                             + ◄────────────────────────────────────────────────┘       │
 │                                 │                                                      │
 │     ├─── [Residual Stream 2] ───┼──────────────────────────────────────────────┐       │
 │     ▼                           ▼                                              │       │
 │  ┌──────────────────────────────────────────────────────────────────────────┐  │       │
 │  │ RMSNorm (LayerNorm)                                                      │  │       │
 │  └──────────────────────────────┬───────────────────────────────────────────┘  │       │
 │                                 ▼                                              │       │
 │  ┌──────────────────────────────────────────────────────────────────────────┐  │       │
 │  │ SwiGLU MLP Feed-Forward Block                                            │  │       │
 │  │                                                                          │  │       │
 │  │  Intermediate Dimension (d_ff = 4864 channels):                          │  │       │
 │  │    Channels [0 .. 4192]:  [PRESERVED] (Operational Syntax Manifold)      │  │       │
 │  │    Channels [4193 .. 4863]: [ZERO-MASKED] (Non-functional Factual Bloat) │  │       │
 │  │                                                                          │  │       │
 │  │  Projection Operations:                                                  │  │       │
 │  │    gate_proj (W_gate) ──┐                                                │  │       │
 │  │                         ├──► Swish(Gate) * Up ──► down_proj (W_down)     │  │       │
 │  │    up_proj (W_up)     ──┘                            ▲                   │  │       │
 │  │                                                      │                   │  │       │
 │  │    Second-Order OBS Compensation Applied Here: ──────┘                   │  │       │
 │  │    Delta W_down = E^T * Z_active * (Z_active^T * Z_active + lambda*I)^-1 │  │       │
 │  └──────────────────────────────┬───────────────────────────────────────────┘  │       │
 │                                 ▼                                              │       │
 │                             + ◄────────────────────────────────────────────────┘       │
 │                                 │                                                      │
 │  Output Tensor: X_out [Batch, SeqLen, HiddenSize]                                      │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Layer Retention Geometry: Interface vs Core Layers

```mermaid
flowchart TD
    subgraph L0_L1["Layers 0 & 1: Embedding Interface"]
        direction TB
        E1["Token & Position Embeddings"]
        L0["Layer 0 Transformer Block"]
        L1["Layer 1 Transformer Block"]
        R1["Retention Policy: 100% PRESERVED<br/>Sparsity = 0.0%"]
        E1 --- L0 --- L1 --- R1
    end

    subgraph L2_L21["Layers 2 through 21: Core Transformation Manifold"]
        direction TB
        Lmid["Layers 2 - 21 (20 Transformer Blocks)"]
        Slicing["Layer-Aware Slicing Applied<br/>• Structural Attention Head Slicing<br/>• SwiGLU Intermediate Channel Slicing<br/>• Non-operational Factual Null-Space Excised"]
        R2["Retention Policy: 86.2% ACTIVE / 13.8% EXCISED<br/>Variance Scaling: sqrt(1 / (1 - s))"]
        Lmid --- Slicing --- R2
    end

    subgraph L22_L23["Layers 22 & 23: Unembedding Interface"]
        direction TB
        L22["Layer 22 Transformer Block"]
        L23["Layer 23 Transformer Block"]
        Head["Final RMSNorm & lm_head"]
        R3["Retention Policy: 100% PRESERVED<br/>Sparsity = 0.0%"]
        L22 --- L23 --- Head --- R3
    end

    L0_L1 --> L2_L21 --> L22_L23

    style L0_L1 fill:#2ed573,stroke:#7bed9f,color:#000
    style L2_L21 fill:#ffa502,stroke:#eccc68,color:#000
    style L22_L23 fill:#2ed573,stroke:#7bed9f,color:#000
```

---

## 5. Active vs Excised Parameter Manifold Block Diagram

```mermaid
graph TB
    subgraph FullCapacity["Full Monolithic Generalist Model (494.0M Params / 100%)"]
        subgraph FactualBloat["Excised Parameter Manifold (-11.51% / 56.8M Params)"]
            F1["World History & Encyclopedia Recall"]
            F2["Conversational Small-talk Pragmatics"]
            F3["Creative Storytelling & Metaphor Manifolds"]
            F4["Unstructured Multilingual Redundancies"]
        end
        
        subgraph ActiveManifold["Active Operational Manifold (88.49% / 437.2M Params)"]
            O1["Syntactic Token Delimiter Alignment ({ } [ ] : , \" )"]
            O2["Type-Safe Schema Field Binding & Key Tracking"]
            O3["Inductive Pointer Heads for Context Copying"]
            O4["OBS Second-Order Compensated Projection Manifolds"]
        end
    end

    FactualBloat -->|Empirical Fisher Saliency Filter| ZERO["Physically Zero-Masked<br/>(Bandwidth Reclaimed)"]
    ActiveManifold -->|OBS Ridge Compensator| OP_MODEL["OS-Slice Micro-Model<br/>(100% Deterministic Schema Fidelity)"]

    style FactualBloat fill:#ff4757,stroke:#ff6b81,color:#fff
    style ActiveManifold fill:#1e90ff,stroke:#70a1ff,color:#fff
    style ZERO fill:#747d8c,stroke:#a4b0be,color:#fff
    style OP_MODEL fill:#2ed573,stroke:#7bed9f,color:#fff
```

---

## 6. oslice Interactive CLI Runtime State Machine

```mermaid
stateDiagram-v2
    [*] --> Init: User types 'oslice' in terminal
    Init --> HeaderRender: Detect Hardware (RAM, CUDA, MPS)
    HeaderRender --> MainMenu: Render Claude Code TUI

    MainMenu --> PreSlicedHub: Option [1]
    MainMenu --> HFModelLibrary: Option [2]
    MainMenu --> BenchmarksDeepAnalysis: Option [3]
    MainMenu --> CreateNewSlice: Option [4]
    MainMenu --> InferencePlayground: Option [5]
    MainMenu --> LaunchWebUI: Option [6]
    MainMenu --> SystemDiagnosis: Option [7]
    MainMenu --> [*]: Option [q] (Exit)

    state PreSlicedHub {
        [*] --> ListSlices
        ListSlices --> InspectSliceDossier: Select Slice [1-6]
        InspectSliceDossier --> DownloadOrActivate: Action [1]
        InspectSliceDossier --> TestPrompt: Action [2]
        InspectSliceDossier --> BenchmarksDeepAnalysis: Action [3]
        InspectSliceDossier --> ListSlices: Back [b]
        ListSlices --> MainMenu: Back [b]
    }

    state HFModelLibrary {
        [*] --> ListTiers: Models 360M to 120B
        ListTiers --> ModelDetail: Select [1-11]
        ModelDetail --> DownloadHFWeights: Action [1]
        ModelDetail --> TriggerSlicePipeline: Action [2]
        ModelDetail --> VRAMCalculator: Action [3]
        ModelDetail --> ListTiers: Back [b]
        ListTiers --> MainMenu: Back [b]
    }

    state BenchmarksDeepAnalysis {
        [*] --> ComparativeScorecard: Base vs OS-Slice Table
        ComparativeScorecard --> InspectTestCases: Option [1] (Legal, Billing, IoT, ETL)
        ComparativeScorecard --> RunLiveBenchmark: Option [2] (Stream real tokens)
        ComparativeScorecard --> InferencePlayground: Option [3]
        ComparativeScorecard --> MainMenu: Back [b]
    }
```
