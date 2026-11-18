# OS-Slice Experiment Report: `OSSL-2026-0001`

- **Status:** `completed`
- **Created:** `2026-11-18T12:19:20.905190+00:00`
- **Operation:** `structured_json_extraction`
- **Base Model:** `Qwen/Qwen2.5-0.5B-Instruct` (arch: `qwen2`, dtype: `float32`)
- **Slicing Ratio:** `20.0%` (protected layers: `[0, 1, 22, 23]`)
- **Compensation:** `diagonal_ridge_damping` (enabled=`True`, ridge_lambda=`0.001`)

## Measured Paired Metrics

| Metric | Base | OS-Slice |
| :--- | :---: | :---: |
| Strict JSON Parseability | 0.0 | 0.0 |
| Normalized JSON Parseability | 1.0 | 1.0 |
| Schema Key Adherence | 1.0 | 1.0 |
| Active Parameters | 494032768 | 437190528 (-11.5058%) |
| Avg Generation Latency (s) | 8.330775000000001 | 16.38635 |
| Avg Throughput (tok/s) | 8.855 | 9.305 |

> **Note on Physical Compression:** Zero-masking reduces active participating parameters; physical tensor dimensions and serialized file sizes remain dense unless structurally compacted.

## Artifact Registry (SHA-256)

| Artifact | Relative Path | Size (bytes) | SHA-256 |
| :--- | :--- | :---: | :--- |
| `model_info` | `model/model_info.json` | 391 | `368a65db81710c2e...` |
| `calibration_metadata` | `calibration/calibration_metadata.json` | 108787 | `b1abd519190ad571...` |
| `fisher_profile` | `profiles/fisher_profile.pt` | 471261 | `ffd709793ee7c8bb...` |
| `profile_summary` | `profiles/profile_summary.json` | 2513 | `8260b24c0ed0ba87...` |
| `attention_masks` | `slicing/attention_masks.pt` | 2977 | `616aee73b5922aee...` |
| `mlp_masks` | `slicing/mlp_masks.pt` | 468535 | `d8e6e3ef7a258d61...` |
| `slicing_report` | `slicing/slicing_report.json` | 15018 | `910854aaba2b2ba8...` |
| `compensation_report` | `compensation/compensation_report.json` | 12381 | `133d66c93d13bfdc...` |
| `predictions` | `evaluation/predictions.json` | 13515 | `a8378257d9bb8cce...` |
| `metrics` | `evaluation/metrics.json` | 6081 | `d8ddac0744b63d2f...` |
| `validation_errors` | `evaluation/validation_errors.json` | 1830 | `e7013df0cc18797a...` |
| `experiment_report_json` | `report/experiment_report.json` | 12320 | `c0539db0e79c41b2...` |
