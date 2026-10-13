# UIE Evidence Benchmark

This package accompanies the revised survey *Underwater Image Enhancement:
Imaging Models, Learning Paradigms, Benchmarks, and Evaluation*. It is designed
to reproduce the article's evidence tables, not to claim standardized retraining
of all networks.

## What is compared

- Main quality track: RAW plus 40 enhancement algorithms, 1,253 images from
  eight data sources, including 800 paired and 453 no-reference images.
- Frontier transfer track: a deterministic 287-image subset, capped at 40 images
  per source, for official released W2WDiff and HRMamba checkpoints together
  with matched established-method outputs.
- Downstream track: the same seven enhancement methods plus RAW, evaluated with
  unchanged detector parameters on TrashCan/YOLOv8m-seg and
  URPC2020/IEUOD-YOLOv12n.

The main and frontier tracks use released checkpoints as-is. They measure
released-model transfer and reproducibility under disclosed adapters; they are
not a fair-training leaderboard.

## Repository contents

- `manifests/`: dataset/sample identifiers, reference availability, deterministic
  selection scores, and SHA-256 values, without private local absolute paths.
- `methods/`: repository URL, revision or archive hash, checkpoint provenance,
  checkpoint SHA-256, implementation status, and limitations.
- `raw_results/`: per-image quality measurements and detection TP/FP/FN and F1
  records. These detection records are not per-box predictions.
- `analysis/`: dataset-balanced summaries, paired 95% confidence intervals,
  cross-dataset ranks, metric-alignment statistics, and two-detector summaries.
- `scripts/`: manifest generation, inference adapters, metric computation,
  bootstrap analysis, and table generation.

## Reproduction boundary

Third-party datasets, source repositories, and model weights are **not**
redistributed. Obtain them from their official sources and verify the recorded
hashes. Dataset references are stored as identifiers and hashes so that licensed
local copies can be resolved without publishing images.

W2WDiff uses 15 sampling steps and seed 12345. HRMamba quality evaluation uses
the official UIEB checkpoint. On this Windows host the official Mamba CUDA
extension could not be compiled, so the released parameterization was executed
with the official pure-PyTorch `selective_scan_ref`; its quality outputs are
retained, while that diagnostic latency is excluded from deployment-speed
ranking.

## Reanalysis from the released records

Python 3.12, NumPy 2.3.5, and SciPy 1.18.1 were used to check the commands
below. Run them from the repository root. No images or model weights are needed
for these analyses; install their two dependencies from `requirements-analysis.txt`.

```text
python -m pip install -r requirements-analysis.txt
python scripts/analyze_expanded_benchmark.py
python scripts/analyze_frontier_track.py
python scripts/build_manuscript_result_tables.py
python scripts/analyze_two_detection_benchmarks.py
python scripts/recompute_trashcan_cluster_bootstrap.py
```

By default these commands regenerate files in `analysis/`. To preserve the
archived outputs, set the `UIE_ANALYSIS_OUTPUT` environment variable to a separate
directory before running all five commands. The table script reads the main
analysis results from that same directory, so run it after the first command.

New model inference is a separate task: the inference adapters require local
dataset paths, third-party source code, checkpoints, and method-specific
environments. This release does not provide a one-command setup for all models.

## Statistical conventions

- Main quality summaries give equal weight to the two paired datasets for
  full-reference scores and to all eight sources for no-reference scores.
  Processing times are pooled per-image means. Main paired intervals use 2,000
  stratified bootstrap replicates, with equal-weight dataset aggregation and
  seed 20261012.
- Frontier summaries are pooled per-image means, not dataset-balanced means.
  Their paired intervals use 2,000 image-level resamples (base seed 20261012).
- The combined detection F1 summary retains the original 2,000 image-level
  resamples: seed 20260721 for TrashCan and seed 20261012 for URPC2020. These
  intervals describe mean per-image F1 changes, not pooled F1 or mAP.
- `analysis/detection/trashcan_video_cluster_bootstrap.csv` separately reports
  5,000 video-cluster resamples, seed 20260722. The full TrashCan sample contains
  1,204 frames in 133 video clusters. Use these intervals when accounting for
  within-video dependence; they are not interchangeable with image-level
  intervals. Both definitions are retained under distinct file names.
- Cross-dataset rank ranges are reported rather than collapsed into a single
  universal score.
- Downstream conclusions are restricted to the stated dataset-detector pairs.
  Detector weights and inference settings are identical for RAW and enhanced
  inputs within each benchmark.

## Public archive status

Version: v1.0.1. Repository:
https://github.com/wwcwfu2025-sudo/uie-evidence-benchmark

Release: https://github.com/wwcwfu2025-sudo/uie-evidence-benchmark/releases/tag/v1.0.1

This maintenance release repairs analysis input paths and adds previously
omitted TrashCan per-image records and video-cluster statistics. It does not
replace the earlier scientific measurements or change the sampling definitions.
The original v1.0.0 release remains available. An archival DOI has not been assigned.

