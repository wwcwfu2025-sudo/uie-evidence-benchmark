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
  selection scores, and SHA-256 values. Local absolute paths must be removed.
- `methods/`: repository URL, revision or archive hash, checkpoint provenance,
  checkpoint SHA-256, implementation status, and limitations.
- `raw_results/`: per-image quality and detection results, including failed rows.
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

## Statistical conventions

- Dataset-balanced means give each of the eight sources equal weight.
- Paired intervals use 2,000 bootstrap replicates with seed 20261012.
- Cross-dataset rank ranges are reported rather than collapsed into a single
  universal score.
- Downstream conclusions are restricted to the stated dataset-detector pairs.
  Detector weights and inference settings are identical for RAW and enhanced
  inputs within each benchmark.

## Public archive status

Version: v1.0. Repository:
https://github.com/wwcwfu2025-sudo/uie-evidence-benchmark

An archival DOI has not yet been assigned. A DOI will be added after the first
versioned release is archived.

