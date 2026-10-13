# Minimum reproducibility checklist

- [x] Dataset names, splits, image counts, paired status, and selection rule
- [x] Per-image identifiers and SHA-256 values
- [x] Method family and distinct-algorithm counting rule
- [x] Official repository or standardized reimplementation status
- [x] Checkpoint source and SHA-256 where a checkpoint is used
- [x] Software, hardware, precision, batch size, and input-size policy
- [x] Preprocessing, padding, resizing, and output-geometry restoration
- [x] Random seeds and iterative sampling counts
- [x] Metric implementation and direction of preference
- [x] Per-image raw results and explicit failure status
- [x] Aggregation unit and dataset weighting
- [x] Confidence-interval unit, replicate count, and seed
- [x] Cross-dataset rank range and worst-domain behavior
- [x] Detector dataset, checkpoint, classes, and unchanged inference settings
- [x] Negative-transfer and improved-image proportions
- [x] Unsupported operators and compatibility substitutions disclosed
- [ ] Immutable public repository tag/commit
- [ ] Public archive URL or DOI
