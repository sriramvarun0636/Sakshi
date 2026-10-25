# Full classifier: three frozen standard seeds

All three runs used the same fixed standard train/dev/test split and the published-compatible official selectors. Each seed was chosen independently of test results, calibrated on development data, then evaluated once with frozen input and output hashes. Seed 2 was restored from a private Kaggle T4 training export and its final official evaluation was completed locally on CPU; the scoring code, selectors and bootstrap count match seeds 0 and 1.

| Official macro-F1 | Rows | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| Informativeness | 1,534 | 0.919 | 0.920 | 0.919 | 0.919 ± 0.001 |
| Humanitarian | 955 | 0.791 | 0.835 | 0.841 | 0.822 ± 0.027 |
| Severity, gold gated | 529 | 0.590 | 0.529 | 0.546 | 0.555 ± 0.032 |
| Severity, predicted gated | 529 | 0.582 | 0.502 | 0.532 | 0.538 ± 0.041 |
| Severity, DMD | 1,138 | 0.445 | 0.450 | 0.433 | 0.443 ± 0.009 |
| Disaster type, combined | 3,213 | 0.859 | 0.851 | 0.856 | 0.856 ± 0.004 |

All three hierarchical contradiction rates were zero. Humanitarian retained coverage was 0.917, 0.908 and 0.913 for seeds 0–2. Per-seed reports include 1,000-item percentile confidence intervals, per-class scores, confusion matrices, calibration and abstention results: [seed 0](classifier.md), [seed 1](classifier_seed1.md), [seed 2](classifier_seed2.md). Seed-to-seed SD measures variation across these three fits; it is not a confidence interval for generalization to new disasters. The held-out-event LOEO study addresses that separate question.

The weakest recurring head is DMD severity (mean macro-F1 0.443), and severity on the 529 gold-gated CrisisMMD images varies more across seeds than informativeness or disaster type. These limits should remain visible in the dashboard and discussion.

## Frozen runs

- `runs/classifier/full-s0-20261024-062724/`
- `runs/classifier/full-s1-20261025-062842/`
- `runs/classifier/full-s2-20261025-143538/`

Each directory contains `evaluation_request.json`, `evaluation_complete.json`, `predictions_official.parquet` and `metrics_test.json`. The request records hashes of the checkpoint, calibration, manifest, feature cache and bootstrap count; the completion marker hashes the predictions and metrics.
