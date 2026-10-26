# B1 text-only and B2 image-only · three frozen seeds

Both baselines were trained independently for seeds 0, 1 and 2, selected and calibrated
on development rows, and evaluated once on the official agreed-label test selectors.
The six restored runs have hash-verified `evaluation_complete.json` markers. Each value
below is macro-F1; the last column is the mean ± sample standard deviation over seeds.

| Model / selector | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |
|---|---:|---:|---:|---:|
| B1 informative | 0.828 | 0.827 | 0.808 | **0.821 ± 0.011** |
| B1 humanitarian | 0.632 | 0.647 | 0.643 | **0.641 ± 0.008** |
| B2 informative | 0.880 | 0.884 | 0.889 | **0.884 ± 0.004** |
| B2 humanitarian | 0.728 | 0.757 | 0.750 | **0.745 ± 0.016** |
| B2 severity, gold-gated | 0.612 | 0.549 | 0.560 | **0.574 ± 0.034** |
| B2 severity, predicted-gated | 0.551 | 0.503 | 0.517 | **0.523 ± 0.025** |
| B2 DMD severity | 0.388 | 0.426 | 0.433 | **0.416 ± 0.024** |
| B2 disaster type, combined | 0.849 | 0.860 | 0.855 | **0.855 ± 0.006** |

B1 has no image-only severity or disaster-type heads. B2's AIDR-DT and DMD
disaster-type subsets average **0.785 ± 0.006** and **0.756 ± 0.013**, respectively.
All six hierarchical runs have zero contradiction rate. The informative and
humanitarian selectors contain 1,534 and 955 agreed-label rows. The corresponding
full-model three-seed means are 0.919 and 0.822; see
[`classifier_three_seed.md`](classifier_three_seed.md). That comparison is descriptive
until paired, seed-aware intervals are computed.

The original per-modality label evaluation is currently frozen only for seed 0 and is
reported separately in [`baselines_seed0.md`](baselines_seed0.md). B3 late fusion is
frozen only for seed 0; its remaining seeds must derive weights from their own
calibrated development predictions. The seven-event B1/B2 LOEO matrix is underway and
is not represented by these standard-split scores.
