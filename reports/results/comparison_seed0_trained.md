# Frozen official test: full seed 0 versus B1–B3

All rows use the same agreed-label official test sample IDs. Differences are full minus baseline macro-F1. Paired item bootstrap: 1,000 draws, seed 0, percentile 95% interval, two-sided empirical p. Zero tail counts are shown as p < 0.002, not p = 0. The six comparisons are exploratory and have no multiple-comparison correction. They measure test-item variation for fixed seed-zero checkpoints, not variation across training seeds.

| Head | Rows | Baseline | Full F1 | Baseline F1 | Full − baseline [95% CI] | p |
|---|---:|---|---:|---:|---:|---:|
| Informativeness | 1,534 | B1 text | 0.919 | 0.828 | +0.091 [+0.070, +0.112] | <0.002 |
| Informativeness | 1,534 | B2 image | 0.919 | 0.880 | +0.038 [+0.023, +0.057] | <0.002 |
| Informativeness | 1,534 | B3 late fusion | 0.919 | 0.922 | −0.004 [−0.016, +0.009] | 0.548 |
| Humanitarian | 955 | B1 text | 0.791 | 0.632 | +0.159 [+0.119, +0.204] | <0.002 |
| Humanitarian | 955 | B2 image | 0.791 | 0.728 | +0.063 [+0.003, +0.127] | 0.040 |
| Humanitarian | 955 | B3 late fusion | 0.791 | 0.871 | −0.080 [−0.133, −0.022] | 0.006 |

B3 tuned informative and conditional humanitarian mixing weights separately on 2,674 agreed development rows (1,621 with humanitarian labels), over the fixed grid {0, 0.1, …, 1}. Both selected weights were 0.5. Its abstention thresholds, also selected on dev, were 0.50 informative and 0.55 humanitarian at 0.904 dev coverage. Only then did it read the frozen official test predictions. B3 inherits severity and disaster type from B2 and preserves zero hierarchy contradictions.

The B3 humanitarian gain over full seed 0 is a result to test across all three seeds. It does not establish a general advantage by itself. B1 and B2 trained on their own modality labels; the shared agreed-label selector makes the comparisons aligned, while the single-modality inputs and different training targets remain part of the baseline definition.

## Prediction identities (SHA256)

- Full: `349602740940bd155f94cfbb203a33f5436fc4ae63fa473f8a0f7f0f8f3f5a8f`
- B1: `ce5600c8f21fe9c1bd753eea95012ac9323e03fab1d5d49ef18f5966e9eee258`
- B2: `db81d4c5b9796d39790cb0cacdbb6367ff57664317d50cb2edf282d3e50bb5c6`
- B3: `7b24daf4f0261df59c250ccf2db11d65bdb50f7c891360379d9e907c1aa79b9e`
- B3 dev weights: `82ad58a9530a54814e0d96e14318769ef1083a863b35c8c1eb2be42721b4bdcf`

The B1/B2 official prediction markers and B3 `complete.json` verify the stored outputs. The B3 request hashes both frozen sources, the manifest and feature cache, the implementation, and the bootstrap count.
