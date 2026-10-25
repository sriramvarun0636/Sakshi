# Frozen official test: full seed 0 versus B5 CLIP zero-shot

Full run: `full-s0-20261024-062724`; B5 run: `clip_zeroshot-s0-20261023-160328`.
Paired item bootstrap: 1,000 draws, seed 0; two-sided p-values.
Each row uses the exact same official samples for both models. Positive Δ favours
the full model. B5 uses only images; the full model can use text and images.

| Task | Rows | Full macro-F1 | B5 macro-F1 | Δ [95% CI] | p |
|---|---:|---:|---:|---:|---:|
| Informativeness | 1,534 | 0.919 | 0.830 | +0.089 [+0.070, +0.110] | <0.002 |
| Humanitarian | 955 | 0.791 | 0.564 | +0.227 [+0.177, +0.275] | <0.002 |
| Severity, gold gated | 529 | 0.590 | 0.382 | +0.208 [+0.150, +0.265] | <0.002 |
| Severity, predicted gated | 529 | 0.582 | 0.377 | +0.205 [+0.148, +0.264] | <0.002 |
| Severity, DMD | 1,138 | 0.445 | 0.317 | +0.128 [+0.082, +0.169] | <0.002 |
| Disaster type | 3,213 | 0.859 | 0.655 | +0.204 [+0.189, +0.218] | <0.002 |

This is a one-seed comparison, not the planned three-seed E2 result. The six
reported comparisons have no multiple-comparison correction. With 1,000 draws,
zero empirical tail counts are reported as p < 0.002 rather than p = 0.
The paired bootstrap captures test-item sampling for these fixed runs; it does not measure training-seed
variation or make the models' input modalities equivalent.

## Artifact identities

- Full predictions SHA256: `349602740940bd155f94cfbb203a33f5436fc4ae63fa473f8a0f7f0f8f3f5a8f`
- B5 predictions SHA256: `7bd4c838451c58a29f8ad42f73bea321ceacaae53c80d89f71794af110ee1189`
- B5 metrics SHA256: `2b7c7b39975262907a68433674b7051a8891ecd936b35674f5f3b06240535922`
