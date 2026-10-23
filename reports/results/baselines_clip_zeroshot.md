# B5 — CLIP zero-shot, image only

The frozen CLIP ViT-B/16 checkpoint and fixed Appendix E.8 prompts were applied to the cached images.
Each class vector is the mean of its unit-normalised prompt embeddings; logits are 100×cosine.
The final humanitarian and severity decisions use the same hierarchy as the full classifier.
No labels from the official test sets were used for prompts, tuning or model selection.

Run: `/Users/sriramvarundittakavi/Documents/GitHub/Sakshi/runs/baselines/clip_zeroshot-s0-20261023-160328`. Checkpoint revision: `57c216476eefef5ab752ec549e440a49ae4ae5f3`.

| Official task | Macro-F1 | 95% bootstrap CI | ECE |
|---|---:|---:|---:|
| Informativeness | 0.830 | [0.810, 0.849] | 0.016 |
| Humanitarian | 0.564 | [0.530, 0.595] | 0.040 |
| Severity, gold-gated | 0.382 | [0.343, 0.425] | 0.176 |
| Severity, predicted-gated | 0.377 | [0.334, 0.419] | 0.176 |
| Severity, DMD | 0.317 | [0.283, 0.350] | 0.048 |
| Disaster type | 0.655 | [0.641, 0.670] | 0.069 |

Official selector sizes: informative 1,534, humanitarian 955, severity 529, severity_dmd 1,138, disaster_type 3,213.
Hierarchy contradiction rate: 0.000.
Predicted-gated severity coverage: 0.970.
This is a zero-shot image baseline; it is not a trained SAKSHI classifier result.
