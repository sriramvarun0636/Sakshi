# Ablations and single-task baselines · three frozen seeds

Every complete variant is compared with the corresponding full-model seed on identical official test posts.
Δ is variant minus full macro-F1. Its 95% interval resamples test posts together across all three seeds
(1,000 paired draws); seed SD describes variation among the three trained runs.
The interval captures test-post sampling, not uncertainty over all possible training seeds.
B4 fits one head at a time with no auxiliary heads; its humanitarian head is flat five-class.

## A1 no cross-attention

Frozen seeds: **3/3**.

| Selector | n | Full F1 | Variant F1 | Δ [95% CI] | Seed SD of Δ |
|---|---:|---:|---:|---:|---:|
| Informativeness | 1,534 | 0.919 | 0.921 | +0.002 [-0.006, +0.010] | 0.004 |
| Humanitarian | 955 | 0.822 | 0.822 | +0.000 [-0.026, +0.025] | 0.032 |
| Severity, gold gated | 529 | 0.555 | 0.557 | +0.002 [-0.028, +0.034] | 0.023 |
| Severity, predicted gated | 529 | 0.538 | 0.539 | +0.001 [-0.028, +0.032] | 0.044 |
| Severity, DMD | 1,138 | 0.443 | 0.427 | -0.016 [-0.037, +0.005] | 0.008 |
| Disaster type | 3,213 | 0.856 | 0.839 | -0.017 [-0.024, -0.009] | 0.010 |

## A2 no modality gate

Frozen seeds: **3/3**.

| Selector | n | Full F1 | Variant F1 | Δ [95% CI] | Seed SD of Δ |
|---|---:|---:|---:|---:|---:|
| Informativeness | 1,534 | 0.919 | 0.914 | -0.006 [-0.012, +0.000] | 0.004 |
| Humanitarian | 955 | 0.822 | 0.809 | -0.013 [-0.037, +0.013] | 0.027 |
| Severity, gold gated | 529 | 0.555 | 0.575 | +0.020 [-0.010, +0.052] | 0.019 |
| Severity, predicted gated | 529 | 0.538 | 0.560 | +0.022 [-0.009, +0.057] | 0.023 |
| Severity, DMD | 1,138 | 0.443 | 0.394 | -0.049 [-0.071, -0.026] | 0.025 |
| Disaster type | 3,213 | 0.856 | 0.860 | +0.004 [-0.001, +0.010] | 0.016 |

## A3 no modality dropout

Frozen seeds: **3/3**.

| Selector | n | Full F1 | Variant F1 | Δ [95% CI] | Seed SD of Δ |
|---|---:|---:|---:|---:|---:|
| Informativeness | 1,534 | 0.919 | 0.914 | -0.005 [-0.012, +0.002] | 0.005 |
| Humanitarian | 955 | 0.822 | 0.823 | +0.001 [-0.017, +0.019] | 0.036 |
| Severity, gold gated | 529 | 0.555 | 0.562 | +0.007 [-0.019, +0.034] | 0.049 |
| Severity, predicted gated | 529 | 0.538 | 0.542 | +0.004 [-0.025, +0.033] | 0.060 |
| Severity, DMD | 1,138 | 0.443 | 0.437 | -0.005 [-0.024, +0.013] | 0.023 |
| Disaster type | 3,213 | 0.856 | 0.856 | +0.000 [-0.005, +0.005] | 0.006 |

## A4 flat humanitarian head

Frozen seeds: **3/3**.

| Selector | n | Full F1 | Variant F1 | Δ [95% CI] | Seed SD of Δ |
|---|---:|---:|---:|---:|---:|
| Informativeness | 1,534 | 0.919 | 0.919 | +0.000 [-0.007, +0.006] | 0.001 |
| Humanitarian | 955 | 0.822 | 0.834 | +0.012 [-0.010, +0.035] | 0.029 |
| Severity, gold gated | 529 | 0.555 | 0.567 | +0.012 [-0.017, +0.044] | 0.030 |
| Severity, predicted gated | 529 | 0.538 | 0.567 | +0.029 [-0.003, +0.063] | 0.039 |
| Severity, DMD | 1,138 | 0.443 | 0.682 | +0.240 [+0.221, +0.261] | 0.025 |
| Disaster type | 3,213 | 0.856 | 0.851 | -0.005 [-0.012, +0.002] | 0.006 |
| Severity, DMD, equal coverage | 1,138 | 0.699 | 0.682 | -0.016 [-0.033, +0.002] | 0.024 |

A4 removes the informative/humanitarian constraint and the severity gate.
Its hierarchy contradiction rate is **0.367 ± 0.017** across seeds (full model: 0).
The ordinary DMD severity row includes the change from gated to ungated coverage.
The equal-coverage row forces the severity argmax for both models on all DMD posts.

## A5 no CIB training

Frozen seeds: **3/3**.

| Selector | n | Full F1 | Variant F1 | Δ [95% CI] | Seed SD of Δ |
|---|---:|---:|---:|---:|---:|
| Informativeness | 1,534 | 0.919 | 0.914 | -0.005 [-0.013, +0.002] | 0.004 |
| Humanitarian | 955 | 0.822 | 0.819 | -0.004 [-0.026, +0.021] | 0.034 |
| Severity, gold gated | 529 | 0.555 | 0.569 | +0.014 [-0.021, +0.050] | 0.058 |
| Severity, predicted gated | 529 | 0.538 | 0.561 | +0.022 [-0.015, +0.060] | 0.062 |
| Severity, DMD | 1,138 | 0.443 | 0.355 | -0.087 [-0.114, -0.059] | 0.016 |

A5 has no disaster-type head. Its DMD severity score measures transfer to CIB images
after all CIB training data were removed.

## B4 informative only

Frozen seeds: **3/3**.

| Selector | n | Full F1 | Variant F1 | Δ [95% CI] | Seed SD of Δ |
|---|---:|---:|---:|---:|---:|
| Informativeness | 1,534 | 0.919 | 0.927 | +0.007 [-0.000, +0.015] | 0.004 |

## B4 humanitarian only

Frozen seeds: **3/3**.

| Selector | n | Full F1 | Variant F1 | Δ [95% CI] | Seed SD of Δ |
|---|---:|---:|---:|---:|---:|
| Humanitarian | 955 | 0.822 | 0.870 | +0.048 [+0.009, +0.088] | 0.042 |

## B4 severity only

Frozen seeds: **3/3**.

| Selector | n | Full F1 | Variant F1 | Δ [95% CI] | Seed SD of Δ |
|---|---:|---:|---:|---:|---:|
| Severity, gold gated | 529 | 0.555 | 0.555 | -0.000 [-0.035, +0.033] | 0.027 |
| Severity, predicted gated | 529 | 0.538 | 0.555 | +0.016 [-0.021, +0.051] | 0.036 |
| Severity, DMD | 1,138 | 0.443 | 0.676 | +0.234 [+0.212, +0.257] | 0.038 |
| Severity, DMD, equal coverage | 1,138 | 0.699 | 0.676 | -0.023 [-0.042, -0.001] | 0.038 |

The severity-only model has no informative or humanitarian head to gate its outputs.
The ordinary DMD row therefore also reflects coverage. The equal-coverage row
forces the severity argmax for both models on all DMD posts.

## B4 disaster type only

Frozen seeds: **3/3**.

| Selector | n | Full F1 | Variant F1 | Δ [95% CI] | Seed SD of Δ |
|---|---:|---:|---:|---:|---:|
| Disaster type | 3,213 | 0.856 | 0.858 | +0.003 [-0.003, +0.009] | 0.004 |

The experiments share fixed official selectors, but the many reported effects have no multiple-comparison correction.
