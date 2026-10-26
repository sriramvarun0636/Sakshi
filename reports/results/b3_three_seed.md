# B3 late fusion · standard split

B3 tunes informative and conditional-humanitarian mixing weights separately on each seed's
agreed development rows, then scores the frozen official test selectors. Severity and disaster
type predictions come from that seed's B2 image model.

Frozen seeds: **3/3**.

Δ is B3 minus full macro-F1. The 95% interval resamples the same test posts across
all three seeds (1,000 paired draws); seed SD describes the three run effects.

| Selector | n | Full F1 | B3 F1 | Δ [95% CI] | Seed SD of Δ |
|---|---:|---:|---:|---:|---:|
| Informativeness | 1,534 | 0.919 | 0.917 | -0.002 [-0.010, +0.006] | 0.006 |
| Humanitarian | 955 | 0.822 | 0.843 | +0.021 [-0.008, +0.049] | 0.052 |
| Severity, gold gated | 529 | 0.555 | 0.574 | +0.019 [-0.012, +0.049] | 0.004 |
| Severity, predicted gated | 529 | 0.538 | 0.524 | -0.014 [-0.051, +0.022] | 0.020 |
| Severity, DMD | 1,138 | 0.443 | 0.416 | -0.027 [-0.045, -0.010] | 0.028 |
| Disaster type | 3,213 | 0.856 | 0.855 | -0.001 [-0.007, +0.005] | 0.010 |

## Development choices

| Seed | Informative text weight | Conditional-humanitarian text weight | Dev rows |
|---:|---:|---:|---:|
| 0 | 0.5 | 0.5 | 2,674 |
| 1 | 0.5 | 0.4 | 2,674 |
| 2 | 0.4 | 0.5 | 2,674 |

The intervals capture test-post sampling for these fixed runs, not every possible
training seed. No multiple-comparison correction was applied.
