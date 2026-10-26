# B3 late fusion · leave one event out

Each fold tunes informative and conditional-humanitarian mixing weights on its own
development events, then scores the held-out event. Severity comes unchanged from B2.
Macro-F1 uses the same agreed-gold fold selectors as the full, B1 and B2 models.

Frozen folds: **7/7**. The seven-fold estimate is complete.

| Held-out event | Rows | Informative n / F1 | Humanitarian n / F1 | Severity n / F1 | Text weight I / H |
|---|---:|---:|---:|---:|---:|
| Srilanka Floods | 1,022 | 859 / 0.945 | 726 / 0.753 | 95 / 0.406 | 0.4 / 0.5 |
| Hurricane Harvey | 4,435 | 3,157 / 0.915 | 1,954 / 0.829 | 892 / 0.544 | 0.4 / 0.5 |
| Hurricane Irma | 4,504 | 2,782 / 0.888 | 1,848 / 0.748 | 795 / 0.545 | 0.4 / 0.5 |
| Hurricane Maria | 4,556 | 3,101 / 0.915 | 2,228 / 0.806 | 862 / 0.456 | 0.5 / 0.4 |
| Mexico Earthquake | 1,380 | 1,114 / 0.875 | 585 / 0.759 | 178 / 0.563 | 0.5 / 0.5 |
| California Wildfires | 1,589 | 1,195 / 0.844 | 511 / 0.687 | 531 / 0.452 | 0.5 / 0.5 |
| Iraq Iran Earthquake | 597 | 500 / 0.868 | 227 / 0.835 | 173 / 0.539 | 0.5 / 0.4 |

Seven-fold macro-F1 (unweighted mean ± sample SD): informative 0.893 ± 0.034; humanitarian 0.774 ± 0.053; severity 0.501 ± 0.061.

Fold sizes and class mixes differ; these are unweighted event-transfer estimates,
not pooled post-level scores. B3 severity is inherited from B2 rather than fused.
