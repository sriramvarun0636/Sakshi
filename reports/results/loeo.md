# Full-model leave-one-event-out evaluation

Each row is seed 0 trained without the named CrisisMMD event in train or dev,
calibrated on that fold's dev rows, then evaluated once on its frozen held-out
event predictions. Macro-F1 uses only classes present in that event's gold labels.
Severity is gold-gated. Disaster type has no CrisisMMD LOEO gold labels.

Frozen folds: **6/7**. Still missing: srilanka_floods.

| Held-out event | Rows | Informative n / F1 | Humanitarian n / F1 | Severity n / F1 |
|---|---:|---:|---:|---:|
| Hurricane Harvey | 4,435 | 3,157 / 0.915 | 1,954 / 0.836 | 892 / 0.534 |
| Hurricane Irma | 4,504 | 2,782 / 0.912 | 1,848 / 0.788 | 795 / 0.558 |
| Hurricane Maria | 4,556 | 3,101 / 0.907 | 2,228 / 0.872 | 862 / 0.569 |
| Mexico Earthquake | 1,380 | 1,114 / 0.835 | 585 / 0.755 | 178 / 0.606 |
| California Wildfires | 1,589 | 1,195 / 0.835 | 511 / 0.710 | 531 / 0.427 |
| Iraq Iran Earthquake | 597 | 500 / 0.863 | 227 / 0.838 | 173 / 0.602 |

The event folds vary in sample count and class mix. Their unweighted mean estimates
transfer to a new event, not the pooled performance on all held-out samples.
Individual fold JSON files preserve per-class scores, present classes, 1,000-draw
bootstrap intervals and both severity gates. B1/B2/B3 fold comparisons remain separate.
