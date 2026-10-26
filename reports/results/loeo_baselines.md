# B1/B2 leave-one-event-out evaluation

Each seed-zero baseline was trained without the named CrisisMMD event,
calibrated on the fold's development data, and scored on the same frozen
held-out event and agreed gold labels as the full model. Severity is gold-gated.
Macro-F1 averages only classes present in that event's gold labels.

## B1 text only

Frozen folds: **1/7**. Missing: srilanka_floods, hurricane_irma, hurricane_maria, mexico_earthquake, california_wildfires, iraq_iran_earthquake.

| Held-out event | Rows | Informative n / F1 | Humanitarian n / F1 |
|---|---:|---:|---:|
| Hurricane Harvey | 4,435 | 3,157 / 0.849 | 1,954 / 0.735 |

## B2 image only

Frozen folds: **2/7**. Missing: srilanka_floods, hurricane_irma, mexico_earthquake, california_wildfires, iraq_iran_earthquake.

| Held-out event | Rows | Informative n / F1 | Humanitarian n / F1 | Severity n / F1 |
|---|---:|---:|---:|---:|
| Hurricane Harvey | 4,435 | 3,157 / 0.868 | 1,954 / 0.719 | 892 / 0.544 |
| Hurricane Maria | 4,556 | 3,101 / 0.877 | 2,228 / 0.748 | 862 / 0.456 |

The folds vary in size and class mix. These unweighted means describe transfer
across events; they are not pooled sample scores. B3 late fusion is reported separately.
