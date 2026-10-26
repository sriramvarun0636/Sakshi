# B1/B2 leave-one-event-out evaluation

Each seed-zero baseline was trained without the named CrisisMMD event,
calibrated on the fold's development data, and scored on the same frozen
held-out event and agreed gold labels as the full model. Severity is gold-gated.
Macro-F1 averages only classes present in that event's gold labels.

## B1 text only

Frozen folds: **7/7**. The seven-fold estimate is complete.

| Held-out event | Rows | Informative n / F1 | Humanitarian n / F1 |
|---|---:|---:|---:|
| Srilanka Floods | 1,022 | 859 / 0.921 | 726 / 0.661 |
| Hurricane Harvey | 4,435 | 3,157 / 0.849 | 1,954 / 0.735 |
| Hurricane Irma | 4,504 | 2,782 / 0.814 | 1,848 / 0.657 |
| Hurricane Maria | 4,556 | 3,101 / 0.823 | 2,228 / 0.670 |
| Mexico Earthquake | 1,380 | 1,114 / 0.676 | 585 / 0.622 |
| California Wildfires | 1,589 | 1,195 / 0.706 | 511 / 0.573 |
| Iraq Iran Earthquake | 597 | 500 / 0.678 | 227 / 0.686 |

Seven-fold macro-F1 (unweighted mean ± sample SD): informative 0.781 ± 0.095; humanitarian 0.658 ± 0.050.

## B2 image only

Frozen folds: **7/7**. The seven-fold estimate is complete.

| Held-out event | Rows | Informative n / F1 | Humanitarian n / F1 | Severity n / F1 |
|---|---:|---:|---:|---:|
| Srilanka Floods | 1,022 | 859 / 0.925 | 726 / 0.677 | 95 / 0.406 |
| Hurricane Harvey | 4,435 | 3,157 / 0.868 | 1,954 / 0.719 | 892 / 0.544 |
| Hurricane Irma | 4,504 | 2,782 / 0.844 | 1,848 / 0.761 | 795 / 0.545 |
| Hurricane Maria | 4,556 | 3,101 / 0.877 | 2,228 / 0.748 | 862 / 0.456 |
| Mexico Earthquake | 1,380 | 1,114 / 0.849 | 585 / 0.597 | 178 / 0.563 |
| California Wildfires | 1,589 | 1,195 / 0.800 | 511 / 0.595 | 531 / 0.452 |
| Iraq Iran Earthquake | 597 | 500 / 0.814 | 227 / 0.767 | 173 / 0.539 |

Seven-fold macro-F1 (unweighted mean ± sample SD): informative 0.854 ± 0.042; humanitarian 0.695 ± 0.074; severity 0.501 ± 0.061.

The folds vary in size and class mix. These unweighted means describe transfer
across events; they are not pooled sample scores. B3 late fusion is reported separately.
