# Urgency ranking — demonstration

Prediction scope: `demo`. This is a gold-label urgency proxy, not an operational utility study.
All orderings use the same eligible, clustered tweets with known agreed labels. Missing labels are excluded
from every ordering. Diversity caps and final category display ordering are applied after this ranking audit.
Gains are 2^relevance−1. Random ordering is the mean of ten recorded seeds per window.
Windows without positive gold have undefined NDCG and are reported separately.

| Model / event | Windows | Scored | No positive gold | Unknown labels | Method | NDCG@10 | NDCG@25 |
|---|---:|---:|---:|---:|---|---:|---:|
| clip_zeroshot-s0 / hurricane_harvey | 24 | 18 | 6 | 1524 | priority | 0.628 | 0.678 |
| clip_zeroshot-s0 / hurricane_harvey | 24 | 18 | 6 | 1524 | random | 0.392 | 0.511 |
| clip_zeroshot-s0 / hurricane_harvey | 24 | 18 | 6 | 1524 | recency | 0.422 | 0.548 |
| clip_zeroshot-s0 / hurricane_harvey | 24 | 18 | 6 | 1524 | confidence | 0.406 | 0.527 |
| clip_zeroshot-s0 / hurricane_maria | 54 | 30 | 24 | 1232 | priority | 0.634 | 0.718 |
| clip_zeroshot-s0 / hurricane_maria | 54 | 30 | 24 | 1232 | random | 0.418 | 0.563 |
| clip_zeroshot-s0 / hurricane_maria | 54 | 30 | 24 | 1232 | recency | 0.442 | 0.594 |
| clip_zeroshot-s0 / hurricane_maria | 54 | 30 | 24 | 1232 | confidence | 0.535 | 0.654 |
| clip_zeroshot-s0 / mexico_earthquake | 17 | 14 | 3 | 509 | priority | 0.804 | 0.836 |
| clip_zeroshot-s0 / mexico_earthquake | 17 | 14 | 3 | 509 | random | 0.612 | 0.688 |
| clip_zeroshot-s0 / mexico_earthquake | 17 | 14 | 3 | 509 | recency | 0.622 | 0.677 |
| clip_zeroshot-s0 / mexico_earthquake | 17 | 14 | 3 | 509 | confidence | 0.764 | 0.813 |
| clip_zeroshot-s0 / srilanka_floods | 34 | 17 | 17 | 171 | priority | 0.893 | 0.906 |
| clip_zeroshot-s0 / srilanka_floods | 34 | 17 | 17 | 171 | random | 0.753 | 0.771 |
| clip_zeroshot-s0 / srilanka_floods | 34 | 17 | 17 | 171 | recency | 0.656 | 0.675 |
| clip_zeroshot-s0 / srilanka_floods | 34 | 17 | 17 | 171 | confidence | 0.804 | 0.814 |
