# Urgency ranking

Prediction scope: `loeo`. This is a gold-label urgency proxy, not an operational utility study.
All orderings use the same eligible, clustered tweets with known agreed labels. Missing labels are excluded
from every ordering. Diversity caps and final category display ordering are applied after this ranking audit.
Gains are 2^relevance−1. Random ordering is the mean of ten recorded seeds per window.
Windows without positive gold have undefined NDCG and are reported separately.

| Model / event | Windows | Scored | No positive gold | Unknown labels | Method | NDCG@10 | NDCG@25 |
|---|---:|---:|---:|---:|---|---:|---:|
| full-s0-loeo_hurricane_harvey / hurricane_harvey | 24 | 18 | 6 | 638 | priority | 0.867 | 0.879 |
| full-s0-loeo_hurricane_harvey / hurricane_harvey | 24 | 18 | 6 | 638 | random | 0.521 | 0.628 |
| full-s0-loeo_hurricane_harvey / hurricane_harvey | 24 | 18 | 6 | 638 | recency | 0.514 | 0.634 |
| full-s0-loeo_hurricane_harvey / hurricane_harvey | 24 | 18 | 6 | 638 | confidence | 0.743 | 0.800 |
| full-s0-loeo_hurricane_maria / hurricane_maria | 54 | 30 | 24 | 549 | priority | 0.867 | 0.913 |
| full-s0-loeo_hurricane_maria / hurricane_maria | 54 | 30 | 24 | 549 | random | 0.668 | 0.760 |
| full-s0-loeo_hurricane_maria / hurricane_maria | 54 | 30 | 24 | 549 | recency | 0.674 | 0.763 |
| full-s0-loeo_hurricane_maria / hurricane_maria | 54 | 30 | 24 | 549 | confidence | 0.788 | 0.831 |
| full-s0-loeo_mexico_earthquake / mexico_earthquake | 17 | 14 | 3 | 228 | priority | 0.901 | 0.930 |
| full-s0-loeo_mexico_earthquake / mexico_earthquake | 17 | 14 | 3 | 228 | random | 0.690 | 0.757 |
| full-s0-loeo_mexico_earthquake / mexico_earthquake | 17 | 14 | 3 | 228 | recency | 0.701 | 0.746 |
| full-s0-loeo_mexico_earthquake / mexico_earthquake | 17 | 14 | 3 | 228 | confidence | 0.821 | 0.859 |
