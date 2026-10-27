# Faithfulness audit — test

Requested 500 posts per head, stratified by the frozen model's predicted class (seed 0). Mean estimates use 1000 post-level bootstrap replicates and percentile 95% intervals. Small dev samples are smoke checks, not evidence of explainer quality.

| Head | Sampled | Explained | Gated/unavailable | Flags |
|---|---:|---:|---:|---|
| informative | 500 | 500 | 0 | {'ig_unconverged': 18} |
| humanitarian | 500 | 369 | 131 | {'ig_unconverged': 6} |
| severity | 500 | 487 | 13 | {} |
| disaster_type | 500 | 500 | 0 | {} |

Uncalibrated probabilities; other modality held fixed. Text uses `<mask>` and preserves special tokens; images zero normalized 16×16 patches and rerun CLIP. Random baselines average seeds 0–4 per post. Comprehensiveness: higher is better; sufficiency: lower is better. Negative differences are retained.

| Head | Modality | Method | Posts | AOPC comprehensiveness [95% CI] | AOPC sufficiency [95% CI] |
|---|---|---|---:|---|---|
| informative | text | explainer | 500 | 0.0810 [0.0713, 0.0910] | 0.1179 [0.0995, 0.1365] |
| informative | text | random | 500 | 0.0136 [0.0096, 0.0175] | 0.1522 [0.1328, 0.1719] |
| informative | image | explainer | 500 | 0.0454 [0.0389, 0.0519] | 0.1641 [0.1442, 0.1859] |
| informative | image | random | 500 | 0.0160 [0.0115, 0.0203] | 0.1987 [0.1766, 0.2206] |
| humanitarian | text | explainer | 369 | 0.1546 [0.1334, 0.1745] | 0.1461 [0.1275, 0.1648] |
| humanitarian | text | random | 369 | 0.0345 [0.0275, 0.0417] | 0.2029 [0.1813, 0.2240] |
| humanitarian | image | explainer | 369 | 0.0522 [0.0429, 0.0610] | 0.2067 [0.1897, 0.2242] |
| humanitarian | image | random | 369 | 0.0228 [0.0169, 0.0284] | 0.2762 [0.2569, 0.2955] |
| severity | image | explainer | 487 | 0.1251 [0.1160, 0.1342] | 0.3946 [0.3719, 0.4164] |
| severity | image | random | 487 | 0.0476 [0.0418, 0.0534] | 0.4495 [0.4263, 0.4714] |
| disaster_type | image | explainer | 500 | 0.1508 [0.1392, 0.1625] | 0.4813 [0.4576, 0.5049] |
| disaster_type | image | random | 500 | 0.0677 [0.0596, 0.0754] | 0.5498 [0.5238, 0.5752] |

Comparisons that fail to beat random are retained. Do not retune explainers from test audit results.
