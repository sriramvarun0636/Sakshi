# Faithfulness audit — dev

Development check only. These are not the official 500-post faithfulness or 100-post sanity results.

Requested 1 posts per head, stratified by the frozen model's predicted class (seed 0). Mean estimates use 1000 post-level bootstrap replicates and percentile 95% intervals. Small dev samples are smoke checks, not evidence of explainer quality.

| Head | Sampled | Explained | Gated/unavailable | Flags |
|---|---:|---:|---:|---|
| informative | 1 | 1 | 0 | {} |
| humanitarian | 1 | 1 | 0 | {'ig_unconverged': 1} |
| severity | 1 | 1 | 0 | {} |
| disaster_type | 1 | 1 | 0 | {} |

Uncalibrated probabilities; other modality held fixed. Text uses `<mask>` and preserves special tokens; images zero normalized 16×16 patches and rerun CLIP. Random baselines average seeds 0–4 per post. Comprehensiveness: higher is better; sufficiency: lower is better. Negative differences are retained.

| Head | Modality | Method | Posts | AOPC comprehensiveness [95% CI] | AOPC sufficiency [95% CI] |
|---|---|---|---:|---|---|
| informative | text | explainer | 1 | 0.4343 [0.4343, 0.4343] | 0.2760 [0.2760, 0.2760] |
| informative | text | random | 1 | 0.0323 [0.0323, 0.0323] | 0.4406 [0.4406, 0.4406] |
| informative | image | explainer | 1 | 0.0302 [0.0302, 0.0302] | 0.6163 [0.6163, 0.6163] |
| informative | image | random | 1 | 0.0180 [0.0180, 0.0180] | 0.5695 [0.5695, 0.5695] |
| humanitarian | text | explainer | 1 | 0.1812 [0.1812, 0.1812] | 0.5330 [0.5330, 0.5330] |
| humanitarian | text | random | 1 | 0.0196 [0.0196, 0.0196] | 0.6081 [0.6081, 0.6081] |
| humanitarian | image | explainer | 1 | 0.0131 [0.0131, 0.0131] | 0.0350 [0.0350, 0.0350] |
| humanitarian | image | random | 1 | 0.0096 [0.0096, 0.0096] | 0.0535 [0.0535, 0.0535] |
| severity | image | explainer | 1 | 0.0812 [0.0812, 0.0812] | -0.3404 [-0.3404, -0.3404] |
| severity | image | random | 1 | 0.0369 [0.0369, 0.0369] | -0.3273 [-0.3273, -0.3273] |
| disaster_type | image | explainer | 1 | 0.0096 [0.0096, 0.0096] | 0.5787 [0.5787, 0.5787] |
| disaster_type | image | random | 1 | 0.0046 [0.0046, 0.0046] | 0.6929 [0.6929, 0.6929] |

Comparisons that fail to beat random are retained. Do not retune explainers from test audit results.
