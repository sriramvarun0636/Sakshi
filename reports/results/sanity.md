# Sanity audit — test

Requested 100 posts per head, stratified by the frozen model's predicted class (seed 0). Mean estimates use 1000 post-level bootstrap replicates and percentile 95% intervals. Small dev samples are smoke checks, not evidence of explainer quality.

| Head | Sampled | Explained | Gated/unavailable | Flags |
|---|---:|---:|---:|---|
| informative | 100 | 100 | 0 | {'ig_unconverged': 4, 'head:ig_unconverged': 6, 'cross_attention:ig_unconverged': 6, 'gate:ig_unconverged': 7, 'head:no_positive_attribution': 1, 'cross_attention:no_positive_attribution': 1} |
| humanitarian | 100 | 80 | 20 | {'ig_unconverged': 3, 'head:ig_unconverged': 6, 'gate:ig_unconverged': 11, 'cross_attention:ig_unconverged': 8, 'head:no_positive_attribution': 1, 'cross_attention:no_positive_attribution': 1} |
| severity | 100 | 87 | 13 | {} |
| disaster_type | 100 | 100 | 0 | {} |

Cascade: head MLP → last cross-attention block → gate. The original predicted target is held fixed. Constant or length-one vectors have undefined Spearman correlation and are excluded from the mean, with their counts retained. Gate randomisation does not affect the image-only head readout.

| Head | Modality | Stage | Defined | Undefined | Mean Spearman rho [95% CI] |
|---|---|---|---:|---:|---|
| informative | text | head | 100 | 0 | -0.2225 [-0.2960, -0.1454] |
| informative | text | cross_attention | 100 | 0 | -0.1587 [-0.2259, -0.0929] |
| informative | text | gate | 100 | 0 | -0.1615 [-0.2237, -0.0962] |
| informative | image | head | 100 | 0 | -0.0206 [-0.0698, 0.0285] |
| informative | image | cross_attention | 100 | 0 | -0.0335 [-0.0814, 0.0170] |
| informative | image | gate | 100 | 0 | 0.0180 [-0.0262, 0.0648] |
| humanitarian | text | head | 80 | 0 | 0.1091 [-0.0243, 0.2359] |
| humanitarian | text | cross_attention | 80 | 0 | 0.0482 [-0.0586, 0.1573] |
| humanitarian | text | gate | 80 | 0 | 0.0625 [-0.0496, 0.1712] |
| humanitarian | image | head | 80 | 0 | -0.0004 [-0.0522, 0.0540] |
| humanitarian | image | cross_attention | 80 | 0 | -0.0132 [-0.0639, 0.0375] |
| humanitarian | image | gate | 80 | 0 | -0.0071 [-0.0518, 0.0373] |
| severity | image | head | 87 | 0 | -0.1095 [-0.1602, -0.0614] |
| severity | image | cross_attention | 87 | 0 | -0.1182 [-0.1606, -0.0735] |
| severity | image | gate | 87 | 0 | -0.1182 [-0.1606, -0.0735] |
| disaster_type | image | head | 100 | 0 | -0.0953 [-0.1480, -0.0463] |
| disaster_type | image | cross_attention | 100 | 0 | -0.0662 [-0.1146, -0.0202] |
| disaster_type | image | gate | 100 | 0 | -0.0662 [-0.1146, -0.0202] |
