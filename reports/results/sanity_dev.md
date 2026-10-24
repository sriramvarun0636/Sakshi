# Sanity audit — dev

Development check only. These are not the official 500-post faithfulness or 100-post sanity results.

Requested 1 posts per head, stratified by the frozen model's predicted class (seed 0). Mean estimates use 1000 post-level bootstrap replicates and percentile 95% intervals. Small dev samples are smoke checks, not evidence of explainer quality.

| Head | Sampled | Explained | Gated/unavailable | Flags |
|---|---:|---:|---:|---|
| informative | 1 | 1 | 0 | {} |
| humanitarian | 1 | 1 | 0 | {'ig_unconverged': 1, 'head:ig_unconverged': 1, 'cross_attention:ig_unconverged': 1} |
| severity | 1 | 1 | 0 | {} |
| disaster_type | 1 | 1 | 0 | {} |

Cascade: head MLP → last cross-attention block → gate. The original predicted target is held fixed. Constant or length-one vectors have undefined Spearman correlation and are excluded from the mean, with their counts retained. Gate randomisation does not affect the image-only head readout.

| Head | Modality | Stage | Defined | Undefined | Mean Spearman rho [95% CI] |
|---|---|---|---:|---:|---|
| informative | text | head | 1 | 0 | -0.7622 [-0.7622, -0.7622] |
| informative | text | cross_attention | 1 | 0 | -0.9231 [-0.9231, -0.9231] |
| informative | text | gate | 1 | 0 | -0.6713 [-0.6713, -0.6713] |
| informative | image | head | 1 | 0 | -0.5427 [-0.5427, -0.5427] |
| informative | image | cross_attention | 1 | 0 | -0.4230 [-0.4230, -0.4230] |
| informative | image | gate | 1 | 0 | 0.2066 [0.2066, 0.2066] |
| humanitarian | text | head | 1 | 0 | -0.5956 [-0.5956, -0.5956] |
| humanitarian | text | cross_attention | 1 | 0 | 0.2703 [0.2703, 0.2703] |
| humanitarian | text | gate | 1 | 0 | 0.6000 [0.6000, 0.6000] |
| humanitarian | image | head | 1 | 0 | 0.0495 [0.0495, 0.0495] |
| humanitarian | image | cross_attention | 1 | 0 | 0.0432 [0.0432, 0.0432] |
| humanitarian | image | gate | 1 | 0 | 0.0447 [0.0447, 0.0447] |
| severity | image | head | 1 | 0 | 0.1161 [0.1161, 0.1161] |
| severity | image | cross_attention | 1 | 0 | 0.0863 [0.0863, 0.0863] |
| severity | image | gate | 1 | 0 | 0.0863 [0.0863, 0.0863] |
| disaster_type | image | head | 1 | 0 | -0.1926 [-0.1926, -0.1926] |
| disaster_type | image | cross_attention | 1 | 0 | -0.1651 [-0.1651, -0.1651] |
| disaster_type | image | gate | 1 | 0 | -0.1651 [-0.1651, -0.1651] |
