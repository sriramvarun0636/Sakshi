# Frozen official test · full seed 0 versus B6 zero-shot VLM

The same official posts are compared for each task. Δ is full minus B6 macro-F1; its 95% interval
resamples posts together (1,000 seeded paired draws). The interval reflects test-post sampling,
not variability across training seeds. B6 uses text and image, but emits JSON labels rather than probabilities.
Its frozen majority fallback was used on 2,235/6,939 posts
(32.2%), so the comparison includes generation reliability.

| Task | n | Full F1 | B6 F1 | Δ [95% CI] | B6 failure rate |
|---|---:|---:|---:|---:|---:|
| Informativeness | 1,534 | 0.919 | 0.543 | +0.376 [+0.346, +0.403] | 40.4% |
| Humanitarian | 955 | 0.791 | 0.490 | +0.301 [+0.254, +0.353] | 51.8% |
| Severity, gold gated | 529 | 0.590 | 0.371 | +0.219 [+0.151, +0.286] | 14.0% |
| Severity, predicted gated | 529 | 0.582 | 0.371 | +0.211 [+0.142, +0.272] | 14.0% |
| Severity, DMD | 1,138 | 0.445 | 0.502 | -0.057 [-0.100, -0.015] | 28.2% |
| Severity, DMD, equal coverage | 1,138 | 0.699 | 0.502 | +0.197 [+0.157, +0.236] | 28.2% |
| Disaster type | 3,213 | 0.859 | 0.680 | +0.179 [+0.161, +0.198] | 26.9% |

The ordinary DMD severity row compares the full model's gated output with B6's always-present label.
The equal-coverage row forces the full model's severity argmax on every DMD post.
These exploratory paired intervals have no multiple-comparison correction.
