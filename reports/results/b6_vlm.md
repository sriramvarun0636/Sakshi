# B6 zero-shot vision-language baseline

Model: `Qwen/Qwen2.5-VL-7B-Instruct` at revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`; quantization: `bitsandbytes-nf4`.
Fallback reason: `None`. Official selector union: **6,939** posts.
Failures: **2,235** (32.2%); failed parses use the frozen training majority.
Latency per post: median **4.68s**, mean **6.12s**, p95 **9.46s**.
Hierarchy contradiction rate: **0.322**.
This is an intent-to-treat baseline: every failed JSON/repair attempt receives the frozen training-majority labels. The high failure rate materially affects the scores.
The model emits labels only; probability calibration is not applicable.

| Official selector | n | Macro-F1 [95% bootstrap CI] | Weighted-F1 | Accuracy | Failure rate |
|---|---:|---:|---:|---:|---:|
| informative | 1,534 | 0.543 [0.519, 0.568] | 0.614 | 0.636 | 40.4% |
| humanitarian | 955 | 0.490 [0.442, 0.536] | 0.623 | 0.653 | 51.8% |
| severity | 529 | 0.371 [0.329, 0.413] | 0.539 | 0.590 | 14.0% |
| severity dmd | 1,138 | 0.502 [0.475, 0.529] | 0.707 | 0.720 | 28.2% |
| disaster type | 3,213 | 0.680 [0.662, 0.698] | 0.700 | 0.719 | 26.9% |

Scores use frozen official test selectors and 1,000 seeded item-bootstrap draws.
Protocol SHA256: `8628a0a681f858c84a14828184ac96fb25e3fd8937e72f94f632077307892957`. Prediction SHA256: `c58934695de947559583270bb35e21cafbe76995c0ce9249dbefa811d17e3d74`.
