# WP5 captioned versus uncaptioned generation

The two research runs use the same 53 windows, aggregate files, classifier folds,
LLM family, NLI model and default thresholds. Captions change the LLM input and
the verifier premise, so this is an end-to-end input comparison. The fixed-sentence
G3 caption ablation is reported separately.

| Scope | Variant | Windows | Round 0 | VR initial | VR final | Evidence coverage | Generation failures |
|---|---|---:|---:|---:|---:|---:|---:|
| Overall | No caption | 53 | 343 | 0.860 | 0.910 | 0.310 | 13 |
| Overall | Qwen caption | 53 | 371 | 0.852 | 0.895 | 0.313 | 11 |
| hurricane_harvey | No caption | 18 | 106 | 0.858 | 0.896 | 0.242 | 6 |
| hurricane_harvey | Qwen caption | 18 | 139 | 0.849 | 0.914 | 0.329 | 3 |
| hurricane_maria | No caption | 26 | 158 | 0.861 | 0.905 | 0.301 | 7 |
| hurricane_maria | Qwen caption | 26 | 199 | 0.844 | 0.864 | 0.355 | 3 |
| mexico_earthquake | No caption | 9 | 79 | 0.861 | 0.937 | 0.477 | 0 |
| mexico_earthquake | Qwen caption | 9 | 33 | 0.909 | 1.000 | 0.173 | 5 |

Rates are automatic verifier outcomes, not human-rated factual accuracy.
NLI thresholds remain uncalibrated because independent reviewers are unavailable.
