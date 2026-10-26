# Three-event research sitreps · HF batch run

The [computed verification table](summary_hf.md) covers all **53** frozen LOEO
windows: Harvey 18, Maria 26 and Mexico 9. The exported sitreps and the input ZIPs
were checked against the frozen event, window, packet, aggregate and evaluation-plan
hashes before restoration. The private per-window JSON and raw LLM cache remain under
ignored `runs/` paths.

| Measure | Overall |
|---|---:|
| Round-0 generated sentences | 343 |
| Initial verifier acceptance | 0.860 |
| Final verifier acceptance | 0.910 |
| Selected-post evidence coverage | 0.310 |
| Generation failures | 13 of 53 windows |

Eleven failed windows contain only system lines; two retain verified claims from
round 0 but record a failed rewrite. These failures are part of the evaluation and
are not silently dropped from window counts or average latency. The verifier's
acceptance rate is a **model check against cited posts**, not an estimate of factual
correctness or human-judged gate precision. The 50-sentence Harvey calibration sheet
has been generated locally with
blank reviewer labels. Independent reviewers and an adjudicator are unavailable, so
the human calibration, Cohen's κ and manual precision/recall claims remain open.

Generation used private Kaggle T4 jobs, Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28` in 4-bit, and DeBERTa-v3-large
MNLI/FEVER/ANLI/Ling/WANLI revision `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`.
The NLI thresholds, entailment **0.5** and contradiction **0.3**, are the design
spec's **uncalibrated defaults** because Harvey human labels do not exist. The
frozen generation source was Git commit `aaed31631b949923a84c45d56f58680dcd33f060`.
