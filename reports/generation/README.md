# Three-event research sitreps · HF batch runs

Both private Kaggle runs cover the same **53 frozen LOEO windows**: Harvey 18,
Maria 26 and Mexico 9. The sitreps, packet files, aggregates and evaluation plan
were hash checked before restoration. Private per-window JSON and raw LLM cache
remain outside Git.

| Measure | No caption | Qwen caption |
|---|---:|---:|
| Round-0 generated sentences | 343 | 371 |
| Initial verifier acceptance | 0.860 | 0.852 |
| Final verifier acceptance | 0.910 | 0.895 |
| Selected-post evidence coverage | 0.310 | 0.313 |
| Generation failures | 13 | 11 |
| System-only windows | 11 | 8 |

See the [no-caption summary](summary_hf.md), [captioned summary](summary_hf_captioned.md),
[matched-window comparison](caption_comparison.md),
[caption coverage](captions.md) and [fixed-sentence gate ablations](ablations_captioned.md).
Captions lowered the overall automatic final rate despite improving Harvey. Mexico
fell from 79 to 33 round-0 sentences and had five invalid-JSON generation failures
in the captioned variant. These failures remain in the reported denominators.

The 1,449 captions used Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`. Both sitrep runs used private T4
jobs, Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28` in 4-bit, and DeBERTa-v3-large
MNLI/FEVER/ANLI/Ling/WANLI revision `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`.
The generation source commits were `aaed31631b949923a84c45d56f58680dcd33f060`
without captions and `ff9e8c057dbd319a070156045dfcf986db6dc6f7` with captions.

The entailment and contradiction thresholds, **0.5** and **0.3**, remain the design
spec's uncalibrated defaults. Automatic verifier acceptance is a model check against
cited posts, not human-rated factual accuracy. The 50-sentence Harvey calibration
sheet has blank reviewer labels. Two independent reviewers and an adjudicator are
unavailable, so human calibration, Cohen's κ and manual precision/recall remain open.
