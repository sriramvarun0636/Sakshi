# WP5 fixed-sentence verifier ablations

All rates use the same round-0 sentences as their denominator. The no-regeneration
condition is full five-gate acceptance before rewrite. The entailment-only condition
uses the saved G3 decision and ignores G1, G2, G4 and G5. The text-only G3
condition rescored G3 after removing image captions from its premise while holding
the other four saved gates fixed. The final full condition includes the original rewrite
round; the ablations do not generate new sentences.

| Scope | Windows | Round 0 | Full initial | Full final | G3 only initial | Text-only G3 initial | G3 decisions changed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 53 | 371 | 0.852 | 0.895 | 0.889 | 0.668 | 75 |
| hurricane_harvey | 18 | 139 | 0.849 | 0.914 | 0.878 | 0.698 | 23 |
| hurricane_maria | 26 | 199 | 0.844 | 0.864 | 0.889 | 0.638 | 45 |
| mexico_earthquake | 9 | 33 | 0.909 | 1.000 | 0.939 | 0.727 | 7 |

NLI thresholds remain the uncalibrated defaults because independent human labels are unavailable.
