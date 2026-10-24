# First full classifier: seed 0 dev review

Date: 2026-10-24. Run: `full-s0-20261024-062724`, model `full-s0`.
[Private Kaggle Version 1](https://www.kaggle.com/code/varun0636/sakshi-classifier-seed-0?scriptVersionId=361702408).

Training, dev calibration, export and verified local restore completed. **Official tests
have not been scored.** These dev results were used for checkpoint and threshold
selection; they are not final test results or directly comparable to B5's official-test scores.

## Configuration and execution

- `configs/classifier/full.yaml`, seed 0; pinned BERTweet and frozen cached CLIP features.
- Unified split: 19,896 train / 5,720 dev / 10,069 test rows. Full cache: 35,685 usable images.
- Batch 32, accumulation 1, 16,000 sampled rows per epoch; max 10 epochs, patience 3,
  minimum selection improvement 0.002. No configuration changes during the run.
- One Tesla T4 used with fp16 and GradScaler; Kaggle exposed two T4s.
- Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, Transformers 5.16.1.
- Completed 8 epochs; **epoch 5** selected. Training wall time **839.04s (13.98 min)**;
  cumulative train/dev epoch compute **793.97s**. Whole notebook **1060.1s (17.67 min)**,
  including bootstrap, CUDA smoke, calibration and export.
- Peak allocated GPU memory: training **2.84 GiB**, separate padded 128-token smoke
  **3.84 GiB**. These are allocated peaks, not total device usage.
- 3,996 optimizer updates, four skipped fp16 overflow updates (epochs 1 and 3),
  zero nonfinite loss batches; all eight heads received supervision.
- All three modality patterns passed the hosted CUDA smoke. Token truncations: zero
  in train/dev/test. Test text lengths were audited without test prediction or scoring.
- PyTorch warned that attention backward is nondeterministic under the spec's
  `torch.use_deterministic_algorithms(True, warn_only=True)` setting. Bitwise CUDA
  reproducibility is not claimed.

## Checkpoint selection

Macro-F1 on supervised dev rows at epoch 5, before temperature scaling:

| Head | Macro-F1 |
|---|---:|
| Informativeness | 0.920990 |
| Hierarchical humanitarian, five classes | 0.809715 |
| Severity, raw head on labelled rows | 0.702408 |
| Disaster type | 0.842833 |
| Mean selection score | 0.818987 |

Severity selection uses the raw head across severity-labelled dev rows. It is distinct
from the predicted-gated official severity protocol. The humanitarian selection score
uses the hierarchical final label, including `not_humanitarian`.

| Epoch | Dev selection score |
|---|---:|
| 1 | 0.799277 |
| 2 | 0.807511 |
| 3 | 0.802992 |
| 4 | 0.814568 |
| 5 | 0.818987 |
| 6 | 0.815096 |
| 7 | 0.812008 |
| 8 | 0.811964 |

## Dev calibration

Temperatures minimize dev NLL under the fixed bounds; all fits completed without fallback.
Humanitarian calibration below is for the **conditional four-class head**, distinct from
the five-class selection score above.

| Head | Labelled rows | Temperature | ECE before | ECE after | NLL before | NLL after |
|---|---:|---:|---:|---:|---:|---:|
| informative | 2674 | 0.723362 | 0.037392 | 0.008647 | 0.194870 | 0.181723 |
| humanitarian | 812 | 0.809584 | 0.027770 | 0.015379 | 0.180383 | 0.172795 |
| severity | 1657 | 1.021259 | 0.035622 | 0.038404 | 0.590183 | 0.590087 |
| disaster_type | 1941 | 0.786870 | 0.061678 | 0.017262 | 0.442647 | 0.423524 |

**Recorded calibration exception (§7.15):** severity ECE rose by 0.002782, while NLL
improved slightly. Its prescribed NLL-fit temperature was retained. No temperature was
retuned to improve the reported ECE.

Dev-selected abstention thresholds: `tau_inf=0.5`, `tau_hum=0.9`; coverage **90.7465%**,
covered humanitarian macro-F1 **0.877398**. These are in-sample dev tuning figures.

## Artifact verification and handoff

- Restored to ignored `runs/classifier/full-s0-20261024-062724/` using `sakshi-gpu restore`.
  Verified the archive part, exact member list and all 16 artifact hashes.
- Export inventory and part: `runs/transfer/kaggle-output-361702408/`.
  Part size **524,891,643 bytes**; SHA256
  `bae23e1f3397257833e72808f870c82a822b436b7e4a4435e2572cbf94502f6c`.
- Best checkpoint: **599,224,881 bytes**; SHA256
  `3c1219674106656eba6bf7e3a3ecfbb04ece0c16c81df35e88dd21f794371e53`.
- Local manifest, three feature files and all recorded training implementation hashes
  match `inputs.json` from the hosted run. Audit: `artifact_audit.json` in the transfer folder.
- Strict checkpoint loading, finite-parameter checks and local CPU inference passed for
  both modalities, text only and image only using a dev sample. Probability, hierarchy
  and missing-modality assertions passed. `local_inference_check.json` records the check;
  its 6.04 seconds includes model loading and is not a serving-latency benchmark.
- The local check used existing cached model/tokenizer files with
  `HF_HOME=/private/tmp/sakshi_hf_cache` and `HF_HUB_OFFLINE=1`.
- Original bundle provenance records commit `009daf2` plus uncommitted source; its exact
  source snapshot was verified against committed baseline `f7fcf81` before launch.
- `last.pt` stays in saved private Kaggle Version 1. It was excluded from the portable
  export; the local run supports inference, but does not contain epoch resume state.
- No official evaluation freeze/request or trained-model test output exists. Preserve
  the checkpoint and calibration; score official selectors once the configuration is final.
- Next implementation milestone: WP3 Integrated Gradients, live CLIP Grad-CAM,
  faithfulness checks and evidence packets from this same trained model.
