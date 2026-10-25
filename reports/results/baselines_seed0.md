# B1 and B2: first frozen seed

Standard-split B1 text-only and B2 image-only were trained on private Kaggle T4 notebooks, calibrated on development data, restored from checksummed exports, and evaluated against the frozen test roles. This is one seed; the planned three-seed comparison and seven LOEO folds per baseline are still in progress.

| Evaluation labels | Head | Rows | B1 text macro-F1 [95% CI] | B2 image macro-F1 [95% CI] |
|---|---|---:|---:|---:|
| Agreed CrisisMMD labels | Informativeness | 1,534 | 0.828 [0.805, 0.848] | 0.880 [0.862, 0.898] |
| Agreed CrisisMMD labels | Humanitarian | 955 | 0.632 [0.584, 0.681] | 0.728 [0.674, 0.776] |
| Original text or image labels | Informativeness | 2,237 each | 0.794 [0.774, 0.813] | 0.876 [0.862, 0.890] |
| Original text or image labels | Humanitarian | 2,237 each | 0.670 [0.647, 0.690] | 0.756 [0.734, 0.778] |

The original-label rows are separate modality-specific test roles: B1 is scored against text labels and B2 against image labels. Those values are not a paired B1-versus-B2 comparison. Agreed-label rows use identical official selectors for both models; full seed 0 scored 0.919 informativeness and 0.791 humanitarian macro-F1 on those selectors. Differences across a single training seed should not be presented as an overall effect.

B2 also scored 0.612 gold-gated severity macro-F1 on 529 official rows, 0.388 DMD severity on 1,138, and 0.849 combined disaster type on 3,213. B1 has no image-only heads. Both baseline runs had zero hierarchy contradictions. All intervals are 1,000 item bootstrap percentile intervals for fixed checkpoints; they do not include training-seed variation.

## Frozen artifacts

- B1: `runs/classifier/b1-text-s0-20261025-065051/` (`evaluation_complete.json`, `evaluation_modality_complete.json`).
- B2: `runs/classifier/b2-image-s0-20261025-065121/` (`evaluation_complete.json`, `evaluation_modality_complete.json`).
- Full seed 0 reference: `runs/classifier/full-s0-20261024-062724/` (`evaluation_complete.json`).

The original-label evaluator hashes each checkpoint, calibration, label map, manifest, feature cache, implementation and bootstrap count before writing predictions and scores. Its completion marker hashes both outputs.
