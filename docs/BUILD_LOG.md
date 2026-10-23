# SAKSHI build log

This file is the handoff record for implementation work. Keep it current when a component
changes. The detailed requirements remain in [`SAKSHI_design_spec_v2.md`](../SAKSHI_design_spec_v2.md).

## Current state — 2026-10-23

- Working tree contains uncommitted implementation changes. `SAKSHI_design_spec_v2.md` was
  already staged and modified when this build session began; do not overwrite that work.
- The original data pipeline and prepared manifest were present before this session.
  `data/processed/manifest.parquet` has 35,685 images. The unified train/dev/test counts
  are 19,896 / 5,720 / 10,069. Official test selector counts were verified as
  1,534 / 955 / 529 / 1,138 / 3,213 for informative / humanitarian / severity /
  DMD severity / disaster type.
- No trained BERTweet classifier checkpoint or final test metrics exist yet. The machine
  reports no CUDA or MPS device, so the full training and baseline matrix need suitable
  GPU compute as described in the spec.
- The full CLIP feature cache completed with **35,685/35,685 usable images**. The token
  array is 35,685×50×768 (`float16`, 2.6 GB); the embedding array is 35,685×512
  (`float32`, 70 MB). The sample of embedding norms checked was 0.99999988–1.00000012.
  The cache lives under ignored `data/processed/features/`.
- The BERTweet tokenizer was downloaded and verified (64,001 tokens); the real pretrained
  BERTweet and fusion-model forward pass succeeded with all nine output heads. CLIP ViT-B/16 was downloaded and verified
  on one synthetic image and two actual manifest images. The latter produced the expected
  50×768 token and 512-vector shapes.

## Implemented this session

### WP0 foundations

- Added shared `Post`, `Prediction`, provenance, explanation and sitrep types in
  `src/sakshi/types.py`.
- Added privacy-aware tweet normalization, robust RGB image loading, safe dataset image
  paths and atomic artifact writes in `src/sakshi/text.py`, `images.py`, `paths.py` and
  `atomic.py`.
- Added frozen, pinned CLIP ViT-B/16 feature extraction and sample-ID-indexed memory-mapped
  cache in `src/sakshi/data/features.py`; wired `sakshi-data prepare features` into the
  existing pipeline. The source checkpoint revision is fixed in that module.
- Added strict nested YAML loading, deterministic seeds and run metadata in
  `src/sakshi/config.py`.
- Added official selector assertions, classification and calibration metrics, bootstrap
  intervals and tables in `src/sakshi/eval/`.
- Added a gold-label `OraclePredictor` for later integration fixtures.
- Added optional dependency groups in `pyproject.toml`; ignored generated `runs/` in
  `.gitignore`.

### WP1 classifier core

- Implemented BERTweet + frozen-CLIP token fusion, bidirectional cross-attention, learned
  missing-modality tokens, modality gate and eight heads in `src/sakshi/models/fusion.py`.
  The BERTweet tower is called only for rows with text.
- Implemented class-balanced masked focal loss, modality-dropout label masking, manifest
  batching, sample-ID cache alignment and 60/40 CrisisMMD/CIB sampling in
  `src/sakshi/models/losses.py` and `data.py`.
- Implemented hierarchical final decisions and abstention, checkpoint loading, live image
  inference and manifest prediction tables in `src/sakshi/models/predict.py`.
- Implemented dev temperature fitting with NLL fallback, review-threshold selection,
  training, calibration and official test evaluation in `src/sakshi/models/`. The
  `sakshi-model` CLI supports `train`, `calibrate`, `predict`, `evaluate`. The default
  config is `configs/classifier/full.yaml`.
- Corrected the training loop's final partial gradient-accumulation group so it has the
  same update weight as a full group, keeps finite pending gradients across a skipped
  nonfinite batch, and advances the learning-rate schedule only after a real AMP update.
  A regression test compares a one-batch update with accumulation factors one and two.
- With a stand-in feature store, classifier training-row counts exactly match spec §7.1:
  informative 6,006; conditional humanitarian 1,788; severity 4,077; disaster type 11,105;
  each auxiliary head 8,729.
- A synthetic end-to-end test trains one epoch, writes a checkpoint and metadata, reloads
  the run, predicts, and writes calibration artifacts. A separate synthetic official-test
  fixture verifies all fixed selector sizes, raw versus predicted severity gating and
  hierarchy consistency.
- Implemented B5 CLIP zero-shot in `src/sakshi/baselines/clip_zeroshot.py` and the
  `sakshi-baselines clip-zeroshot` CLI. The exact Appendix E.8 prompts and pinned CLIP
  checkpoint score the full image cache; the final humanitarian/severity labels follow
  the common hierarchy. The first real run exposed a contradiction in humanitarian
  post-processing; a regression test caught the corrected edge case before the final run.
  Final B5 run: `runs/baselines/clip_zeroshot-s0-20261023-160328/`; the fixed-selector
  report is `reports/results/baselines_clip_zeroshot.md`. Official macro-F1 was 0.830
  informativeness, 0.564 humanitarian, 0.382 gold-gated severity, 0.317 DMD severity,
  and 0.655 disaster type, with 1,000 bootstrap resamples. This is a zero-shot baseline,
  not the trained full model.

### WP2 provenance core

- Implemented all nine temporal status decisions, strict time visibility, same-tweet
  exclusion, two-tier pHash/CLIP matches and earliest dated origin selection in
  `src/sakshi/provenance/detect.py`.
- Implemented cache-aligned provenance index construction, real CLIP graphic scoring,
  vectorized candidate retrieval, live PIL-image checks and chronological event replay in
  `src/sakshi/provenance/`. The `sakshi-provenance` CLI supports `build-index` and `replay`.
- Tests cover every status, onset equality, graphic templates, undated matches, source
  ordering, cache alignment, live-image checks and replay order.
- Corrected AIDR folder onsets to 00:00 UTC and resolved AIDR sample-query events from
  their folder IDs, as §8.2 requires. Exposed prepared provenance queries and candidate
  retrieval for efficient CLIP-threshold sweeps. Added all nine seeded benchmark image
  transformations in `src/sakshi/provenance/transforms.py`, with deterministic tests.
- Built the real provenance index: 35,685 unique rows, 34,422 dated, 8,386 flagged
  graphics with the default 0.5 threshold. Replay of all seven CrisisMMD events completed:
  **18,083 queries**, including 574 `recycled` results. The generated status table is
  `reports/provenance/replay_summary.md`. An audit confirmed that every dated origin is
  strictly earlier than its query, every `recycled` origin predates its event onset, and
  every `duplicate` origin belongs to the query event.
- Added `src/sakshi/provenance/benchmark.py`, the fixed-role injection plan and scoring,
  threshold sweep, held-out report generation, and `sakshi-provenance benchmark` command.
  The full injection benchmark has not been run; its published metrics/figures do not yet
  exist. It needs substantial image inference compute. `matplotlib` was added to the ML
  extra and installed locally. A real seeded injected image was transformed, encoded,
  retrieved, correctly classified `recycled`, and assigned its source component. Synthetic
  report rendering produced all ten required PNG figures.
  The graphic threshold remains at its explicit 0.5 default pending tuning/validation.
- Added blind two-rater label export and adjudication scoring in
  `src/sakshi/provenance/labelling.py`, exposed as `sakshi-provenance label-export` and
  `label-score`. The real export selected **897 pairs** (including 50 graphics), wrote a
  blank reviewer sheet at `reports/provenance/labelling_sheet.csv`, and embedded 1,794
  thumbnails in the ignored 56.2 MB local viewer `runs/provenance/labelling.html` with
  no missing images. Detector decisions are kept out of the sheet/viewer in ignored
  metadata. Export refuses to overwrite human labels; scoring requires both reviewers and
  final adjudication. No real human labels or study metrics exist yet.

## Checks and findings

- The latest full suite passed: **93 tests** before the new B5 tests; its two focused tests
  passed. Ruff lint, formatting and `git diff --check`
  passed. Re-run these checks after further edits.
- Every benchmark event had a sufficient older-image positive pool for the maximum 20%
  injection rate. The observed pool range was 4,480–20,896 sources; 472 CrisisMMD images
  belonged to cross-event components and are marked ambiguous for scoring.
- Real CLIP image inference and a two-image cache build passed. The installed Transformers
  version is 5.20.2; PyTorch is 2.14.1.
- A partial Hugging Face cache caused Transformers to silently tokenize every CLIP graphic
  prompt as unknown tokens. The pinned tokenizer files were downloaded, yielding 49,408
  tokens and distinct graphic scores. `ClipGraphicScorer` and BERTweet tokenizer loading
  now reject incomplete vocabularies. This guard is important for reproducibility.
- Generated model, cache and prediction artifacts are excluded from git via `/data/` and
  `/runs/`. There are no new commits.

## Next actions

1. Run the full nine-transformation, four-rate, three-seed injection benchmark on suitable
   GPU compute; review the fixed-role threshold selection and generated tables/figures.
   The 0.5 graphic threshold still needs tuning or explicit validation on tuning data.
2. Have two reviewers label the exported pairs independently, then adjudicate and run
   `sakshi-provenance label-score`. No human labels should be inferred from detector output.
3. Run `ruff check .`, `ruff format --check .`, `pytest -q` and `git diff --check` after new work; fix any failure.
4. Run the first full classifier on a CUDA T4, time it, calibrate on dev, predict held-out
   rows and evaluate. No test metrics should be claimed until this is done.
5. Implement the remaining WP1 baselines, ablations, LOEO and HumAID evaluation. Continue
   WP3–WP6 in spec order.

## How to resume in another chat

Read this log, `README.md` and the relevant section of `SAKSHI_design_spec_v2.md`. Inspect
`git status --short` and the feature-cache index before running any expensive job. The
repository is intentionally uncommitted, and the spec file had pre-existing changes.
