# SAKSHI build log

This file is the handoff record for implementation work. Keep it current when a component
changes. The detailed requirements remain in [`SAKSHI_design_spec_v2.md`](../SAKSHI_design_spec_v2.md).

## Current state — 2026-10-24

- The user committed and pushed the previous work as `db819c4` (WP0, WP1 core, B5,
  WP2 and the initial WP4 algorithms). The working tree was clean when work resumed
  on 2026-10-24. New changes in this session are uncommitted.
- The original data pipeline and prepared manifest were present before this session.
  `data/processed/manifest.parquet` has 35,685 images. The unified train/dev/test counts
  are 19,896 / 5,720 / 10,069. Official test selector counts were verified as
  1,534 / 955 / 529 / 1,138 / 3,213 for informative / humanitarian / severity /
  DMD severity / disaster type.
- No trained BERTweet classifier checkpoint or trained-classifier test metrics exist yet. The machine
  reports no CUDA or MPS device, so the full training and baseline matrix need suitable
  GPU compute as described in the spec.
- The full CLIP feature cache completed with **35,685/35,685 usable images**. The token
  array is 35,685×50×768 (`float16`, 2.6 GB); the embedding array is 35,685×512
  (`float32`, 70 MB). The sample of embedding norms checked was 0.99999988–1.00000012.
  The cache lives under ignored `data/processed/features/`.
- The BERTweet tokenizer was downloaded and verified (64,001 tokens); the real pretrained
  BERTweet and fusion-model forward pass succeeded with all eight classifier heads and
  modality weights. CLIP ViT-B/16 was downloaded and verified
  on one synthetic image and two actual manifest images. The latter produced the expected
  50×768 token and 512-vector shapes.

## Implementation record — 2026-10-23

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

### WP4 initial algorithms

- Added sample-to-tweet rollup, conservative provenance exclusion, evidence-group union,
  the fixed priority formula and selection diversity limits in `src/sakshi/aggregate/core.py`.
- Added pinned MiniLM text embeddings, category-local claim clustering, the 1,000-tweet cap,
  UTC windows and aggregate JSON writers in `clusters.py` and `replay.py`.
- The real MiniLM model produced 384-dimensional vectors. The Sri Lanka rollup using B5
  predictions produced 830 tweets from 1,022 samples, with 309 eligible tweets and 792
  evidence groups. This was an integration check using a baseline, not a LOEO evaluation.
- CLI integration, packet validation and ranking evaluation were not yet implemented.

## Session — 2026-10-24: aggregation integration and ranking

### Changes

- Added `sakshi-aggregate` (`src/sakshi/aggregate_cli.py`) and strict YAML configuration
  at `configs/aggregation/default.yaml`. The default research mode is LOEO. Classifier
  `config.yaml` and `env.json` must identify the requested model and held-out event;
  the `heldout` fallback accepts only a main model's unified test rows.
- Added `aggregate/workflow.py`: manifest/prediction/provenance joins, run metadata,
  resolved configuration, seed, input/aggregation-source hashes, tweet table, per-window
  JSON, ranking audits and a completion marker. Complete reruns verify artifact hashes
  and return without re-encoding. Changed inputs/configuration require a different output
  root. Demonstrations use a separate `demo/` directory and carry `evidence_ready: false`
  when packets are absent.
- Added `aggregate/packets.py` and packaged `schemas/evidence_packet.json`, copied from
  Appendix E.1. Research aggregation requires same-model packets. Checks reject schema
  violations, private handles/URLs/filenames, duplicate IDs, non-finite JSON, wrong model
  IDs, changed text, labels, confidences, context, abstention and provenance. Where present,
  prediction modality masks and weights must also match. **This validates the WP3 contract;
  it does not implement Integrated Gradients, Grad-CAM or the WP3 packet producer.**
- Added `aggregate/ranking.py` with NDCG@10/25 and priority/random/recency/confidence
  comparisons. Priority scores are shared with selection through `core.rank_posts`.
  Unknown gold and windows without positive relevance are counted explicitly.
- Corrected two bugs exposed by the real integration run: NumPy evidence-group roots now
  become JSON-safe Python integers, and the infrastructure system sentence counts severe
  infrastructure tweets rather than severe tweets across every category.
- Corrected duplicate-text grouping to apply `humaid.normalise_text` to the original tweet
  text. URL presence no longer prevents identical claims from joining. Clustering receives
  display-cleaned text. Added canonical-label, provenance, timestamp and probability checks.
- Centralised experiment environment capture in `config.environment`, added the CLI and
  packaged schema to `pyproject.toml`, and added required aggregation dependencies.
  Ignored `/build/` and `/dist/` after verifying a wheel includes the schema and CLI.

### Protocol choices recorded before research use

- Ranking uses exponential gains `2^relevance - 1`. Tweet gold relevance is the maximum
  known grade among that tweet's agreed sample labels. A missing infrastructure severity
  receives grade 2; missing humanitarian labels do not imply grade 0.
- All four methods rank the same eligible, clustered tweets with known gold. This audits
  ordering after filtering, before diversity caps and category presentation order. It does
  not measure retrieval recall for tweets rejected by the classifier/provenance filters.
- Category and severity counts use eligible tweets. Overall informative/abstention and
  provenance exclusion counts include the entire window. Priority weights are unchanged.

### Real integration results and checks

- Ran cached, pinned MiniLM on the existing fixed B5 prediction table and provenance
  replays for Sri Lanka plus the three evaluation-plan events. These are **demonstrations**,
  not LOEO research experiments; no prompts, classifier or ranking weights were tuned.
- Final artifacts:
  `runs/aggregate/validated-20261024/demo/clip_zeroshot-s0/<event>/`.
  Each event directory contains `config.json`, `config.yaml`, `inputs.json`, `env.json`,
  `tweets.parquet`, `ranking.json`, `ranking.md`, timestamped windows and `complete.json`.
- Output totals: **10,055 tweets, 129 daily windows, 1,647 selected tweet entries**.
  Harvey: 3,992 tweets / 24 windows; Maria: 3,995 / 54; Mexico: 1,238 / 17;
  Sri Lanka: 830 / 34. The report is `reports/results/ranking_demo.md`.
- Audited every final artifact hash, selected tweet eligibility, sample membership, window
  boundaries, sequential P-IDs, the 25/8/2 diversity caps, and eligible-only severity counts.
  All passed. A real CLI rerun verified the completed artifacts without loading MiniLM again.
- Earlier exploratory outputs remain under `/private/tmp/sakshi_aggregate_check`,
  `runs/aggregate/demo/` and `runs/aggregate/build-20261024/`; use only the final path above
  for this session's handoff. No existing data was deleted.
- **107 tests passed** with no dataset download required. Ruff lint, format checks and
  `git diff --check` passed. A wheel was built under `/private/tmp/sakshi-wheel-check`;
  its packet schema and CLI entries were verified. No commits were created in this session.

## Checks and findings

- Latest full suite: **107 passed** on 2026-10-24; lint, formatting and whitespace checks
  passed. Re-run appropriate checks after new implementation changes.
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
  `/runs/`. The current build session has not created commits.

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
   WP3 explanations and packet production, then WP5 verification/generation and WP6.
   The WP4 core/CLI/ranking harness is implemented; its research acceptance criteria await
   trained LOEO predictions and genuine same-model WP3 packets. `ranking_demo.md` must not
   be presented as the research `ranking.md`.

## How to resume in another chat

Read this log, `README.md` and the relevant section of `SAKSHI_design_spec_v2.md`. Inspect
`git status --short` and the feature-cache index before running any expensive job. The
latest committed baseline is `db819c4`; inspect the working tree for subsequent changes.
