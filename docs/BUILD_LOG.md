# SAKSHI build log

This file is the handoff record for implementation work. Keep it current when a component
changes. The detailed requirements remain in [`SAKSHI_design_spec_v2.md`](../SAKSHI_design_spec_v2.md).

## Current state — 2026-10-24

- Latest committed baseline: `f7fcf81` (hosted GPU workflow), following `009daf2`
  (WP4 workflow, ranking and packet validation) and `db819c4` (WP0, WP1 core, B5, WP2
  and initial WP4 algorithms). The user pushed the GPU workflow; the tree was clean
  when the hosted launch session began. Current documentation/report updates are uncommitted.
- The original data pipeline and prepared manifest were present before this session.
  `data/processed/manifest.parquet` has 35,685 images. The unified train/dev/test counts
  are 19,896 / 5,720 / 10,069. Official test selector counts were verified as
  1,534 / 955 / 529 / 1,138 / 3,213 for informative / humanitarian / severity /
  DMD severity / disaster type.
- The first full BERTweet classifier **trained and calibrated on Kaggle T4**, then its
  16 exported artifacts were checksum-verified and restored to
  `runs/classifier/full-s0-20261024-062724/`. Seed 0 selected epoch 5 of 8 with dev
  selection macro-F1 **0.8189865**; training took **839.04 seconds**, peak allocation
  **2.84 GiB**. See `reports/results/classifier_seed0_dev.md` and the session record below.
  **No trained-classifier official test metrics exist yet**; `FINAL_EVALUATION` stayed
  false. Other seeds/baselines/ablations/LOEO and the remaining work packages are pending.
  This 8 GiB laptop has MPS but no CUDA; training remains a hosted task.
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

## Session — 2026-10-24: reproducible hosted classifier workflow

### Changes

- Added `notebooks/train_classifier_kaggle.ipynb`, with clean outputs and ordered cells
  for verified source staging, constrained dependency installation, CUDA/data checks,
  a bounded smoke, timed full training, dev calibration, optional frozen official
  evaluation and checksummed export. It imports the tested package functions. Official
  evaluation is **disabled by default**. No account credentials or automatic uploads.
- Added `docs/GPU_TRAINING.md`; updated README and relevant spec sections for the CLI,
  pinned encoder, recovery files and private source-bundle workflow. A source snapshot
  replaces the need for a private Git clone token; the spec still allows cloning.
- Added `sakshi-gpu` and `compute/{bootstrap,bundle,files,preflight,runner}.py`. The bundle
  allowlists source/docs/configs/tests/notebook plus exactly four prepared files. It
  records original Git identity and every file's size/SHA256; bootstrap checks all bytes
  using the standard library before pip/imports, rejects unsafe paths and stale/changed
  workspace source, and leaves the large arrays on the read-only input mount.
- Added `SAKSHI_PROCESSED_DIR` relocation in `paths.py`, set before package imports.
  `config.environment` preserves original source identity when running the staged
  snapshot without Git, including hosts without a Git executable.
- Preflight checks exact sample-ID coverage, cache shape/dtype/availability, every value's
  finiteness, embedding norms, zero missing-image rows, tweet/component split isolation,
  official selectors and standard training-label counts.
- Pinned both BERTweet model and tokenizer to
  `b349c1243407b0dcffeabb2337497477286e27ab`; explicitly disabled built-in normalization
  and fast tokenization. The ML dependency now requires PyTorch >=2.6, matching current
  Transformers' loader requirement for the pinned `.bin` weights. The notebook constrains
  the existing CUDA PyTorch version rather than installing a replacement CUDA stack.
- Added explicit `configs/classifier/t4.yaml` (name `full-t4`, batch 16, accumulation 2).
  The default full config remains batch 32. Smoke uses the actual microbatch and pads
  text to 128 tokens; it checks head shapes/gates and real updates for both, text-only
  and image-only inputs. CUDA memory failure requests an explicit config change.
- Added `models/state.py` and `sakshi-model train --resume`: atomic `last.pt` includes
  current/best model, optimizer, scheduler, AMP scaler, sampler and CPU/CUDA/MPS/Python/
  NumPy RNG states, records and early-stop state. Resume checks saved config, manifest,
  feature bytes, training implementation and key runtime versions/device identity.
  Recovery starts after the last completed epoch; partial epochs repeat. Best checkpoint,
  metrics and logs are reconstructed from the authoritative recovery state.
- Retained GradScaler across epochs; clipping now lets fp16 scaler overflow handling
  skip updates, while other precision modes fail on nonfinite gradients. The scheduler
  advances only on real optimizer updates. Native bf16 support is required, excluding
  emulation so T4 uses fp16/scaling. Empty-update epochs fail. Fusion now converts reduced
  precision encoder states to the scatter buffer dtype without detaching gradients.
- Added `tokenization.json` with normalized text/truncation counts per split, and
  `training_summary.json` with best/completed epochs, wall/epoch compute times, precision,
  effective batch and peak CUDA memory. Model/data/source hashes are in `inputs.json`.
- Added a final-evaluation wrapper used by both `sakshi-model evaluate` and `sakshi-gpu
  final-evaluate`. It freezes model/config/calibration/maps/data/bootstrap hashes before
  scoring, verifies/reuses completed results, and recovers publication interruptions.
  Frozen runs cannot resume or recalibrate. Global official scoring rejects LOEO because
  other events' official rows may have been used for training; a LOEO evaluator is pending.
- Added allowlisted result export and verified restore. ZIP parts are at most 768 MiB,
  streamed with 1 MiB buffers. All downloaded parts, archive membership and restored
  artifact hashes are checked before publishing; existing destinations are preserved.
  Best checkpoint is included; larger `last.pt` recovery state is opt-in.

### Validation and local artifacts

- **133 tests passed**. Tests cover exact CPU uninterrupted/interrupted/resumed equality
  (weights, optimizer, scheduler, RNG, sampling and dropout), partial-artifact recovery,
  changed-input/config/freeze rejection, simulated AMP overflow, native precision
  selection, reduced-precision fusion gradients, cache corruption/leakage, bundle path
  safety/source integrity, three-pattern training smoke, one-time evaluation/reuse and
  multipart export/restore. All use small encoders and require no dataset/model download.
- Notebook schema validation and compilation of every Python cell passed. Its real
  bootstrap cell executed in an isolated Python process against a synthetic bundle.
  **The notebook's hosted CUDA cells have not been executed.**
- Ruff lint/format and `git diff --check` passed. A wheel under
  `/private/tmp/sakshi-gpu-wheel-check` was inspected for the new modules, schema and
  `sakshi-gpu` entry point; the installed CLI help was exercised.
- The real full-array preflight passed: **35,685 usable images**, **19,896 training** /
  **5,720 dev** rows; all eight training-label counts and five official-selector counts
  match the fixed release. No official predictions or metrics were computed.
- A real offline pinned-tokenizer audit found **zero truncated rows** in every split:
  train 8,729 text rows / longest 61 tokens; dev 3,770 / 63; test 5,584 / 60. Audit:
  `runs/transfer/tokenization-audit-20261024.json`. No model backward pass was attempted
  on this 8 GiB laptop.
- Local private-upload bundle path: **`runs/transfer/gpu-inputs-20261024/`**. The bundle
  contains the prepared manifest/cache and this session's exact source snapshot. It is
  ignored by Git. Keep the original directory structure for private Kaggle attachment.
  Rebuild to a new output directory after future implementation changes.
- No cloud uploads, hosted jobs, new full model scores or commits were made. Existing
  B5 and aggregation demonstration artifacts were preserved.

### Handoff before the hosted launch

This preparation checkpoint is historical; the completed hosted run is recorded in the
next session section.

Follow `docs/GPU_TRAINING.md` to attach the private bundle and import the notebook on a
CUDA host. Run preflight/smoke/training/calibration; review dev and measured timing, then
enable final evaluation only after configuration freeze. Save private outputs, download
all export parts and restore the run locally. Record actual GPU/time/memory/dev findings
here and update the compute estimates in spec §6.2. The WP1 full-run acceptance criterion
and trained-classifier metrics remain pending until that job completes. Next code
milestone: WP3 Integrated Gradients, Grad-CAM, faithfulness and same-model packets.

## Session — 2026-10-24: private Kaggle launch and completed seed 0 run

- Started from clean commit `f7fcf81`. Verified the source, configs and notebook in the
  existing bundle match the committed workspace. The bundle retains its original
  `009daf2` plus dirty-source provenance, with exact bytes recorded in its inventory.
- Created `runs/transfer/gpu-inputs-20261024.zip` with the bundle's 112 files and preserved
  directories. Size: **2,604,810,407 bytes**. SHA256:
  `cfd808bf7cc14d155264a45c4f12648caedb48600ebe1d05443c12bfe1c5bf9a`.
  The original uncompressed bundle and existing datasets were preserved.
- Connected Chrome to the user's signed-in Kaggle session. The user enabled the
  extension's file-URL access after Chrome initially rejected file selection, and
  completed Kaggle phone verification to unlock accelerator and Internet controls.
  No API keys, phone numbers or SMS codes were read or stored.
- Uploaded and created the **private** dataset
  `https://www.kaggle.com/datasets/varun0636/sakshi-gpu-inputs-20261024`.
  Version 1 finished processing: **112 files / 2.82 GB**, with `prepared/`, `source/`,
  `bundle.json`, `preflight.json` and `SHA256SUMS` intact.
- Imported `notebooks/train_classifier_kaggle.ipynb` into the **private** notebook
  `https://www.kaggle.com/code/varun0636/sakshi-classifier-seed-0` and attached that
  private dataset. Selected **GPU T4 x2**, enabled Internet, and retained the default
  **full config / seed 0 / batch 32 / final evaluation disabled**. The trainer uses
  one CUDA device. Kaggle reported 30 GPU hours available at setup.
- Submitted **Version 1**, named `seed0 train and dev calibration`, using **Save & Run
  All (Commit)** with GPU execution. Run URL:
  `https://www.kaggle.com/code/varun0636/sakshi-classifier-seed-0?scriptVersionId=361702408`.
  Startup verified all bundled bytes at the mounted path, installed the package against
  hosted Python 3.13 / PyTorch 2.11.0+cu128 / Transformers 5.16.1, and passed data preflight.
  The real CUDA smoke passed all three modality patterns at batch 32 / 128 padded tokens:
  32.58 seconds, peak allocation 4,118,389,760 bytes (3.84 GiB). An initial fp16 overflow
  recovered on the second both-modality attempt. The run subsequently completed training,
  dev calibration and export successfully. No official tests have been scored.
  PyTorch logged the spec-permitted nondeterministic attention warning under
  `torch.use_deterministic_algorithms(True, warn_only=True)`; this run is not a claim
  of bitwise CUDA reproducibility.
- Machine-readable session state is kept under ignored
  `runs/transfer/kaggle-session-20261024.json`. Future chats should inspect this state and
  the existing saved version before submitting another run, to avoid duplicate jobs.
- No model or algorithm changes were made in this launch session. The prior suite result
  remains 133 passing tests; rerun relevant checks if implementation changes are needed.

### Completion, artifacts and review

- Kaggle confirmed **Successfully ran in 1060.1s** (17.67 minutes for the complete
  notebook, including bootstrap, smoke, calibration and export). Training itself took
  **839.0367s**, with **793.9651s** cumulative train/dev epoch compute. Effective batch
  32, fp16, one Tesla T4; peak training allocation **3,050,460,672 bytes** (2.84 GiB).
  Eight epochs completed; early stopping selected **epoch 5**. Dev selection scores:
  `0.799277, 0.807511, 0.802992, 0.814568, 0.818987, 0.815096, 0.812008, 0.811964`.
- Best checkpoint's **dev** macro-F1: informative **0.920990**, hierarchical humanitarian
  **0.809715**, raw severity on labelled rows **0.702408**, disaster type **0.842833**.
  These checkpoint-selection scores must not be compared directly to B5's official-test
  scores. They are not the final trained-classifier research results.
- Audited every saved epoch: **3,996 optimizer updates**, **four skipped fp16 overflow
  updates** (three in epoch 1, one in epoch 3), **zero nonfinite loss batches**. All eight
  heads received supervision. Hosted tokenization recorded zero truncations in every split.
- Dev calibration fit all four temperatures without fallback. ECE before → after:
  informative `0.037392 → 0.008647`, conditional humanitarian `0.027770 → 0.015379`,
  severity `0.035622 → 0.038404`, disaster type `0.061678 → 0.017262`.
  **Recorded §7.15 calibration exception:** severity ECE increased by 0.002782, while
  NLL improved `0.590183 → 0.590087`. The prescribed NLL-fit temperature was retained;
  calibration was not retuned. The humanitarian calibration figures are conditional
  four-class metrics; its selection score above uses the hierarchical five-class output.
  Abstention thresholds `tau_inf=0.5`, `tau_hum=0.9` give dev coverage **90.7465%** and
  covered humanitarian macro-F1 **0.877398**. This is a dev-selected operating point.
- Downloaded the **524,891,643-byte** export part from the saved private notebook.
  Its SHA256 is `bae23e1f3397257833e72808f870c82a822b436b7e4a4435e2572cbf94502f6c`.
  Chrome named it `~/Downloads/full-s0-20261024-062724.zip.zip`; after verifying the
  hash, copied it to its declared `.zip.part001` name. Saved the exact exported JSON
  inventory from Kaggle's file page under `runs/transfer/kaggle-output-361702408/`.
  `sakshi-gpu restore` verified the part, archive member list and all **16 artifact
  hashes**, then published `runs/classifier/full-s0-20261024-062724/`.
  Checkpoint SHA256: `3c1219674106656eba6bf7e3a3ecfbb04ece0c16c81df35e88dd21f794371e53`.
  The best checkpoint is 599,224,881 bytes; `last.pt` remains on Kaggle and was excluded
  from the download. Preserve saved Version 1 for hosted resume/final evaluation.
- Verified all **18 local input/implementation hashes** against hosted `inputs.json`.
  Loaded the real checkpoint with strict state loading on CPU, checked every parameter
  for finiteness, and inferred a dev sample with both modalities, text only and image
  only. All three passed probability, hierarchy and missing-modality assertions. The
  inference check took 6.04s including model load; it is not a serving-latency benchmark.
  Audit files: `artifact_audit.json` and `local_inference_check.json` in the transfer
  folder. The first offline attempt used an empty default Hugging Face cache; rerunning
  with the existing `HF_HOME=/private/tmp/sakshi_hf_cache` succeeded without downloads.
- Updated the current README status, GPU guide (account verification, ZIP upload and
  browser filename handling), measured compute figures in spec §6.2, and the
  dev run report. Model/data files and detailed session state remain ignored by Git.
- Validation for this session: real hosted smoke/train/calibration/export, verified
  restore, local input/implementation audit and real-checkpoint inference all passed.
  `git diff --check` passed. The previous automated suite remains **133 passed**; no
  package, config, test or notebook code changed, so the suite was not rerun for these
  documentation/report changes. No commits or pushes were made.

## Checks and findings

- Latest full suite: **133 passed** on 2026-10-24; lint, formatting and whitespace checks
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

1. First hosted full classifier training, calibration and verified local restore are
   complete. Use `runs/classifier/full-s0-20261024-062724/` for WP3. Preserve Kaggle
   Version 1 and its recovery state. Freeze and score official selectors once the
   configuration is final; no trained-model test metrics exist yet.
2. Implement WP3 explanations and packet production with the trained run. Then continue
   WP5 verification/generation and WP6.
3. Run the full nine-transformation, four-rate, three-seed injection benchmark on suitable
   GPU compute; review the fixed-role threshold selection and generated tables/figures.
   The 0.5 graphic threshold still needs tuning or explicit validation on tuning data.
4. Have two reviewers label the exported pairs independently, then adjudicate and run
   `sakshi-provenance label-score`. No human labels should be inferred from detector output.
5. Implement the remaining WP1 baselines, ablations, LOEO and HumAID evaluation.
   The WP4 core/CLI/ranking harness is implemented; its research acceptance criteria await
   trained LOEO predictions and genuine same-model WP3 packets. `ranking_demo.md` must not
   be presented as the research `ranking.md`.
6. Run `ruff check .`, `ruff format --check .`, `pytest -q` and `git diff --check` after new work; fix any failure.

## How to resume in another chat

Read this log, `README.md` and the relevant section of `SAKSHI_design_spec_v2.md`. Inspect
`git status --short` and the feature-cache index before running any expensive job. The
latest committed baseline is `f7fcf81`; inspect the working tree for subsequent changes.
For the GPU workflow, read `docs/GPU_TRAINING.md` and use the bundle/notebook named above.
The first run is already complete: inspect its restored files and session JSON before
launching another job. Do not repeat training just to recover already saved artifacts.
