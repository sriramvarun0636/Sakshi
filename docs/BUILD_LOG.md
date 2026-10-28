# SAKSHI build log

This file is the handoff record for implementation work. Keep it current when a component
changes. The detailed requirements remain in [`SAKSHI_design_spec_v2.md`](../SAKSHI_design_spec_v2.md).

## Current state — 2026-10-25

- The signed-in private Kaggle account is connected through a locally stored CLI token.
  The token is outside the repository. A fresh private input dataset,
  `varun0636/sakshi-gpu-inputs-20261025`, has the checksum-inventoried source and
  prepared feature tree. All three standard full-model seeds are now restored and
  officially frozen; full-model Harvey and Maria LOEO are restored and frozen. Sri
  Lanka and Mexico LOEO are running; a two-slot queue then launches Harvey B1/B2
  LOEO pilots. Kaggle permits two simultaneous GPU sessions on this account.
- B1 text-only and B2 image-only classifier paths now select their modality's labels,
  input mask and active heads through explicit configs. Standard and LOEO baseline
  notebooks are prepared. Both passed CPU train/calibration/prediction checks and local
  split preflight. Standard seed 0 for both baselines completed, was restored with
  verified hashes, and has frozen official and original-modality evaluations. B3 late
  fusion is now frozen from those same seed-zero runs, with both mixing weights (0.5)
  chosen on 2,674 agreed dev rows. Its official informative/humanitarian macro-F1 is
  0.922/0.871, zero contradictions. The paired fixed-seed comparisons are in
  `reports/results/comparison_seed0_trained.md`; full multi-seed comparisons are pending.
- Full seed 1 frozen official test macro-F1 is 0.920 informative, 0.835 humanitarian,
  0.529 gold-gated severity, 0.450 DMD severity and 0.851 disaster type. The Harvey
  held-out LOEO fold scores 0.915 informative, 0.836 humanitarian and 0.534
  gold-gated severity across 4,435 held-out samples. Harvey is one of seven required
  held-out events; the seven-fold estimate remains incomplete.
- The three standard full-model seeds now average **0.919 ± 0.001** informative,
  **0.822 ± 0.027** humanitarian, **0.555 ± 0.032** gold-gated severity, and
  **0.856 ± 0.004** disaster type macro-F1 (mean ± sample SD). Seed 2's hosted
  notebook exported its dev-trained model without official scoring; its frozen test
  evaluation was completed locally on CPU after restore. See
  `reports/results/classifier_three_seed.md` and the three individual seed reports.
- The Maria held-out fold has 4,556 samples and macro-F1 0.907 informative, 0.872
  humanitarian and 0.569 gold-gated severity. Its WP4 packet plan selects 666 samples
  across 31 UTC windows. The Harvey plan selects 504 across 18 windows.
- B1 text-only seed 0 official agreed-label macro-F1 is 0.828 informative and 0.632
  humanitarian; on the original 2,237-row text-label test roles, it is 0.794 and
  0.670. B2 image-only seed 0 official agreed-label scores 0.880 and 0.728; on the
  original image-label roles, 0.876 and 0.756. The latter evaluations use a separate
  frozen evaluator and do not replace the agreed-label comparison.
- Harvey WP4 packet planning selects 504 samples across 18 UTC windows from 4,435
  held-out samples. A real one-post packet passed WP4 schema, same-model prediction,
  timestamp and provenance validation; checkpoint resumption reproduced the same hash.
  The full 504-post MPS explanation pass is resumable and has 159 completed post
  checkpoints. It was stopped to restore laptop responsiveness after severe memory
  pressure (about 11 GB swap in use). Research aggregates and sitreps remain pending
  until the selected packets complete; do not restart MPS inference alongside other
  local model runs on this 8 GB laptop.
- The specified local `qwen2.5:7b-instruct` Ollama model and DeBERTa NLI checkpoint
  have been downloaded for WP5; they have not yet produced a research sitrep.
- A frozen HumAID text-only transfer evaluator is implemented. It verifies 8,112
  overlap-free official test tweets (2,537 unseen-event, 5,575 shared-event), keeps
  strict and full label mappings separate, and resumes inference by batch. Full seed 0
  HumAID inference has 33 completed 64-row batch checkpoints and was stopped alongside
  the explanation pass to restore laptop responsiveness. No transfer score is claimed yet.
- **The seed 0 official classifier test is frozen and scored:**
  `runs/classifier/full-s0-20261024-062724/evaluation_request.json` records the input
  hashes, and `evaluation_complete.json` records the output hashes. The generated
  `reports/results/classifier.md` reports macro-F1 0.919 informative, 0.791 humanitarian,
  0.590 gold-gated severity, 0.445 DMD severity and 0.859 disaster type. Extended test
  predictions cover all 10,069 unified test rows (3,267 abstentions) for the review queue.
  The report includes generated per-class/confusion tables, reliability diagrams and a
  risk–coverage curve. `comparison_seed0_b5.md` gives paired 1,000-draw official-test
  comparisons against the completed B5 CLIP zero-shot run; all six point differences
  favour the full model, but this remains one seed and an unequal-modality comparison.
  These are one-seed results;
  no test-driven retuning was performed, and E1–E6 remain incomplete.
- **The full provenance benchmark completed:** all 432 held-out test cases across four
  events, nine transforms, four injection rates and three seeds were scored after tuning
  the CLIP threshold on the three designated tuning events. The selected threshold is
  0.925. Macro-averaged precision/recall/F1 are 0.606/0.694/0.595, with genuine-query
  false-positive rate 0.021. Identity and JPEG recall are at least 0.999, while
  screenshot, caption-bar and combined-edit recall are 0.069, 0.208 and 0.129. This is
  a closed-world test with every injected original in the index; it does not estimate
  open-world retrieval recall. The full tables, nine PR curves and degradation figure
  are in `reports/provenance/benchmark.md` and `reports/figures/`.
  CLIP image preparation used batches of 16; cases are cached under ignored
  `runs/provenance/benchmark_cache/` with config, data and implementation hashes. A real
  two-image batch matched individual query semantics. The independent human review is
  still pending.
- Human reviewers are not currently available, so the independent provenance and sitrep
  audits cannot be scored. Blind review materials can still be prepared.
- **WP6 local dashboard is implemented:** the primary custom fieldroom
  has five workspaces for live CPU classification, on-demand IG/Grad-CAM and image
  provenance; event replay;
  cited sitreps when WP5 artifacts exist; a local review queue; and read-only results.
  The original Streamlit implementation remains available through `sakshi-streamlit`.
  The custom fieldroom is a deliberate presentation-layer departure from the spec's
  Streamlit stack. Presentation mode cleans visible post handles/links and blurs detected
  faces. Browser checks covered all five pages and live text/image explanations. The existing B5
  aggregates are clearly labeled as baseline previews; full LOEO sitreps remain blocked
  by the upstream requirements recorded in `docs/GENERATION.md`. See
  `docs/DASHBOARD.md` for launch, artifact handling and limitations.
  The results workspace now selects the three-seed report by default and displays
  figures associated with the selected report. Its focused tests pass (10/10).

- The previous implementation checkpoint was `83d3fe0` (WP5). The records below retain
  the earlier build history.
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
  The official test was subsequently frozen and scored locally, as recorded above.
  Other seeds, baselines, ablations and LOEO folds remain pending as specified above.
  This 8 GiB laptop has MPS but no CUDA; training remains a hosted task.
- The separate LOEO evaluator and hosted notebook are implemented. It freezes one fold's
  inputs and completed artifacts, checks held-out-event isolation, and scores informative,
  humanitarian and two severity gates with present-gold-class bootstrap intervals. All
  seven folds passed local preflight with their own train/dev roles. Harvey is complete;
  the remaining GPU experiment matrix is in progress.
- **WP3 implementation is available:** Captum IG, differentiable live CLIP Grad-CAM,
  spans/regions, schema/private packets, standalone commands, resumable faithfulness and
  sanity audits, cached-caption integration and face-blurred figure export. Real dev
  integration produced **3 eligible same-model packets**, **12 explanations/heatmaps**,
  and both audit reports for all four heads. See `docs/EXPLAINABILITY.md` and the WP3
  session below. These are development checks; formal test audits, caption generation
  and packet coverage for every eventual WP4/WP5 research post remain pending.
- **WP5 code is available:** frozen three-event window planning, Ollama/HF structured
  generation, cache/retries, five-gate verification, one rewrite round, sitrep/report
  publication, Qwen caption infrastructure, per-post rationales and a blind human-audit
  path. Fake-backend and fixture integration tests pass. The live Harvey Ollama run,
  LOEO research sitreps and human calibration/evaluation are pending their upstream
  packets, model weights and reviewers; see `docs/GENERATION.md`.
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
  other events' official rows may have been used for training; the separate frozen LOEO
  evaluator is implemented and awaits its hosted fold results.
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

## Session — 2026-10-24: WP3 explanations and packet workflow

### Implementation

- Added `src/sakshi/explain/` modules for single-post input/gradient management, IG,
  Grad-CAM, packet construction, caption-cache reads, face-blurred rendering,
  faithfulness, randomisation sanity, reporting and workflow/protocol handling.
  `sakshi-explain` now exposes **ig, gradcam, packets, faithfulness, sanity**. Updated
  the `xai` dependency group and installed it without changing PyTorch or NumPy.
- IG targets the predicted uncalibrated fused-head logit at BERTweet's word embeddings,
  holding image input fixed. It uses 50/150 steps, internal batch 16 and records delta,
  output difference, retry count and convergence flags. Whitespace/BPE alignment handles
  trailing `@@`, unknown tokens and truncation; display alignment fallbacks are logged.
  Span extraction respects the positive threshold, eight-word window, top-three limit,
  privacy placeholders and global positive-attribution denominator.
- Live CLIP shares the existing pinned tower/processor and preserves pixel gradients
  through post-layernorm and 2×2 pooling. Grad-CAM uses the last vision block's
  `layer_norm1`; CLS is dropped only during CAM reshape. Every real image is checked
  against cached tokens at absolute tolerance `1e-2` before explanations. Regions use
  8-connected components, the fixed percentile/area limits and original-image coordinates.
- The packet builder uses calibrated rounded probabilities, same-model explanation
  identity, the existing schema and privacy scanner, plus semantic validation of weights,
  labels, status/gating and evidence boxes/IDs/scores. Gated/unavailable heads get no
  explanation. A noninformative post therefore gets no humanitarian image evidence.
  Diagnostics and heatmap paths stay outside packets. Captions come only from an existing
  §9.4 cache and are marked model-generated; absent captions remain null.
- Faithfulness separately masks text words and 16×16 normalized image patches at the
  five fixed percentages, preserves the other modality, and retains negative deltas.
  Five random rankings are averaged per post; reports include AOPC, curves and post-level
  bootstrap intervals. Cascading sanity randomises head → final cross-attention block →
  gate while holding the original target fixed; parameters and CPU/CUDA/MPS RNG states
  are restored, including failures. Undefined Spearman correlations remain explicitly null.
- Audit requests bind classifier/calibration, manifest/cache, source, package and method
  identities. Test rows require completed frozen classifier evaluation; the protocol is
  frozen before attribution and changed requests are rejected. Atomic per-post records
  allow recovery; sampling predictions are checksummed. Completed audits verify their
  result/report hashes and reuse outputs. Dev reports use `_dev` names and explicitly
  disclaim research conclusions; formal output names are reserved for the specified test
  sample sizes. The CLI publishes generated summaries/curves to `reports/`.
- Corrected WP4 packet context validation for null event/time values (`NaT` is not equal
  to itself). Both nullable and UTC-dated same-model roundtrips are covered by tests.
- Added `docs/EXPLAINABILITY.md`, updated README and clarified the spec's position-ID,
  live-wrapper, backend and diagnostic contracts. Existing classifier artifacts, model
  weights, prepared arrays and calibration values were preserved.

### Real failures found and resolved

1. **Installing torchvision changed image preprocessing.** The initial live/cache check
   raised at maximum error **0.087348**. The differentiable wrapper matched the current
   NumPy encoder within **4.77e-7**, isolating preprocessing. Transformers had selected
   the torchvision backend after the XAI dependency install; the cache used PIL. Explicit
   `AutoImageProcessor(..., backend="pil", use_fast=False)` reduced the same image's
   error to **0.001941** (float16 cache rounding), below `1e-2`. Added a constructor
   regression and an opt-in equivalence test on five real dev images. No cache was rebuilt
   and the tolerance was not relaxed.
2. **Pad baselines moved RoBERTa positions.** The first real word-layer IG had large
   completeness deltas even after 150 steps. Replacing word IDs with pad also changed
   automatically derived position IDs, outside the attributed word-embedding layer.
   A temporary, cleaned-up encoder pre-hook now holds positions fixed, as it does the
   attention mask. The first dev post's informative delta fell to **-0.008812** for
   output difference **0.969588** (50 steps); humanitarian delta was **0.000174** for
   difference **0.543448** (150 steps), both within the prescribed tolerance. Added an
   analytic position-sensitive regression. This was corrected on dev, before any test
   attribution inspection; the trained classifier's normal inference path is unchanged.
3. **OpenCV 5 removed the base Haar detector.** Real overlay export failed rather than
   leaking an unblurred figure. Constrained the `xai` and `app` extras to
   `opencv-python-headless>=4.9,<5`, installed **4.14.0.94**, and added a real default
   detector check plus missing-detector refusal test. Automatic face-blurred export
   subsequently succeeded. The guide links the upstream migration explanation.

### Validation and generated artifacts

- **153 default tests passed, 1 private-data test skipped**, with no model/data download.
  The opt-in private-data test separately **passed on five real dev images** using
  existing cached weights. Ruff lint/format, package dependency checks and whitespace
  checks passed. Built and inspected a wheel under
  `/private/tmp/sakshi-wp3-wheel-check-final`: all 12 explanation modules, schema,
  CLI and console entry point are present.
- Final private validation root: **`runs/explain/wp3-verified-20261024/`**. The packet
  selector contains three dev posts chosen using informativeness, available modalities
  and abstention only, before examining explanations. All three packets are eligible,
  pass `load_packets` and WP4 `validate_selected`, and contain actual explanations from
  `full-s0`. All **12 head explanations and 12 heatmaps** had no diagnostic flags in
  this three-post packet check. Evidence is capped at three spans plus two regions.
- `integration_check.json` records this validation and the **16 original classifier
  artifact hashes still matching the downloaded export**. The face-blurred overlay was
  visually inspected; source-image figures stay private. Standalone `ig` and `gradcam
  --overlays` CLI commands also completed on an eligible dev post.
- Both **faithfulness and sanity ran with one dev post per head**, all four heads, five
  random rankings where applicable and 1,000 bootstrap replicates. Reports are generated
  at `reports/results/faithfulness_dev.md`, `sanity_dev.md`, and
  `reports/figures/faithfulness_curves_dev.png`. Their private records, requests,
  checksummed sampling predictions and per-post artifacts are under the final validation
  root. Each audit's sampling pool contained **4,723 supervised dev rows**.
  A repeat of the completed faithfulness command verified and reused its outputs without
  loading model weights or recomputing attributions. Machine-readable validation and
  runtime versions are recorded in `runs/explain/wp3-verified-20261024/session.json`.
- **Limitations retained:** the sampled humanitarian audit post still has
  `ig_unconverged` after 150 steps; two randomisation stages also flag it. Counts remain
  in the reports. Some image sufficiency comparisons do not beat random; negative
  perturbation deltas and all sanity correlations are kept. With one post per head,
  bootstrap intervals are degenerate and demonstrate report plumbing only. No claim of
  explanation faithfulness, uncertainty quality or research acceptance is made from this.
- Earlier diagnostic roots (`wp3-dev-20261024*`, `wp3-dev-final-20261024`) are preserved
  for failure history. Use **`wp3-verified-20261024`** for handoff. No official test
  predictions/metrics or explanation attributions were computed, no new Kaggle job was
  submitted, and no 7B caption model was downloaded or run. Formal §9.9 research acceptance
  and complete downstream packet coverage remain pending.

## Session — 2026-10-24: WP5 verified generation pipeline

- Implemented `sakshi-generate` with the Appendix E.2 prompt/schema, Ollama structured
  output and CUDA HF Qwen backends, a request-keyed raw/parsed cache, two JSON retries,
  explicit generation failures, post blocks that escape input text, and a model-tokenizer
  prompt budget. The default config is `configs/generation/sitrep.yaml`.
- Added the fixed five-gate verifier: P-ID validity, conservative provenance and
  unverified-origin warnings, NLI entailment/contradiction with dynamic checkpoint label
  indices and long-premise fallback, severity/casualty lexicons with negation, and the
  Appendix E.6 number normaliser. Normalisation splits multi-sentence items, deduplicates
  citations and duplicate sentences. One rewrite round keeps original and rewrite
  rejections with all gate diagnostics. System lines remain WP4-computed.
- Bound each run to WP4's completed artifact hashes and exact validated WP3 packet
  checksum, then rechecked selected packet semantics, model IDs, event/timestamps,
  representative samples, tweet eligibility, rollup severity and conservative
  multi-image provenance. Runs publish per-window sitreps, environment/config/request
  metadata, implementation hashes and a completion marker. Identical reruns reuse
  finished outputs. A three-event evaluation planner freezes balanced, seeded eligible
  24-hour windows before any research generation.
- Added Qwen caption generation into WP3's append-only caption cache, optional
  deterministic-prefix/per-post rationale generation, and blind audit export with
  face-blurred local images. Harvey calibration and Maria/Mexico evaluation have
  separate roles; evaluation export checks that sitreps used the Harvey-calibrated
  threshold. Scoring computes κ, supported precision, weighted unsupported-content
  recall and per-gate counts. `reports/generation/summary.md` is generated from verified
  sitrep artifacts.
- Added `docs/GENERATION.md`, updated README and added 26 focused fixture tests.
  **179 default tests pass, 1 private-data test is skipped**; Ruff lint, format and
  whitespace checks pass. The installed CLI entry point and a wheel containing all
  12 WP5 modules were checked. No LLM/NLI weights were downloaded and no real LOEO sitrep
  or human audit was claimed. The formal §11.11 acceptance remains pending upstream
  LOEO predictions/packets, a local Ollama model, NLI weights and human reviewers.

## Checks and findings

- Latest full suite: **179 passed, 1 private-data test skipped** on 2026-10-24. The
  opt-in five-image check separately passed. Lint, formatting, dependency and whitespace
  checks passed. Re-run appropriate checks after implementation changes.
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

1. Restore and freeze the active Sri Lanka and Mexico jobs, then verify the B1/B2 Harvey
   pilot folds. Complete all remaining standard B1/B2 seeds and seven-event full/B1/B2
   LOEO matrix. Derive each additional B3 run only from calibrated dev predictions.
2. Finish the resumable Harvey selected-post explanations, validate all packets against
   WP4 predictions/provenance, and complete its research aggregate. Repeat for Maria and
   Mexico, then freeze the three-event window plan and run verified sitreps with actual
   Ollama/NLI checkpoints. The formal 500-post-per-head faithfulness and 100-post-per-head
   sanity audits are still required.
3. Implement the remaining B4/B6 baselines, A1–A5 ablations and HumAID transfer checks;
   generate multi-seed estimates and paired comparisons. The full provenance injection
   benchmark is complete, but independent human reviews remain unavailable.
4. Prepare blind review sheets without filling human labels. The provenance and sitrep
   human study metrics require two independent reviewers and an adjudicator.
5. Run `ruff check .`, `ruff format --check .`, `pytest -q` and `git diff --check` after
   code changes. Keep `ranking_demo.md` separate from research `ranking.md`.

## How to resume in another chat

Read this log, `README.md` and the relevant section of `SAKSHI_design_spec_v2.md`. Inspect
`git status --short` and the feature-cache index before running any expensive job. The
latest committed packet-planning checkpoint is `e94508a`; inspect the working tree and
current log before changing code frozen into an explanation protocol. For the GPU
workflow, read `docs/GPU_TRAINING.md`, inspect restored runs and the active private
Kaggle queue under `runs/transfer/kaggle_jobs/`. Do not repeat training just to recover
already saved artifacts.
For WP3, read `docs/EXPLAINABILITY.md` and inspect the final validation root above.
Do not report the `_dev` audit tables as the formal test study.
The original Kaggle bundle contains the pre-WP3 source snapshot. Stage a fresh source
bundle for hosted XAI jobs; preserve the original training version for its compatible
resume state. New source and dependency requirements must not be mixed silently with
an old training recovery checkpoint.

## 26 October 2026, 13:41 IST checkpoint

- The full classifier has three frozen standard-split seeds and all seven LOEO folds.
  B1 text-only and B2 image-only each have three standard-split seeds and all seven
  LOEO folds restored; their complete transfer report is `reports/results/loeo_baselines.md`.
  B3 late fusion remains complete only for standard seed 0. HumAID transfer and the
  432-case computational provenance benchmark are complete.
- The three LOEO explanation packet sets are complete and validated: Harvey 504,
  Maria 666, Mexico 279. All three research aggregates and the balanced 53-window
  sitrep evaluation plan are frozen. Harvey sitrep generation reached 5 windows.
  It was stopped, along with Ollama and local bulk restores, when the laptop became
  hot. Do not count the remaining 48 windows or human review metrics as complete.
- A1–A3 (all nine seeds) and B1/B2 LOEO (all 14 folds) completed on Kaggle. Local
  restoration of A1–A3 stopped after A1 and A2 seed 0. A4 seed 0 and B4 disaster-type
  seed 0 completed and were restored. A4 seeds 1–2 and A5 seed 1 completed remotely;
  the remote queue continues with A5, the corrected B6 VLM notebook, and the remaining
  B4 jobs. The code snapshot for A4/A5/B4 remains on the `work/ablations` worktree;
  `scripts/prepare_experiment_code.py` records its source hashes. The B6 notebook
  generator and evaluator are on main. Its first Kaggle attempt failed before inference
  because it lacked notebook kernel metadata; that was fixed and B6 was requeued.
- `runs/transfer/kaggle_outputs/experiment-dispatch.log` is the live, detached GPU
  queue log. The dispatcher checks Kaggle status and submits at most two concurrent
  jobs. `runs/transfer/kaggle_outputs/baseline-restore.log` records the paused local
  restores. The sitrep output and generation log are under
  `runs/sitreps/full-s0-loeo_hurricane_harvey/hurricane_harvey/`. These large/private
  run artifacts are Git-ignored. Do not start a second dispatcher or sitrep generator
  before checking the process table and completion markers.
- The first priority after the laptop cools is to restore the remaining frozen runs,
  render `reports/results/experiments_three_seed.md`, inspect/retry B6 if needed, and
  derive B3 seeds 1–2 and LOEO folds. The formal WP3 faithfulness/sanity audits and
  the remaining sitreps are still pending. There are no independent reviewers;
  provenance and sitrep human-study outcomes must remain unreported.

## 26 October 2026, 18:23 IST checkpoint

- The 13:41 checkpoint above is superseded for classifier status: B3 has all three
  standard seeds and seven LOEO folds, with frozen reports in `reports/results/`.
  A1–A5 each have all three standard seeds, and B4 has all 12 single-head runs
  restored and officially evaluated. Their paired three-seed report is
  `reports/results/experiments_three_seed.md`. The A4 and B4 DMD severity comparisons
  now include an equal-coverage forced-argmax row; ordinary gated-versus-ungated
  differences must not be interpreted as isolated gains in discrimination.
- The private B6 zero-shot VLM job `varun0636/sakshi-b6-vlm-zero-shot` is still running
  on Kaggle. Do not report B6 metrics until its full 6,939-post output has been
  downloaded and checked against the frozen official selector union.
- The local Ollama sitrep run stopped after five Harvey windows because the laptop
  became hot. A private three-event WP5 input dataset is ready on Kaggle. The
  pinned Qwen2.5-7B-Instruct + DeBERTa NLI Harvey job is running on Kaggle; Maria
  and Mexico jobs are queued to start as GPU slots free up. The new queue's live
  output is `runs/transfer/kaggle_outputs/sitrep-dispatch.log`. Its outputs go under
  `runs/sitreps_hf/` and must be validated before generating the WP5 summary.
- The old generated `runs/transfer/gpu-inputs-20261024.zip` upload archive was
  removed for disk headroom after verifying that its source directory and the ready
  private Kaggle dataset still exist. Local training and local LLM inference are off.
- Formal WP3 faithfulness/sanity audits and the independent provenance/sitrep human
  studies remain incomplete. No reviewers are available; do not invent human labels
  or inter-rater statistics. The research artifacts and blind review sheets can be
  prepared without claiming those studies are complete.

## 26 October 2026, 19:30 IST checkpoint

- All three private Kaggle WP5 HF jobs completed. The hash-checking collector restored
  all 53 frozen windows (Harvey 18, Maria 26, Mexico 9). The computed report is
  `reports/generation/summary_hf.md`, with interpretation in
  `reports/generation/README.md`: initial verifier acceptance 0.860, final 0.910,
  evidence coverage 0.310, and 13 recorded generation failures (11 system-only
  windows; two failed rewrites with retained verified round-0 claims). These are
  automatic verifier results with uncalibrated 0.5/0.3 thresholds, not human scores.
- The 50-sentence Harvey calibration sheet and face-blurred local viewer were generated.
  The sheet contains private post text, is Git-ignored and has no filled reviewer
  labels. Human calibration/evaluation remain pending because no reviewers exist.
- The local dashboard now prefers the completed HF research sitreps over the partial
  Ollama run and shows the model and uncalibrated-review status in readable form.
  Ten app tests, Ruff and JavaScript syntax checks passed after this change.
- B6 zero-shot VLM remains the only active Kaggle GPU job. The low-CPU collector will
  download and score it on completion; do not report B6 metrics before its full
  6,939-post output passes the frozen-selector check.

## 27 October 2026, 07:25 IST checkpoint

- B6 completed all 6,939 official-selector-union posts on Kaggle with the pinned
  7B Qwen2.5-VL model in NF4. The collector verified the protocol/manifest, every
  sample ID, all four labels, latency and the frozen majority fallback before scoring.
  The JSON/repair failure rate was high: 2,235 posts (32.2%). The truthful baseline
  report is `reports/results/b6_vlm.md`; the paired seed-0 comparison with the full
  model is `reports/results/comparison_seed0_b6.md`, including equal-coverage DMD
  severity. These results must not be described as error-free zero-shot inference.
- A private formal WP3 audit input dataset contains the full seed-0 checkpoint and
  source snapshot; it reuses the official-image and feature-cache datasets already
  on Kaggle. A T4 timing pilot completed: faithfulness took 151 seconds for two
  posts per head, and sanity took 40 seconds for one post per head. These are timing
  checks, not the formal study. Eight head-specific 500/100-post jobs were built.
  Informative faithfulness and sanity are running; a two-slot dispatcher queues the
  other six. `runs/transfer/kaggle_outputs/formal-xai-dispatch.log` and
  `formal-xai-collect.log` record status and validated recovery. No laptop model
  inference is running.
- Independent human reviews are still unavailable. The provenance and sitrep human
  studies, including Harvey threshold calibration and κ, remain unscored.

## 27 October 2026, 11:00 IST checkpoint

- All eight formal WP3 GPU jobs completed and the collector restored their exports
  without failure. Each shard was checked against the frozen request, source protocol,
  classifier checkpoint, sampling predictions, post IDs and completion hashes before
  merging. The final official-test samples are **500 per head for faithfulness** and
  **100 per head for randomisation sanity** (2,000 and 400 sampled posts). Gated rows
  remain in the denominator/accounting and are marked unavailable for explanation.
- The formal reports are `reports/results/faithfulness.md` and
  `reports/results/sanity.md`; the figure is
  `reports/figures/faithfulness_curves.png`. Every available head/modality has a
  higher mean AOPC comprehensiveness and lower mean AOPC sufficiency than its random
  baseline. Sanity correlations and unconverged-IG flags are reported without hiding
  ambiguous cases. The full audit queue and collector exited successfully.
- The core classification, provenance, aggregation, sitrep and explanation benchmarks
  are now complete and reproducible to the scope described in the reports. The design
  spec's independent human provenance/sitrep
  studies still cannot be scored without two reviewers and an adjudicator. The local
  Ollama end-to-end acceptance run stopped at five Harvey windows because the laptop
  overheated; the complete 53-window research generation used the documented pinned
  Kaggle HF backend. Do not describe either outstanding item as completed.

## 27 October 2026, 12:30 IST checkpoint

- Three private Qwen2.5-VL-7B Kaggle jobs generated captions for all **1,449** frozen
  Harvey, Maria and Mexico packet images. The collector checked event IDs, the pinned
  model revision, Appendix E.5 prompt hash, shard hashes, uniqueness and packet privacy
  before publishing the private cache. The model exceeded the prompt's 20-word limit
  on 248 captions; outputs were retained verbatim as required. See
  `reports/generation/captions.md`. Captioned packets were derived from the original
  frozen explanations and schema validated. Recomputed WP4 windows and all 53 selected
  window hashes exactly match the original no-caption study.
- Three more private Kaggle jobs generated and restored the **53 captioned sitreps**.
  The captioned automatic verifier rate is **0.852 initial / 0.895 final**, with 11
  generation failures, eight system-only windows and 0.313 evidence coverage. The
  no-caption run remains **0.860 / 0.910**, with 13 failures and 0.310 coverage.
  Mexico's captioned run degraded to 33 round-0 sentences and five invalid-JSON
  failures from 79 sentences and zero failures without captions. The failure cases
  remain in the reports; no evaluation-driven prompt or threshold change was made.
- A pinned DeBERTa GPU ablation rescored G3 on the same 371 captioned round-0
  sentences. Full five-gate initial acceptance is **0.852**, G3-only is **0.889**,
  and removing captions only from G3 gives **0.668** with 75 changed G3 decisions.
  The final full rate after the original rewrite is **0.895**. The output and its
  source, packet, sitrep and plan hashes were checked before publication in
  `reports/generation/ablations_captioned.md`.
- The custom dashboard now selects completed captioned LOEO aggregates and sitreps
  by default. Its adapter test and real-artifact sitrep/replay smoke checks pass.
  All remote jobs have completed; the laptop has no active model inference process.
  The independent human provenance/sitrep reviews and the full local Ollama acceptance
  remain outstanding. The automatic verifier rates are not human-rated factual accuracy.

## 27 October 2026, professor-evaluation preflight

- The app was selecting the most recently restored classifier checkpoint, which was a
  B4 single-head ablation. It now selects the frozen full standard seed-0 run with a
  completed official evaluation. A selection regression test covers the case.
- The pinned BERTweet and CLIP weights were restored to the default Hugging Face cache
  after a temporary project cache was cleared. The app no longer depends on that
  temporary path. Offline model loading, image-and-text live classification, real
  Maria replay/sitrep views, 12 dashboard tests and Ruff all passed.
- The professor demo is technically runnable. Formal WP2/WP5 human studies and the
  full laptop Ollama end-to-end acceptance remain outside the completed evidence.
- A live browser walk-through opened all five workspaces. The Maria replay showed
  real posts and a recycled-image comparison; the captioned Maria sitrep showed
  accepted and rejected claims, gate reasons and an opening citation. The review
  queue and the three-seed results report loaded. The review preview initially
  rendered missing CIB text as `nan`; it now shows image-only posts cleanly and
  prioritizes records with post/event context. Thirteen dashboard tests pass.

## 27 October 2026, professor presentation pass

- Replaced the Investigate cover's decorative graphic with a real, face-blurred Maria
  versus Irma image pair from the frozen provenance replay. It is explicitly labeled
  as a system flag awaiting human review. A single button runs that fixed local case
  through the four-head classifier and temporal provenance check; the browser flow
  returned the earlier indexed Irma image and an abstained classifier head.
- Brought the Maria sitrep's automatic verification verdict into its first viewport,
  exposed the original-claim denominator and failed rewrites, and added a jump to
  rejected claims. The Results cover now charts four actual mean macro-F1 scores from
  the committed official three-seed report, including the weak DMD severity result.
- Added a five-minute walkthrough to `docs/DASHBOARD.md`. The missing independent human
  studies and full local Ollama acceptance remain limitations for professor evaluation.

## 27 October 2026, fieldroom visual redesign

- Replaced the warm white, orange, and green presentation palette with a navy, cobalt,
  slate, and rose system across the custom fieldroom, favicon, README cover, and screenshots.
- Added a default four-scene **Present** route: the dated Maria–Irma image pair, a live
  four-head classifier run with abstention and earlier indexed image result, a Maria
  sitrep claim comparison with automatic verifier gates, and the frozen three-seed
  macro-F1 chart. Scene navigation uses buttons and arrow keys; the five workspaces
  remain available for deeper inspection.
- Checked the live classifier path in Chrome. It returned all four task heads, an
  abstention, and an earlier indexed Irma appearance. Checked the desktop presentation,
  all five workspace treatments, and the 390 px mobile briefing without horizontal
  overflow. The presentation remains a research interface; the human studies and full
  local Ollama acceptance are still outstanding.

## 27 October 2026, explanation presentation pass

- Made the Investigate XAI control explicit and replaced the empty word-attribution panel
  for image-only heads with a clear method note and a larger side-by-side source/Grad-CAM
  comparison. Text-capable heads now display signed Integrated Gradients words with a
  legend alongside image focus when both are available.
- Removed the disabled evidence-rationale control from the user flow and clear an old
  explanation when a different head is selected. Checked live damage-severity Grad-CAM
  and informative-head Integrated Gradients plus Grad-CAM in Chrome; the phone-width
  layout has no horizontal overflow.
- Follow-up from the image-only flood inspection: the repeated source/overlay layout
  obscured differences between heads. The Investigate view now leads with each head's
  actual predicted label and calibrated probability, keeps revealed maps in memory, and
  compares two or more class-target maps side by side. It explicitly warns that Grad-CAM
  maps are normalized separately, so color strength should not be compared across heads.
  This supersedes the earlier clear-on-change presentation.
- Simplified the minor-project presentation route: the four-scene briefing now uses plain
  explanations for the image match, model decisions, automatic claim check and weak task.
  The XAI view leads with what the colors mean and places Grad-CAM/Integrated Gradients
  details behind a disclosure. Refreshed the README briefing screenshot to match the
  simplified claim scene. Public dashboard documentation describes the interface;
  presenter notes are maintained separately from this repository.
