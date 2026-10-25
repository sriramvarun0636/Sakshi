# Classifier training on a hosted GPU

Use [the standard-split notebook](../notebooks/train_classifier_kaggle.ipynb) to train the
standard-split classifier, calibrate on dev, and export its artifacts. Use the separate
[LOEO notebook](../notebooks/train_classifier_loeo_kaggle.ipynb) for each held-out event.
The [standard baseline notebook](../notebooks/train_classifier_baseline_kaggle.ipynb)
and [LOEO baseline notebook](../notebooks/train_classifier_baseline_loeo_kaggle.ipynb)
run B1 text-only and B2 image-only from explicit configs. B1 uses the per-modality text
labels and scores informative/humanitarian only. B2 uses image labels plus severity and
disaster type; both use zero modality dropout. Their active heads, calibration, predictions
and frozen evaluators follow the same setting. Train seeds 0–2 on the standard split and
one seed per seven held-out events; derive B3 late fusion from their calibrated dev
probabilities after the pairs complete. These baseline runs are pending hosted GPU time.
The first hosted
CUDA run completed on 2026-10-24 and was restored locally as
`runs/classifier/full-s0-20261024-062724/`: 14 minutes of training on one T4, best epoch
5 of 8, batch 32, fp16. See the [dev run report](../reports/results/classifier_seed0_dev.md).
The seed 0 official test is now frozen and scored; see
[`classifier.md`](../reports/results/classifier.md). CPU tests exercise the workflow with a
small encoder; the hosted run also passed the full encoder's CUDA smoke.

## 1. Prepare the input bundle locally

Install the CLI and build a snapshot from the completed feature cache:

```sh
source .venv/bin/activate
python -m pip install -e '.[dev,ml]'
sakshi-gpu bundle --output runs/transfer/gpu-inputs-20261024
```

Use a fresh output path when rebuilding. The command verifies all arrays, split isolation
and the spec's fixed selector/training-label counts before copying. It needs about 2.9 GB
for the copied inputs plus 1 GiB free headroom. It performs no network upload.

The folder contains:

```text
bundle.json                  source identity and per-file sizes/SHA256 hashes
SHA256SUMS                   checksums for source and prepared inputs
preflight.json               local data checks
source/                      Python package, configs, notebook, tests and docs
  source_provenance.json     original Git commit and working-tree status
prepared/
  manifest.parquet
  features/
    features_index.parquet
    clip_vitb16_tokens.f16.npy
    clip_vitb16_embed.f32.npy
```

Only these prepared files and allowlisted source files are copied. Raw images, `.git`,
local environment files, model downloads and experiment directories are excluded.
The manifest contains tweet text and metadata: keep the whole bundle private. Source
snapshots may include uncommitted work; the inventory records their exact bytes, while
the Git commit and dirty flag identify the starting revision. After changing code or the
notebook, prepare a new snapshot so the attached source and notebook agree.

An existing cache can be checked independently:

```sh
sakshi-gpu check --config configs/classifier/full.yaml --output runs/gpu-preflight.json
```

The current release has 35,685 cached usable images, 19,896 training rows and 5,720 dev
rows. Main-head training counts must be 6,006 / 1,788 / 4,077 / 11,105. Every auxiliary
head has 8,729 training labels. The checks use sample IDs, so manifest order may differ
from cache order.

## 2. Attach to a private Kaggle notebook

1. Upload the prepared folder as a **private** Kaggle Dataset. Preserve `source/`,
   `prepared/` and the top-level inventory. Do not flatten the files.
2. Import `notebooks/train_classifier_kaggle.ipynb` into a **private** notebook, attach
   the dataset, enable Internet and choose a GPU accelerator.
   Kaggle may require account phone verification before enabling GPU and Internet.
   A ZIP of the bundle can be uploaded with directory structure intact. Wait for dataset
   processing to finish, then attach it from **Your Work → Datasets**.
3. Leave `CONFIG_PATH = "configs/classifier/full.yaml"`, `SEED = 0`,
   `RESUME_RUN = None` and `FINAL_EVALUATION = False` for the first run.
4. Run cells in order. If multiple input bundles are attached, set `BUNDLE_ROOT`
   explicitly to the directory containing the intended `bundle.json`.

The notebook discovers the bundle under `/kaggle/input`, verifies every listed file,
copies only source to `/kaggle/working/sakshi/source`, and mounts the prepared arrays
without copying them. `SAKSHI_PROCESSED_DIR` must be set **before** any `sakshi` import;
the notebook does this. Restart the kernel when switching bundles.

Installation constrains PyTorch to the hosted preinstalled version (at least 2.6), then
checks CUDA availability. It does not install a replacement CUDA stack. Public BERTweet
weights and tokenizer are pinned to revision `b349c1243407b0dcffeabb2337497477286e27ab`.
The pinned weights use PyTorch's `.bin` format; current Transformers requires PyTorch 2.6
or later for that loading path (see the
[Transformers loader check](https://github.com/huggingface/transformers/blob/v4.52.0/src/transformers/utils/import_utils.py)).
The existing CLIP cache uses the pinned revision recorded in `data/features.py`; the CLIP
model is not loaded during training. Resolved package versions are recorded in `env.json`
and the completed run's `requirements.lock.txt`.

Kaggle's current accelerator/settings, input and saved-output behavior are documented in
[Notebooks](https://www.kaggle.com/docs/notebooks) and
[Packages](https://www.kaggle.com/docs/packages). The notebook writes all generated files
under `/kaggle/working/sakshi/`; save the notebook version **with outputs** before ending
the session. Weekly availability and runtime limits depend on the account.

## 3. Smoke, train and calibrate

The CUDA smoke uses actual **training** rows, the full pinned encoder, the configured
microbatch size, and padding to 128 tokens. It checks finite head outputs, missing-modality
weights and a forward/backward/optimizer update for both, text-only and image-only input.
An initial fp16 scale overflow can retry up to eight times. Smoke weights are discarded;
the full run starts from a freshly seeded pretrained model.

If smoke reports out of memory, restart the session and choose:

```python
CONFIG_PATH = "configs/classifier/t4.yaml"
```

This explicitly records batch 16 with accumulation 2 and model name `full-t4`. The default
is batch 32 with accumulation 1. Both follow the spec's effective batch of 32. There is
no silent configuration change on memory failure. One GPU is used even if the host
exposes several.

Training selects the best dev score over the four main heads, then calibration fits
temperatures and abstention thresholds on dev only. Outputs include:

| Artifact | Purpose |
|---|---|
| `checkpoint.pt` | best dev model; inference and final evaluation |
| `last.pt` | last completed epoch, current model, optimizer, scheduler, scaler, all relevant RNG states, sampler state and best model |
| `config.yaml`, `env.json`, `inputs.json` | resolved configuration, environment and input/implementation hashes |
| `label_maps.json`, `class_weights.json` | canonical output order and weights from training counts |
| `train_log.jsonl`, `metrics_dev.json` | per-epoch losses/counts/updates/dev metrics and selected dev result |
| `tokenization.json` | normalized text-row and truncation counts per split, including special tokens |
| `training_summary.json` | completed/best epochs, session wall time, cumulative epoch compute, precision, effective batch, peak CUDA bytes |
| `temperatures.json`, `thresholds.json`, `calibration_metrics.json` | dev calibration, fallbacks and review-threshold results |
| `gpu_preflight.json`, `gpu_smoke.json`, `requirements.lock.txt` | hosted checks and installed versions |

Wall time includes setup, input hashing and checkpoint writes. Epoch compute includes
training and dev inference. Resumed runs record this session's wall time separately from
cumulative completed-epoch compute. Review `nonfinite_batches` and `overflow_updates` in
the epoch log; an epoch with no finite optimizer update is rejected.

Gradient scaling is retained between epochs and included in resume state. Gradients are
unscaled before clipping, overflowed steps are skipped, and the learning rate schedule
advances only on an optimizer update. The precision check requires native bf16 support;
T4 uses fp16 with gradient scaling. This follows the
[PyTorch AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html).

## 4. Resume an interrupted session

`last.pt` is published atomically only after a completed train/dev epoch. If a session ends
mid-epoch, that epoch is repeated. If it ends before the first saved epoch, start a fresh
run. Best checkpoint, dev metrics and logs are reconstructed from the saved state when
recovering, including an interruption during publication of those smaller artifacts.

On Kaggle, attach the previous private notebook output and copy the **entire run directory**
into a new writable directory under `/kaggle/working/sakshi/runs/classifier/`. Set
`RESUME_RUN` to that copied path and rerun the notebook with the same bundle, config,
seed, package versions and GPU type. The source and all prepared files must have identical
hashes. Preserve the hosted image/package versions recorded in `env.json`; use the saved
package list when available. No clone token or secret is required for the bundled source.

The CLI supports the same recovery:

```sh
SAKSHI_PROCESSED_DIR=/path/to/bundle/prepared \
  sakshi-model train --config runs/classifier/<run_id>/config.yaml \
  --resume runs/classifier/<run_id>
```

Do not pass `--output-dir` with `--resume`. Configuration/input/runtime mismatches fail;
parameter changes need a new run. A run already frozen for final evaluation cannot be
resumed or recalibrated.

## 5. Score official tests after freezing the configuration

Keep `FINAL_EVALUATION = False` during dev review. Once the configuration is final, enable
it in the parameter cell, execute that cell, then execute the final-evaluation cell.
The helper writes `evaluation_request.json` **before** any test inference, freezing the
checkpoint, configuration, calibration, canonical label maps, manifest, feature bytes and
bootstrap count. It then scores all official test selectors with 1,000 replicates.

Expected selector counts are 1,534 / 955 / 529 / 1,138 / 3,213. The evaluator asserts them,
checks the hierarchy, and writes `predictions_official.parquet`, `metrics_test.json` and
an artifact-hash completion marker. An identical repeat verifies and reuses the stored
results. Changed frozen inputs or changed output artifacts are rejected. Recovery after
metrics publication does not rerun scoring. Do not tune against test scores.

Both entry points use this freeze/reuse workflow:

```sh
sakshi-model evaluate --run runs/classifier/<run_id>
sakshi-gpu final-evaluate --run runs/classifier/<run_id>
```

LOEO is deliberately rejected by the global official evaluator: a LOEO model may have
trained on other events' official rows. For a completed run trained with `--fold
loeo_<event>`, use `sakshi-model evaluate-loeo --run runs/classifier/<run_id>`. The separate
evaluator freezes its input hashes, predicts exactly the held-out CrisisMMD event and
scores informative, humanitarian and both gold-gated and predicted-gated severity with
1,000 bootstrap draws. It writes `predictions_loeo.parquet`, `metrics_loeo.json` and
completion hashes. These are event-specific results, never pooled with the unified
official-test score. Existing baseline results are not rescored by this workflow.

The LOEO notebook sets `FOLD` explicitly, runs fold-aware preflight and calibration, and
uses the held-out-event evaluator before exporting the run. Keep one private notebook and
one model ID per event. The seven folds and their local preflights were verified against
the prepared manifest; GPU results must be recorded only after each hosted run completes.

## 6. Export, download and restore

The final cell creates a checksummed ZIP split into parts of at most **768 MiB**. All parts
plus `export.json` are needed to restore. The export includes the best checkpoint and
allowlisted run metadata/calibration/results; feature arrays and raw images are excluded.
`last.pt` is optional because it includes the larger optimizer and recovery state.

```sh
sakshi-gpu export --run runs/classifier/<run_id> --output runs/exports/<run_id>
# Add --include-resume when portable epoch recovery is needed.
sakshi-gpu restore --export /path/to/downloaded-export --output runs/classifier/<run_id>
```

Restore verifies part ordering, sizes and hashes, verifies the exact archive member list,
then verifies each artifact before publishing the restored directory. It rejects an
existing output directory. The full run and exports stay under ignored `runs/`.

In Kaggle's Output file tree, expand `sakshi/exports/<export_id>/` and download the
individual archive parts. A browser may rename a `.zip.part001` download to `.zip.zip`;
verify its exact byte count and SHA256 against `export.json`, then copy it to the declared
part filename before restoring. JSON may open as a separate page; save its contents as
`export.json`. Avoid downloading the entire notebook output: it also contains the Hugging
Face cache and the much larger `last.pt` recovery state.

When the first hosted run finishes, update `docs/BUILD_LOG.md` with the actual GPU,
wall time, memory, best epoch, dev calibration findings and final-evaluation status. Update
the spec's compute estimates using measured timings. Then proceed with WP3 explanations,
same-model packets, and the remaining seed/baseline/LOEO experiments.
