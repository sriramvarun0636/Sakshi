# SAKSHI

Implementation progress and handoff notes: [build log](docs/BUILD_LOG.md).

**Situational Awareness through Knowledge Synthesis and Human-Interpretable AI** — multi-task,
explainable disaster intelligence from social media. SAKSHI classifies crisis posts (text and
image), checks whether an image has been recycled from an earlier disaster, and generates
situation reports in which every sentence is cited and verified.

## Status

| Component | Status |
|---|---|
| Data preparation: download, harmonisation, deduplication, leakage-free splits | complete — see [reports/data](reports/data/README.md) |
| WP0 foundations: shared types, text/image loading, evaluation helpers, CLIP cache code | implemented; full 35,685-image CLIP cache built and checked |
| WP1 classifier: fusion network, masked focal loss, training data, calibration, prediction, official evaluation | implemented and CPU smoke tested; full model run pending; B5 CLIP zero-shot baseline evaluated |
| WP2 temporal provenance: index, graphics filter, decisions, replay | implemented; real 18,083-query replay complete; full injection benchmark pending |
| Explanations and verified report generation | planned |

## Installation

Requires Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Data

Datasets are not redistributed with this repository. They are downloaded to `data/`, which is git-ignored.

```bash
sakshi-data download     # CrisisMMD, Crisis Image Benchmark (AIDR-DT, DMD), HumAID — ~5.3 GB, resumable
sakshi-data prepare      # extract → manifests → hash → features (if ML installed) → dedupe → audit → splits → report
```

Individual steps can be re-run, e.g. `sakshi-data prepare dedupe splits`.
To build the CLIP feature cache, install `pip install -e ".[dev,ml]"` and run
`sakshi-data prepare features` after the manifest and hash steps. This writes several GB
under the ignored `data/processed/features/` directory.

After the cache is ready, train the first classifier run with:

```bash
sakshi-model train --config configs/classifier/full.yaml --seed 0
```

The command saves the best dev checkpoint, class weights, label order, per-epoch logs and
run metadata under the ignored `runs/classifier/` directory. The full model is intended
for a CUDA GPU; CPU tests use a small synthetic encoder and do not download model weights.
After training, fit dev calibration and write test predictions:

```bash
sakshi-model calibrate --run runs/classifier/<run_id>
sakshi-model predict --run runs/classifier/<run_id> --rows official
sakshi-model evaluate --run runs/classifier/<run_id>
```

The image-only B5 baseline uses the spec's fixed CLIP prompts and the same official
evaluation selectors. It has been run; see
[reports/results/baselines_clip_zeroshot.md](reports/results/baselines_clip_zeroshot.md).
To reproduce it after building the feature cache:

```bash
sakshi-baselines clip-zeroshot --bootstrap 1000
```

After the feature cache is complete, build and replay the temporal provenance index:

```bash
sakshi-provenance build-index
sakshi-provenance replay --event all
```

The replay summary is at [reports/provenance/replay_summary.md](reports/provenance/replay_summary.md).
The closed-world injection benchmark is implemented but has not been run end to end; it
requires substantial CLIP inference and plotting compute:

```bash
sakshi-provenance benchmark --config configs/provenance/benchmark.yaml
```

For the blind two-rater provenance study, generate the sheet and local image viewer with
`sakshi-provenance label-export`. The sheet is at
[reports/provenance/labelling_sheet.csv](reports/provenance/labelling_sheet.csv); the image
viewer is at `runs/provenance/labelling.html` and is ignored by git. Once `label_a`,
`label_b` and `final` contain valid labels for every pair, run
`sakshi-provenance label-score` to write the study report. The export command refuses to
overwrite any sheet with human labels.

| Output | Contents |
|---|---|
| `data/processed/manifest.parquet` | one row per image (35,685): harmonised labels `y_*`, unified `split`, official `split_<task>`, leave-one-event-out roles `loeo_<event>` |
| `data/processed/near_duplicates.parquet` | image pairs within pHash distance 12 |
| `data/processed/humaid_exclude.csv` | HumAID tweets that repeat CrisisMMD tweets, to be excluded from evaluation |
| `reports/data/README.md` | data preparation and audit report |
| `reports/data/tables.md` | generated statistics behind the report |
| `reports/data/provenance_candidates.csv` | cross-event image matches for manual provenance verification |

**Use the `split` column for training and model selection.** The official CrisisMMD splits
were drawn independently per task, and 3,704 images are in the test set of one task and the
training set of another. The unified split keeps every official test set intact: to report
results comparable with published work, evaluate each head on the rows where
`split_<task> == "test"`, all of which are held out.

## Repository layout

```
src/sakshi/
  cli.py              sakshi-data command
  model_cli.py        sakshi-model command
  paths.py            filesystem layout
  config.py           strict experiment config, seeds and run metadata
  data/
    download.py       resumable parallel downloads
    crisismmd.py      CrisisMMD manifest and consistency checks
    cib.py            Crisis Image Benchmark manifest and disaster-type label audit
    humaid.py         HumAID loading and overlap detection
    labels.py         canonical label spaces and mappings
    events.py         event metadata and tweet-timestamp decoding
    phash.py          perceptual hashing, near-duplicate search, threshold calibration
    splits.py         unified split resolution and leave-one-event-out folds
    pipeline.py       data preparation pipeline
    report.py         generated statistics tables
  models/
    data.py           manifest-to-batch adapter and weighted sampler
    fusion.py         multimodal network
    losses.py         class-balanced focal loss and label masks
    calibration.py    temperature and abstention fitting
    calibrate_run.py   dev calibration artifacts
    predict.py        hierarchical final decision rules
    train.py          classifier training loop and dev selection
    evaluate_run.py   official test-set evaluation
  eval/               selectors, metrics, bootstrap and tables
  provenance/         image index, graphic scores, temporal decisions and replay
tests/                unit tests (no dataset download required)
reports/data/         data preparation report and generated statistics
```

## Development

```bash
pytest            # unit tests
ruff check .      # lint
ruff format .     # format
```

## Data sources and licences

- **CrisisMMD v2.0** — Alam, Ofli & Imran, ICWSM 2018. [crisisnlp.qcri.org/crisismmd](https://crisisnlp.qcri.org/crisismmd)
- **Crisis Image Benchmark** — Alam et al., ASONAM 2020. Harvard Dataverse, [doi:10.7910/DVN/QXT5QL](https://doi.org/10.7910/DVN/QXT5QL). The DMD source is by Mozannar, Rizk & Awad, ISCRAM 2018.
- **HumAID** — Alam et al., ICWSM 2021. [crisisnlp.qcri.org/humaid_dataset](https://crisisnlp.qcri.org/humaid_dataset)

All datasets are released for non-commercial research use under their providers' terms.
Review those terms before sharing any derived data.
