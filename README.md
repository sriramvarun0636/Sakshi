# SAKSHI

**Situational Awareness through Knowledge Synthesis and Human-Interpretable AI** — multi-task,
explainable disaster intelligence from social media. SAKSHI classifies crisis posts (text and
image), checks whether an image has been recycled from an earlier disaster, and generates
situation reports in which every sentence is cited and verified.

## Status

| Component | Status |
|---|---|
| Data preparation: download, harmonisation, deduplication, leakage-free splits | complete — see [reports/data](reports/data/README.md) |
| Multi-task multimodal classifier | planned |
| Temporal provenance detection | planned |
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
sakshi-data prepare      # extract → manifests → hash → dedupe → audit → splits → report
```

Individual steps can be re-run, e.g. `sakshi-data prepare dedupe splits`.

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
  paths.py            filesystem layout
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
