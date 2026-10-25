"""Frozen text-only transfer evaluation on HumAID's unseen and shared test events."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sakshi import atomic, paths
from sakshi.compute.files import hash_file
from sakshi.config import load
from sakshi.data import labels
from sakshi.data.humaid import read_tweets
from sakshi.models.evaluate_run import _score
from sakshi.models.predict import Predictor
from sakshi.models.train import TrainConfig
from sakshi.types import Post

UNSEEN = {
    "canada_wildfires_2016",
    "cyclone_idai_2019",
    "ecuador_earthquake_2016",
    "hurricane_matthew_2016",
    "italy_earthquake_aug_2016",
    "kaikoura_earthquake_2016",
}
SHARED = {
    "hurricane_harvey_2017",
    "hurricane_irma_2017",
    "hurricane_maria_2017",
    "puebla_mexico_earthquake_2017",
    "srilanka_floods_2017",
}
STRICT = {
    "infrastructure_and_utility_damage": "infrastructure_and_utility_damage",
    "injured_or_dead_people": "affected_individuals",
    "missing_or_found_people": "affected_individuals",
    "displaced_people_and_evacuations": "affected_individuals",
    "rescue_volunteering_or_donation_effort": "rescue_volunteering_or_donation_effort",
    "other_relevant_information": "other_relevant_information",
    "not_humanitarian": "not_humanitarian",
}
FULL = STRICT | {
    "caution_and_advice": "other_relevant_information",
    "sympathy_and_support": "other_relevant_information",
    "requests_or_urgent_needs": "rescue_volunteering_or_donation_effort",
}


def evaluation_rows(humaid: pd.DataFrame, excluded: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixed overlap exclusion before choosing the two test event sets."""
    required = {"tweet_id", "tweet_text", "class_label", "event", "split"}
    if required - set(humaid) or {"tweet_id", "event", "split"} - set(excluded):
        raise ValueError("HumAID or overlap records are missing required columns")
    if len(humaid) != 41152 or len(excluded) != 261 or humaid.tweet_id.duplicated().any():
        raise ValueError("HumAID or overlap inventory differs from the fixed release")
    if set(humaid.event) != UNSEEN | SHARED or set(humaid.class_label) != set(FULL):
        raise ValueError("HumAID events or labels differ from the fixed release")
    ids = set(excluded.tweet_id.astype(str))
    if len(ids) != 261 or not ids <= set(humaid.tweet_id.astype(str)):
        raise ValueError("HumAID overlap IDs are duplicated or unknown")
    selected = humaid.loc[(humaid.split == "test") & ~humaid.tweet_id.astype(str).isin(ids)].copy()
    selected["sample_id"] = "humaid:" + selected.tweet_id.astype(str)
    selected["group"] = np.where(selected.event.isin(UNSEEN), "unseen", "shared")
    if len(selected) != 8112 or selected.group.value_counts().to_dict() != {"shared": 5575, "unseen": 2537}:
        raise ValueError("HumAID test roles differ from the fixed post-overlap counts")
    if selected.tweet_text.isna().any() or selected.tweet_text.astype(str).str.strip().eq("").any():
        raise ValueError("HumAID test rows need usable tweet text")
    return selected.reset_index(drop=True)


def score_humaid(rows: pd.DataFrame, predictions: pd.DataFrame, *, bootstrap: int = 1000) -> dict:
    """Report strict and full mappings separately on both fixed event groups."""
    if bootstrap < 1 or predictions.sample_id.duplicated().any() or set(predictions.sample_id) != set(rows.sample_id):
        raise ValueError("HumAID predictions must cover the fixed test rows once each")
    aligned = rows.merge(predictions, on="sample_id", how="left", validate="one_to_one", indicator=True)
    if not (aligned._merge == "both").all():
        raise ValueError("HumAID predictions are misaligned")
    result = {"n_test": len(rows), "groups": {}}
    for group in ("unseen", "shared"):
        result["groups"][group] = {}
        for mapping_name, mapping in (("strict", STRICT), ("full", FULL)):
            subset = aligned.loc[(aligned.group == group) & aligned.class_label.isin(mapping)].copy()
            gold_hum = subset.class_label.map(mapping).astype(str).to_numpy(dtype=object)
            gold_inf = np.where(gold_hum == "not_humanitarian", "not_informative", "informative")
            heads = {}
            for head, classes, gold in (
                ("informative", labels.INFORMATIVE, gold_inf),
                ("humanitarian", labels.HUMANITARIAN, gold_hum),
            ):
                guessed = subset[f"{head}_label"].astype(str).to_numpy(dtype=object)
                probabilities = subset[[f"p_{head}__{name}" for name in classes]].to_numpy(dtype=float)
                heads[head] = _score(gold, guessed, classes, probabilities, bootstrap)
            result["groups"][group][mapping_name] = {"n": len(subset), "heads": heads}
    return result


def run_humaid(run: Path, output: Path, *, device: str = "auto", bootstrap: int = 1000) -> Path:
    """Freeze inputs, resume per-batch text inference and score only after all predictions exist."""
    run, output = Path(run), Path(output)
    config = load(run / "config.yaml", TrainConfig)
    if config.fold is not None or config.model.modality != "both":
        raise ValueError("HumAID robustness uses a standard full-model run with text-only input")
    if device not in ("auto", "cpu", "mps", "cuda") or bootstrap < 1:
        raise ValueError("invalid HumAID device or bootstrap count")
    marker = json.loads((run / "evaluation_complete.json").read_text(encoding="utf-8"))
    if marker != {name: hash_file(run / name) for name in marker}:
        raise ValueError("full classifier official evaluation is not frozen")
    root = paths.HUMAID_ROOT / "events_set1"
    sources = sorted(root.glob("*/*.tsv"))
    if not sources:
        raise FileNotFoundError("HumAID raw event files are absent")
    exclude_path = paths.PROCESSED / "humaid_exclude.csv"
    signature = {
        "classifier": {
            name: hash_file(run / name)
            for name in (
                "checkpoint.pt",
                "config.yaml",
                "temperatures.json",
                "thresholds.json",
                "label_maps.json",
                "evaluation_complete.json",
            )
        },
        "humaid": {str(path.relative_to(root)): hash_file(path) for path in sources},
        "overlap_exclusion": hash_file(exclude_path),
        "implementation": hash_file(Path(__file__)),
        "device": device,
        "bootstrap": bootstrap,
    }
    request = output / "request.json"
    if request.exists():
        if json.loads(request.read_text(encoding="utf-8")) != signature:
            raise FileExistsError("HumAID evaluation inputs changed; choose a new output path")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("HumAID output already contains a different run")
    else:
        atomic.text(request, json.dumps(signature, indent=2) + "\n")
    complete = output / "complete.json"
    if complete.exists():
        saved = json.loads(complete.read_text(encoding="utf-8"))
        if saved != {name: hash_file(output / name) for name in saved}:
            raise ValueError("completed HumAID artifacts changed")
        return complete
    rows = evaluation_rows(read_tweets(), pd.read_csv(exclude_path, dtype=str))
    parts = []
    predictor = None
    for start in range(0, len(rows), 64):
        path = output / "batches" / f"{start:05d}.parquet"
        if not path.exists():
            if predictor is None:
                predictor = Predictor.from_run(run, device=device)
            chunk = rows.iloc[start : start + 64]
            posts = [
                Post(str(row.sample_id), "humaid", str(row.tweet_text), None, None, str(row.event), str(row.tweet_id))
                for _, row in chunk.iterrows()
            ]
            records = []
            for prediction in predictor.predict(posts):
                record = {
                    "sample_id": prediction.post_id,
                    "model_id": prediction.model_id,
                    "informative_label": prediction.informative.label,
                    "humanitarian_label": prediction.humanitarian.label,
                    "severity_label": prediction.severity.label,
                    "disaster_type_label": prediction.disaster_type.label,
                    "abstain": prediction.abstain,
                }
                for head, classes in (("informative", labels.INFORMATIVE), ("humanitarian", labels.HUMANITARIAN)):
                    record.update({f"p_{head}__{name}": getattr(prediction, head).probs[name] for name in classes})
                records.append(record)
            atomic.parquet(path, pd.DataFrame(records))
        parts.append(path)
    prediction_path = output / "predictions.parquet"
    if not prediction_path.exists():
        atomic.parquet(prediction_path, pd.concat((pd.read_parquet(part) for part in parts), ignore_index=True))
    predictions = pd.read_parquet(prediction_path)
    if predictions.severity_label.notna().any() or predictions.disaster_type_label.notna().any():
        raise ValueError("HumAID text-only predictions exposed image heads")
    metrics_path = output / "metrics.json"
    if not metrics_path.exists():
        metrics = score_humaid(rows, predictions, bootstrap=bootstrap)
        metrics["model_id"] = str(predictions.model_id.iloc[0])
        atomic.text(metrics_path, json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    saved = [prediction_path, metrics_path, *parts]
    atomic.text(complete, json.dumps({path.relative_to(output).as_posix(): hash_file(path) for path in saved}, indent=2) + "\n")
    return complete
