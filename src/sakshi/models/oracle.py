"""Gold-label predictions for integration fixtures before the trained model exists."""

from __future__ import annotations

import pandas as pd

from sakshi.data import labels
from sakshi.types import HeadOutput, Prediction


def _gold(value: object, classes: list[str]) -> HeadOutput:
    if pd.isna(value):
        return HeadOutput(None, {}, "unavailable")
    if value not in classes:
        raise ValueError(f"oracle label {value!r} is outside canonical classes")
    return HeadOutput(str(value), {label: float(label == value) for label in classes}, "predicted")


class OraclePredictor:
    """Produce contract-shaped predictions from manifest gold labels without a model."""

    model_id = "oracle"

    def predict_row(self, row: pd.Series) -> Prediction:
        if pd.isna(row.get("sample_id")):
            raise ValueError("oracle row needs a sample_id")
        text = row.get("text")
        has_text = isinstance(text, str) and bool(text.strip())
        has_image = pd.notna(row.get("image_path")) and pd.isna(row.get("error"))
        if not has_text and not has_image:
            raise ValueError("oracle row has neither usable text nor image")

        informative = _gold(row.get("y_informative"), labels.INFORMATIVE)
        humanitarian = _gold(row.get("y_humanitarian"), labels.HUMANITARIAN)
        severity = _gold(row.get("y_severity"), labels.SEVERITY) if has_image else HeadOutput(None, {}, "unavailable")
        disaster_type = (
            _gold(row.get("y_disaster_type"), labels.DISASTER_TYPE) if has_image else HeadOutput(None, {}, "unavailable")
        )
        if informative.label == "not_informative":
            humanitarian = HeadOutput(
                "not_humanitarian", {v: float(v == "not_humanitarian") for v in labels.HUMANITARIAN}, "gated"
            )
            severity = HeadOutput(None, {}, "gated") if has_image else severity
        elif humanitarian.label == "not_humanitarian":
            severity = HeadOutput(None, {}, "gated") if has_image else severity

        if has_text and has_image:
            weights = {"text": 0.5, "image": 0.5}
        elif has_text:
            weights = {"text": 1.0, "image": 0.0}
        else:
            weights = {"text": 0.0, "image": 1.0}
        return Prediction(
            post_id=str(row.sample_id),
            informative=informative,
            humanitarian=humanitarian,
            severity=severity,
            disaster_type=disaster_type,
            modality_weights=weights,
            abstain=False,
            abstain_reason=None,
            model_id=self.model_id,
        )

    def predict_rows(self, rows: pd.DataFrame) -> list[Prediction]:
        """Return one prediction per unique sample ID, preserving input order."""
        if rows.sample_id.duplicated().any():
            raise ValueError("oracle input has duplicate sample_id values")
        return [self.predict_row(row) for _, row in rows.iterrows()]
