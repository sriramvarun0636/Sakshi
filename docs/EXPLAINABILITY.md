# Explanations and evidence packets

WP3 uses the trained classifier in `runs/classifier/full-s0-20261024-062724/`.
The code supports all four main heads, gated/missing modalities, packet production,
faithfulness and cascading randomisation. Official explanation research runs remain
pending; the current reports with `_dev` in their names are small integration checks.

## Install and run a small check

```sh
python -m pip install -e '.[dev,ml,xai]'
sakshi-explain packets --run runs/classifier/full-s0-20261024-062724 \
  --rows dev --limit 3 --device cpu --output runs/explain/my-check/packets.jsonl
sakshi-explain faithfulness --run runs/classifier/full-s0-20261024-062724 \
  --split dev --count 1 --device cpu --output runs/explain/my-check/faithfulness
sakshi-explain sanity --run runs/classifier/full-s0-20261024-062724 \
  --split dev --count 1 --device cpu --output runs/explain/my-check/sanity
```

Use fresh packet output paths; existing packets are preserved. `--rows` accepts
`train`, `dev`, `test`, `official:<head>`, or a Parquet file with unique `sample_id`
values. Selector files supply IDs only; content is joined from the canonical manifest.
Without `--output`, packets go to `data/processed/packets/<model_id>.jsonl`.

Standalone commands compute one modality:

```sh
sakshi-explain ig --run runs/classifier/full-s0-20261024-062724 \
  --rows dev --head informative --limit 1 --device cpu --output runs/explain/ig.jsonl
sakshi-explain gradcam --run runs/classifier/full-s0-20261024-062724 \
  --rows dev --head severity --limit 1 --device cpu --overlays --output runs/explain/cam.jsonl
```

The existing local pretrained weights are cached under `/private/tmp/sakshi_hf_cache`.
To reuse them offline, set `HF_HOME=/private/tmp/sakshi_hf_cache` and
`HF_HUB_OFFLINE=1` before starting Python. Both the classifier checkpoint and the
pinned pretrained tokenizer/encoder files are needed by the current loader.

## Methods and numerical checks

- Captum `LayerIntegratedGradients` targets the uncalibrated predicted-class logit at
  BERTweet's word embeddings. Image features, attention mask and **position IDs** stay
  fixed. The baseline pads non-special word IDs. RoBERTa otherwise derives different
  position IDs from the pad baseline, breaking the word-layer completeness comparison.
- Use 50 integration steps, internal batch 16; retry at 150 when absolute delta exceeds
  5% of the input/baseline logit difference. Keep and flag a still-unconverged result.
  BPEs ending in `@@` are merged; unknown tokens and truncated final words retain their
  whitespace-word alignment. Display/model word-count mismatches are logged.
- Grad-CAM uses the last CLIP block's `layer_norm1`, dropping CLS only when reshaping
  the CAM to a 14×14 grid. The live frozen tower remains differentiable with respect
  to pixels; its output passes through post-layernorm and 2×2 pooling to 50×768.
- Preprocessing explicitly uses **PIL**, matching the existing cache. Installing
  torchvision must not change it. See the
  [Transformers backend documentation](https://huggingface.co/docs/transformers/en/main_classes/image_processor).
  Live/cached features are checked with maximum absolute tolerance `1e-2`; disagreement
  raises before an explanation can be published. Existing caches are never rebuilt here.
- Region boxes map from the centre crop to the EXIF-corrected original image. Raw
  heatmaps are `.npy` files. Optional overlays always run face detection/blurring first,
  then use alpha 0.4. Missing detectors refuse export. OpenCV is constrained to 4.x:
  [OpenCV 5 moved Haar classifiers out of the base package](https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration).
- Gated/unavailable heads are skipped. In particular, the packet rule requests
  humanitarian image regions for a noninformative post, but that head is gated, so its
  image evidence is empty. No substitute explanation is invented.

## Packets, diagnostics and captions

Every packet is schema-validated and privacy-scanned before atomic publication.
Validation also checks canonical labels, probabilities, modality weights, hierarchy,
region boxes, evidence IDs and same-model explanation identity. Predictions use calibrated
probabilities rounded to two decimals. Diagnostic flags, deltas, signed words and heatmap
paths stay in a separate `<output_stem>_artifacts/explanations.jsonl`.

The artifact folder includes the row selection, checkpoint/config/calibration hashes,
source/runtime hashes, predictions and a completion marker. Its prediction table and
packets can be checked with WP4's `load_packets` and `validate_selected`.

Existing §9.4 captions are read from `data/processed/captions.parquet` and marked
`model_generated: true`. Missing captions stay null. The Qwen caption generation job is
still pending and is not replaced with inferred descriptions. Region captions remain
optional. Data, words, packets, heatmaps and source-image overlays stay under ignored
`data/` or `runs/`; only tables and non-image audit curves are suitable for Git.

## Audits and research freeze

Faithfulness masks whitespace words or normalized 16×16 image patches, holding the other
modality fixed. It measures comprehensiveness/sufficiency at 1, 5, 10, 20 and 50%, with
five random rankings (seeds 0–4), AOPC and post-level bootstrap intervals. Randomisation
sanity checks cascade through the head, last cross-attention block, and gate, holding the
original target fixed. Weights and CPU/CUDA/MPS RNG states are restored even on failure.
Constant or length-one attribution vectors have undefined Spearman correlation; these
are counted and reported, not converted to zero.

Audits resume from atomic per-post records. Changed requests or sampling prediction
hashes are rejected. Identical completed requests verify the final output hashes and
reuse the reports without loading a model. Progress and gated/unavailable counts are logged.

Test-row commands require an already completed, frozen classifier evaluation. They
verify/reuse that evaluation, then freeze `explanation_protocol.json` before attribution.
The fixed protocol includes source, package, model, calibration and data identities.
Changed protocols are rejected. The CLI does not launch official evaluation implicitly.

For the formal GPU runs, use `--split test --count 500` for faithfulness and
`--split test --count 100` for sanity, with all four heads. Sampling uses each head's
official selector and seed 0, stratified by predicted class; gated/unavailable sampled
rows stay in the accounting and have no attribution. Dev sampling uses labelled unified
dev rows. Formal runs publish `reports/results/faithfulness.md`, `sanity.md` and
`reports/figures/faithfulness_curves.png`; dev runs use `_dev` names. A dev check with
one post per head cannot establish explainer quality or meaningful uncertainty.

## Tests

The default suite uses small offline encoders and tests actual Captum/Grad-CAM gradients,
alignment, masks, coordinate mapping, privacy, protocol freeze and restoration.
The private-data equivalence test is opt-in:

```sh
HF_HOME=/private/tmp/sakshi_hf_cache HF_HUB_OFFLINE=1 SAKSHI_RUN_DATA_TESTS=1 \
  pytest -q tests/test_explain_data.py
```

This checks five real dev images against the existing frozen CLIP cache. It neither
trains a model nor predicts/scores official tests.
