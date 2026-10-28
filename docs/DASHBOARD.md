# WP6 dashboard

The primary dashboard is a custom, local web interface. A four-scene field briefing opens
first; five workspaces provide investigation, event replay, verified sitrep, review queue,
and results. The workspaces use distinct page compositions and real
post-volume timelines. A small Python HTTP server supplies precomputed
artifacts and runs CPU inference only for live classification. The server binds to
`127.0.0.1`, serves only packaged local assets, and never persists uploads. The earlier
Streamlit view remains available with `sakshi-streamlit` as a functional fallback. This
presentation-layer change departs from the Streamlit stack specified in §12.1 of the
design spec, while preserving the five workspaces and their acceptance behavior.

## Run

```bash
pip install -e ".[ml,xai,app]"
HF_HUB_OFFLINE=1 sakshi-app
```

Set `HF_HOME` only if the pinned BERTweet and CLIP weights are cached outside the
default Hugging Face cache. The classifier checkpoint is under `runs/classifier/`,
separate from the pretrained encoder cache. `sakshi-app` launches the custom
fieldroom at `http://127.0.0.1:8501`.
Use `sakshi-app --port 8502` to choose another local port. The server validates actual
JPEG/PNG/WebP content, rejects files over 10 MB, and reduces the longest side to 1024
pixels. `sakshi-streamlit` launches the original Streamlit implementation.

## Interface behavior

**Present** opens the saved four-scene briefing. **Investigate** accepts new text and/or
image uploads and runs local CPU inference. Under **What moved the decision**, users can
request one prediction's word attribution and image focus map. The maps explain model
behavior; they do not verify a post. **Event replay**, **Sitrep**, **Review**, and **Results**
open the historical and research artifacts. The header returns to Present; **Esc** opens
Investigate from the briefing. Human provenance and sitrep studies, and the full local
Ollama acceptance run, remain outstanding.

## What works now

- **Field briefing:** four keyboard-accessible scenes with a responsive layout, one fixed
  live analysis, source-backed sitrep claims, and official three-seed scores. The briefing
  is the default route; the header's **Present** control returns to it.
- **Classify:** calibrated four-head CPU prediction, abstention, text-only/image-only
  behavior, modality weights, live provenance, and on-demand IG/Grad-CAM. Revealed maps
  can be compared without recomputing them. The optional event and UTC time field enables
  full temporal provenance status. Live LLM rationale remains unavailable until matching
  evidence packets and the specified Qwen/NLI models are prepared.
- **Replay:** chronological event windows, play/pause and 1–3× speed, real post-volume
  timeline and running counts, feed lenses, and recycled-post/original comparison.
  Completed captioned LOEO aggregates are selected by default for Harvey, Maria and
  Mexico; baseline previews remain available only when research artifacts are absent.
- **Sitrep:** displays the completed captioned WP5 outputs, including accepted/rejected
  sentences, citations, gate details, rates, and model identifiers. All three research
  events are available. It labels the NLI thresholds as uncalibrated and the human audit
  as pending.
- **Review:** joins trained-model abstentions with uncertain provenance statuses, hides
  dataset identifiers on screen, and appends reviewer confirm/dismiss decisions to
  ignored `runs/review/decisions.jsonl`. The seed 0 extended test prediction table is
  available; restart the app after generating it so the server reads the new artifact.
- **Results:** reads committed reports and figures without running experiments. Dev and
  demo reports are labeled as such.

Presentation mode is on by default. All displayed post text passes through the privacy
sanitizer. Images are face-blurred **before** thumbnailing, using OpenCV's bundled frontal
Haar cascade with `scaleFactor=1.1`, `minNeighbors=5`, and `minSize=(24, 24)`. Failed image
loading or blurring yields a placeholder. The cascade can miss faces, so manually inspect
any figure or video before submission.

## Verification and remaining demo inputs

The WP6 tests cover upload rejection/downscaling, handle and link removal, synthetic-face
blurring, opaque review tokens, append-only decisions, and startup on an empty artifact
root. Run:

```bash
pytest -q tests/test_app.py
ruff check src/sakshi/app tests/test_app.py
```

The custom fieldroom's briefing, five workspaces, and desktop/mobile layout were exercised
in Chrome with no browser exceptions or horizontal overflow. The live CPU classifier, image provenance,
and IG/Grad-CAM flows were also exercised through the interface. On the current laptop,
warmed base prediction and provenance fit the three-second target; explanations run on
demand and have not been shown to fit that target together with classification. The
captioned LOEO aggregates and 53 verified sitrep windows are now wired into the default
views, with a small adapter test and real-artifact smoke check. A final recorded
five-minute presentation and independent human study remain outstanding. See
[GENERATION.md](GENERATION.md) for the evaluation limits.
