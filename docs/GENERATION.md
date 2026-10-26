# WP5 verified generation

The WP5 code consumes completed WP4 aggregates and matching WP3 evidence packets. It
never reads raw tweet metadata into an LLM prompt. The aggregation run must have
`evidence_ready: true`; research generation additionally requires a LOEO classifier
run and the frozen three-event evaluation plan. Outputs under `runs/` are private.

## Implementation and current limits

`sakshi-generate` provides the Ollama and CUDA Hugging Face backends, immutable raw/parsed
LLM caching, schema validation with two retries, sentence splitting, five verifier gates,
one rewrite round, per-window sitreps, computed summaries, caption generation, optional
per-post rationales, and the blind audit workflow. Each research sitrep uses only the
selected post packets, and every generated claim retains all gate results. Rejected
claims remain in the JSON. Multi-image tweets use the WP4 rollup severity and the most
cautious provenance flag. Empty windows make no LLM call.

The default LLM is `qwen2.5:7b-instruct` through Ollama. The HF backend uses 4-bit
loading on CUDA cards below 24 GiB and bf16 on larger cards. The NLI model is
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`; entailment and contradiction
indices come from its `id2label`. The default thresholds are `0.5` and `0.3` until the
Harvey calibration study chooses an entailment threshold. A Qwen tokenizer checks
the prompt budget before a real backend call; 200 tokens are reserved for retry text.

The Harvey, Maria and Mexico LOEO classifiers and their held-out evaluations are frozen.
Their real evidence packets are complete and validated: 504, 666 and 279 posts,
respectively. All three WP4 research aggregates and the balanced 53-window evaluation
plan are frozen. A laptop Ollama run produced five Harvey windows before being stopped
for heat. Separate, private Kaggle GPU jobs now generate the research sitreps using
pinned Qwen2.5-7B-Instruct and DeBERTa NLI snapshots; their outputs are under
`runs/sitreps_hf/`. The first Harvey job is running, and Maria and Mexico are queued.
The fake-backend integration test verifies file and logic flow; it is not a
model-quality result. Two independent reviewers and an adjudicator are unavailable,
so human calibration and evaluation scores remain pending and must not be replaced
with model-generated labels.

## Prepare evidence and models

Install the dependencies needed by the chosen backend:

```bash
pip install -e ".[dev,ml,xai,aggregate,llm]"
ollama pull qwen2.5:7b-instruct
```

The three LOEO classifier folds, WP2 predictions, same-model packets and WP4 aggregates
have already been produced. The commands below document the reproducible packet flow
for a new event or a fresh checkout:

```bash
sakshi-aggregate --model-id full-s0-loeo_hurricane_harvey --event hurricane_harvey \
  --mode loeo --run runs/classifier/full-s0-loeo_hurricane_harvey-20261025-063127 \
  --predictions runs/classifier/full-s0-loeo_hurricane_harvey-20261025-063127/predictions_loeo.parquet \
  --plan-packets runs/aggregate/plans/full-s0-loeo_hurricane_harvey.parquet
sakshi-explain packets --run runs/classifier/full-s0-loeo_hurricane_harvey-20261025-063127 \
  --rows runs/aggregate/plans/full-s0-loeo_hurricane_harvey.parquet --device mps \
  --output data/processed/packets/full-s0-loeo_hurricane_harvey.jsonl
```

Packet generation writes one atomic post checkpoint at a time and can resume with the
same command after an interruption. The plan freezes source hashes and exact sample IDs.
Caption input is a JSON array of
sample IDs. `sakshi-generate captions --ids selected_ids.json` appends only missing
captions to `data/processed/captions.parquet`; packets must be built after captions exist
to include them. Unavailable captions remain null, and the verifier uses only packet text
and caption fields.

Run WP4 in LOEO mode for each event with `--run`, `--predictions` and `--packets`
pointing to the same classifier and planned packet file. Its completed aggregate directory is
`runs/aggregate/<model_id>/<event>/`, with `evidence_ready: true`. Then freeze the study
windows, using the default seed 0, at least 10 eligible tweets, 24-hour windows and a
60-window cap balanced across the three events:

```bash
sakshi-generate plan --aggregate-root runs/aggregate
```

The plan is written to `runs/generation/evaluation_plan.json`. An existing different
plan is never overwritten. The runner checks its window hashes and eligibility.

## Generate and calibrate

First generate Harvey with the default thresholds in
`configs/generation/sitrep.yaml`:

```bash
sakshi-generate sitreps --config configs/generation/sitrep.yaml \
  --aggregate-dir runs/aggregate/full-s0-loeo_hurricane_harvey/hurricane_harvey \
  --packets data/processed/packets/full-s0-loeo_hurricane_harvey.jsonl
sakshi-generate audit-export --phase calibration
```

Two reviewers independently fill `label_a` and `label_b` in
`reports/generation/calibration_sheet.csv`; a third reviewer fills `label_final` after
adjudication. Use the face-blurred local viewer at
`runs/generation/calibration_viewer.html` to inspect each cited image. Labels are
`supported`, `partially_supported`, or `unsupported`; judge only against cited posts
and images. Then run:

```bash
sakshi-generate audit-score --phase calibration
```

The chosen `tau_entail` is in `reports/generation/audit_calibration.json`. With no
reviewers available, the ongoing Kaggle runs use the design spec's explicit uncalibrated
defaults, `tau_entail: 0.5` and `tau_contra: 0.3`. They must be labelled as
uncalibrated outputs, and the human-study acceptance claim remains open.

```bash
sakshi-generate sitreps --config configs/generation/sitrep-calibrated.yaml \
  --aggregate-dir runs/aggregate/full-s0-loeo_hurricane_maria/hurricane_maria \
  --packets data/processed/packets/full-s0-loeo_hurricane_maria.jsonl
sakshi-generate sitreps --config configs/generation/sitrep-calibrated.yaml \
  --aggregate-dir runs/aggregate/full-s0-loeo_mexico_earthquake/mexico_earthquake \
  --packets data/processed/packets/full-s0-loeo_mexico_earthquake.jsonl
sakshi-generate audit-export --phase evaluation
```

The evaluation export refuses to sample sitreps whose NLI thresholds differ from the
Harvey-calibrated values. Two reviewers and an adjudicator fill
`reports/generation/audit_sheet.csv`, blind to gate outcomes. Then run
`sakshi-generate audit-score --phase evaluation`. The score output includes Cohen's
κ, estimated accepted-sentence precision, weighted recall of unsupported claims and
per-gate error counts. `reports/generation/summary.md` reports VR initial/final, gate
failure rates, evidence coverage, sentence/citation counts, latency and generation
failures by window, event and overall. Demo and research results have separate sections.

## Optional post inspection

`sakshi-generate explanations --packet <one-packet.json> --output
runs/generation/rationale.json` produces a deterministic label prefix plus an optional
LLM rationale. The rationale must pass entailment and number checks; otherwise the
rejected text is stored and the displayed rationale is null.

## Interpretation

The verifier checks faithfulness to cited posts, not truth. Posts can be wrong; provenance
and evidence grouping address only some failure modes. NLI models make errors, captions
can hallucinate, lexicons have limited coverage, and the pipeline is English-only.
The human audit is required before reporting research-quality verification results.
