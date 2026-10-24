"""WP5 verifier, caching, regeneration and artifact-bound workflow contracts."""

import json
from dataclasses import replace

import pandas as pd
import pytest
from PIL import Image

from sakshi import paths
from sakshi.explain.captions import CaptionCache
from sakshi.generate.audit import export_audit, score_audit
from sakshi.generate.captions import generate_captions
from sakshi.generate.lexicons import CASUALTY, NO_DAMAGE, SEVERE, terms
from sakshi.generate.llm import CachedLLM, GenerationFailure, OllamaClient, cache_key
from sakshi.generate.numbers import extract, unsupported
from sakshi.generate.prompts import SITREP_SCHEMA
from sakshi.generate.rationale import explain_post
from sakshi.generate.sitrep import generate_sitrep
from sakshi.generate.verify import Verifier, split_sentences
from sakshi.generate.workflow import GenerateConfig, _hash, _selected, build_evaluation_plan, run_sitreps


class Scorer:
    model_name = "fake-nli"

    def score(self, premise, hypothesis):
        return (0.1, 0.8) if "unsupported" in hypothesis else (0.9, 0.01)

    def token_count(self, premise):
        return len(premise.split())


class Raw:
    backend, model = "fake", "fixed"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, system, user, schema, *, max_tokens, temperature, seed):
        self.calls += 1
        return next(self.responses)


class Scripted:
    def __init__(self, values):
        self.values = iter(values)
        self.prompts = []

    def generate_json(self, system, user, schema, **kwargs):
        self.prompts.append(user)
        return next(self.values)


def packet(*, severity="mild_damage", status="novel", text="The bridge was damaged.", post_id="sample-a"):
    return {
        "schema_version": "1.0",
        "post_id": post_id,
        "event": "hurricane_harvey",
        "posted_at": "2017-08-27T10:00:00Z",
        "has_text": True,
        "has_image": True,
        "text_clean": text,
        "image_caption": None,
        "evidence": [],
        "provenance": {"status": status, "origin_event": None, "origin_first_seen": None, "similarity": None},
        "modality_weights": {"text": 0.5, "image": 0.5},
        "abstain": False,
        "abstain_reason": None,
        "model_id": "full-s0-loeo_hurricane_harvey",
        "predictions": {
            "informative": {"label": "informative", "p": 0.9, "status": "predicted"},
            "humanitarian": {"label": "infrastructure_and_utility_damage", "p": 0.8, "status": "predicted"},
            "severity": {"label": severity, "p": 0.8, "status": "predicted"},
            "disaster_type": {"label": "hurricane", "p": 0.8, "status": "predicted"},
        },
    }


@pytest.mark.parametrize(
    ("sentence", "exact", "vague"),
    [
        ("Over 1,200 homes were damaged", {1200}, set()),
        ("About 1.2k homes flooded", {1200}, set()),
        ("twenty-five people were rescued", {25}, set()),
        ("25–30 roads are closed", {25, 30}, set()),
        ("Power is out for 60% of the island", {60}, set()),
        ("The 3rd shelter opened", {3}, set()),
        ("Flooding reported on Sept 20 at 5pm", {20, 5}, set()),
        ("Hundreds of homes are flooded", set(), {"hundreds"}),
        ("Roads are closed [P12]", set(), set()),
        ("Harvey made landfall as a Category 4 storm in 2017", {4, 2017}, set()),
        ("I-45 is closed near Houston", {45}, set()),
        ("two hundred and five families evacuated", {205}, set()),
        ("No figures were given", set(), set()),
    ],
)
def test_number_normalizer_spec_table(sentence, exact, vague):
    result = extract(sentence)
    assert result.exact == exact and result.vague == vague


def test_vague_bounds_and_number_support_are_citation_scoped():
    assert unsupported("Hundreds of homes flooded", "120 homes flooded") == []
    assert unsupported("Hundreds of homes flooded", "99 homes flooded") == ["hundreds"]
    assert unsupported("25–30 roads closed", "25 roads closed") == ["30"]
    assert extract("One, two and twenty-five homes").exact == {1, 2, 25}


def test_lexicons_negation_and_splitter():
    assert terms("No buildings were destroyed", SEVERE) == [("destroyed", True)]
    assert terms("Several people were injured", CASUALTY) == [("injured", False)]
    assert terms("The bridge had no damage", NO_DAMAGE) == [("no damage", False)]
    assert split_sentences("Dr. Lee arrived. Roads were closed. U.S. teams helped!") == [
        "Dr. Lee arrived.",
        "Roads were closed.",
        "U.S. teams helped!",
    ]
    for abbreviation in ("U.S.", "U.K.", "St.", "Dr.", "Mr.", "Mrs.", "No.", "Gov.", "Sept.", "Oct.", "Nov."):
        assert split_sentences(f"{abbreviation} Smith reported damage. Roads closed.") == [
            f"{abbreviation} Smith reported damage.",
            "Roads closed.",
        ]


def test_each_gate_runs_and_retains_multiple_failure_reasons():
    verifier = Verifier(Scorer())
    posts = {"P1": packet(status="cross_event_concurrent")}
    passed = verifier.verify("The bridge was damaged.", ["P1"], posts)
    assert all(item["passed"] for item in passed.values())
    assert passed["provenance"]["detail"] == "unverified_image_origin"
    failed = verifier.verify("Ten people died in a destroyed bridge; unsupported.", ["P1", "P9"], posts)
    assert "unknown_id:P9" in failed["citation"]["detail"]
    assert not failed["entailment"]["passed"]
    assert "severe" in failed["label"]["detail"] and "casualty" in failed["label"]["detail"]
    assert "10" in failed["number"]["detail"]
    assert not verifier.verify("No damage was reported.", ["P1"], {"P1": packet(severity="severe_damage")})["label"]["passed"]
    assert not verifier.verify("The bridge was damaged.", [], posts)["citation"]["passed"]
    assert not verifier.verify("The bridge was damaged.", ["P1"] * 6 + ["P2", "P3", "P4", "P5", "P6"], posts)["citation"][
        "passed"
    ]
    recycled = {"P1": packet(status="recycled")}
    assert not verifier.verify("The bridge was damaged.", ["P1"], recycled)["provenance"]["passed"]
    malformed = verifier.verify("The bridge was damaged.", ["post-one"], {"post-one": packet()})
    assert "unknown_id:post-one" in malformed["citation"]["detail"]


def test_long_premise_checks_each_cited_post_and_pairs_contradiction_with_best_entailment():
    class Long(Scorer):
        def score(self, premise, hypothesis):
            return (0.8, 0.1) if premise.startswith("Post P2") else (0.3, 0.4)

        def token_count(self, premise):
            return 500

    result = Verifier(Long()).verify("The bridge was damaged.", ["P1", "P2"], {"P1": packet(), "P2": packet(post_id="sample-b")})
    assert result["entailment"]["passed"] and result["entailment"]["p_contra"] == 0.1


def test_cache_key_validation_retry_and_failure_preserve_raw(tmp_path):
    with pytest.raises(ValueError, match="local Ollama"):
        OllamaClient(url="https://example.com/api/chat")
    key = cache_key("fake", "fixed", "s", "u", SITREP_SCHEMA, max_tokens=10, temperature=0, seed=42)
    assert key != cache_key("fake", "other", "s", "u", SITREP_SCHEMA, max_tokens=10, temperature=0, seed=42)
    raw = Raw(
        [
            "garbage",
            '{"sentences":[{"text":"A bridge was damaged.","citations":[]]}',
            '{"sentences":[{"text":"A bridge was damaged.","citations":["P1"]}]}',
        ]
    )
    client = CachedLLM(raw, tmp_path)
    result = client.generate_json("s", "u", SITREP_SCHEMA, max_tokens=10)
    assert len(result["sentences"]) == 1 and raw.calls == 3
    assert client.generate_json("s", "u", SITREP_SCHEMA, max_tokens=10) == result and raw.calls == 3
    assert json.loads((tmp_path / f"{key}.json").read_text())["raw"] == "garbage"
    broken = CachedLLM(Raw(["bad"] * 3), tmp_path / "broken")
    with pytest.raises(GenerationFailure):
        broken.generate_json("s", "u", SITREP_SCHEMA)
    assert len(list((tmp_path / "broken").glob("*.json"))) == 3


def aggregate():
    return {
        "model_id": "full-s0-loeo_hurricane_harvey",
        "event": "hurricane_harvey",
        "input_scope": "loeo",
        "window": {"start": "2017-08-27T00:00:00Z", "end": "2017-08-28T00:00:00Z", "hours": 24},
        "counts": {"by_category": {"infrastructure_and_utility_damage": 12}},
        "selected": [{"pid": "P1", "sample_ids": ["sample-a"], "category": "infrastructure_and_utility_damage"}],
        "system_lines": ["One informative post in this window."],
    }


def test_regeneration_keeps_original_rejection_and_accepted_rewrite():
    llm = Scripted(
        [
            {"sentences": [{"text": "The bridge was destroyed. The bridge was damaged.", "citations": ["P1", "P1"]}]},
            {
                "rewrites": [
                    {"replaces": "S1", "text": "The bridge was damaged.", "citations": ["P1"]},
                    {"replaces": "S99", "text": "Other claim.", "citations": ["P1"]},
                ]
            },
        ]
    )
    result = generate_sitrep(aggregate(), {"P1": packet()}, llm, Verifier(Scorer()), llm_name="fake:fixed")
    assert result["verification_rate"] == {"initial": 0.5, "final": 1.0}
    assert [item["accepted"] for item in result["sentences"]] == [True, True, False]
    assert result["ignored_rewrites"] == ["S99"]
    assert result["sentences"][-1]["text"] == "The bridge was destroyed."
    assert "sample-a" not in llm.prompts[0]


def test_generation_failure_and_empty_window_never_claim_success():
    class Failed:
        def generate_json(self, *args, **kwargs):
            raise GenerationFailure("invalid", ())

    result = generate_sitrep(aggregate(), {"P1": packet()}, Failed(), Verifier(Scorer()), llm_name="fake")
    assert result["generation_failures"] == 1 and result["sentences"] == []
    assert "failed" in result["system_lines"][-1]
    empty = aggregate() | {"selected": []}
    result = generate_sitrep(empty, {}, Failed(), Verifier(Scorer()), llm_name="fake")
    assert result["generation_failures"] == 0 and result["sentences"] == []


def test_multi_image_tweet_uses_rollup_severity_and_cautious_origin():
    item = aggregate()
    item["selected"][0]["sample_ids"] = ["sample-a", "sample-b"]
    packets = {"sample-a": packet(), "sample-b": packet(post_id="sample-b", severity="severe_damage", status="match_undated")}
    tweets = pd.DataFrame(
        [
            {
                "sample_ids": ["sample-a", "sample-b"],
                "representative_id": "sample-a",
                "informative_label": "informative",
                "humanitarian_label": "infrastructure_and_utility_damage",
                "event": "hurricane_harvey",
                "posted_at": pd.Timestamp("2017-08-27T10:00:00Z"),
                "abstain": False,
                "excluded": False,
                "severity_label": "severe_damage",
                "provenance_status": "match_undated",
            }
        ]
    )
    selected, severity, origin = _selected(item, tweets, packets)
    llm = Scripted([{"sentences": [{"text": "The bridge was destroyed.", "citations": ["P1"]}]}])
    report = generate_sitrep(
        item, selected, llm, Verifier(Scorer()), llm_name="fake", severity_by_pid=severity, origin_status_by_pid=origin
    )
    assert 'severity="severe_damage"' in llm.prompts[0]
    assert 'image_origin="unverified"' in llm.prompts[0]
    assert report["sentences"][0]["gates"]["label"]["passed"]


def test_workflow_binds_completed_aggregate_and_packets_then_reuses_output(tmp_path):
    source = tmp_path / "aggregate"
    source.mkdir()
    name = "20170827T000000Z.json"
    (source / name).write_text(json.dumps(aggregate()))
    (source / "config.json").write_text(json.dumps({"mode": "loeo", "classifier_run": "fixture"}))
    pd.DataFrame(
        [
            {
                "sample_ids": ["sample-a"],
                "representative_id": "sample-a",
                "informative_label": "informative",
                "humanitarian_label": "infrastructure_and_utility_damage",
                "event": "hurricane_harvey",
                "posted_at": pd.Timestamp("2017-08-27T10:00:00Z"),
                "abstain": False,
                "excluded": False,
                "severity_label": "mild_damage",
            }
        ]
    ).to_parquet(source / "tweets.parquet")
    artifacts = {path.name: _hash(path) for path in source.iterdir() if path.is_file()}
    (source / "complete.json").write_text(json.dumps({"evidence_ready": True, "windows": 1, "artifacts": artifacts}))
    packets = tmp_path / "packets.jsonl"
    packets.write_text(json.dumps(packet()) + "\n")
    (source / "inputs.json").write_text(json.dumps({"packets": {"path": str(packets), "sha256": _hash(packets)}}))
    artifacts = {path.name: _hash(path) for path in source.iterdir() if path.is_file() and path.name != "complete.json"}
    (source / "complete.json").write_text(json.dumps({"evidence_ready": True, "windows": 1, "artifacts": artifacts}))
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "max_windows": 60,
                "min_eligible": 10,
                "selected": [
                    {
                        "model_id": aggregate()["model_id"],
                        "event": "hurricane_harvey",
                        "window": name,
                        "aggregate_sha256": _hash(source / name),
                    }
                ],
            }
        )
    )
    config = GenerateConfig(
        aggregate_dir=str(source),
        packets=str(packets),
        output_root=str(tmp_path / "sitreps"),
        report_path=str(tmp_path / "summary.md"),
        evaluation_plan=str(plan),
    )
    llm = Scripted([{"sentences": [{"text": "The bridge was damaged.", "citations": ["P1"]}]}])
    marker = run_sitreps(config, llm=llm, scorer=Scorer())
    assert marker.is_file() and "1.000" in (tmp_path / "summary.md").read_text()
    assert len(llm.prompts) == 1 and "sample-a" not in llm.prompts[0]
    assert run_sitreps(config, llm=llm, scorer=Scorer()) == marker and len(llm.prompts) == 1
    sheet = tmp_path / "calibration.csv"
    viewer = tmp_path / "calibration.html"
    export_audit(tmp_path / "sitreps", sheet, viewer, phase="calibration", calibration_count=1, include_images=False)
    assert "image_path" not in sheet.read_text() and "The bridge was damaged." in viewer.read_text()
    assert (tmp_path / "calibration_sampling.json").is_file()
    frame = pd.read_csv(sheet, keep_default_na=False)
    frame[["label_a", "label_b", "label_final"]] = "supported"
    frame.to_csv(sheet, index=False)
    score_audit(sheet, tmp_path / "calibration_sampling.json", tmp_path / "calibration.json", phase="calibration")
    frame.loc[0, "sentence"] = "Tampered claim"
    frame.to_csv(sheet, index=False)
    with pytest.raises(ValueError, match="changed after export"):
        score_audit(sheet, tmp_path / "calibration_sampling.json", tmp_path / "calibration.json", phase="calibration")
    with pytest.raises(FileExistsError):
        run_sitreps(replace(config, nli=replace(config.nli, tau_entail=0.6)), llm=llm, scorer=Scorer())
    packets.write_text(json.dumps(packet(text="Changed evidence text.")) + "\n")
    with pytest.raises(ValueError, match="differ from the file validated"):
        run_sitreps(config, llm=llm, scorer=Scorer())


def test_research_window_plan_filters_low_evidence_and_balances_events(tmp_path):
    root = tmp_path / "aggregates"
    events = ("hurricane_harvey", "hurricane_maria", "mexico_earthquake")
    for event in events:
        directory = root / f"full-s0-loeo_{event}" / event
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps({"mode": "loeo", "event": event, "model_id": f"full-s0-loeo_{event}"}))
        artifacts = {"config.json": _hash(directory / "config.json")}
        for index, eligible in enumerate((12, 2)):
            name = f"2017082{index}T000000Z.json"
            (directory / name).write_text(
                json.dumps(
                    {
                        "model_id": f"full-s0-loeo_{event}",
                        "event": event,
                        "input_scope": "loeo",
                        "window": {"hours": 24},
                        "counts": {"by_category": {"a": eligible}},
                    }
                )
            )
            artifacts[name] = _hash(directory / name)
        (directory / "complete.json").write_text(json.dumps({"evidence_ready": True, "artifacts": artifacts}))
    output = tmp_path / "plan.json"
    build_evaluation_plan(root, output, max_windows=3)
    selected = json.loads(output.read_text())["selected"]
    assert len(selected) == 3 and {item["event"] for item in selected} == set(events)
    assert all(item["eligible"] == 12 for item in selected)
    assert build_evaluation_plan(root, output, max_windows=3) == output


def test_audit_scoring_requires_independent_labels_and_calibrates_on_harvey(tmp_path):
    sheet = tmp_path / "sheet.csv"
    pd.DataFrame(
        [
            {
                "item_id": "G0001",
                "phase": "calibration",
                "label_a": "supported",
                "label_b": "supported",
                "label_final": "supported",
            },
            {
                "item_id": "G0002",
                "phase": "calibration",
                "label_a": "unsupported",
                "label_b": "unsupported",
                "label_final": "unsupported",
            },
        ]
    ).to_csv(sheet, index=False)
    metadata = tmp_path / "sampling.json"
    metadata.write_text(
        json.dumps(
            {
                "items": [
                    {"item_id": "G0001", "phase": "calibration", "gates": {"entailment": {"score": 0.8, "p_contra": 0.1}}},
                    {"item_id": "G0002", "phase": "calibration", "gates": {"entailment": {"score": 0.2, "p_contra": 0.6}}},
                ]
            }
        )
    )
    output = tmp_path / "calibration.json"
    score_audit(sheet, metadata, output, phase="calibration")
    result = json.loads(output.read_text())
    assert result["kappa"] == 1 and result["chosen"]["f1"] == 1
    frame = pd.read_csv(sheet, keep_default_na=False)
    frame.loc[0, "label_b"] = ""
    frame.to_csv(sheet, index=False)
    with pytest.raises(ValueError, match="independent"):
        score_audit(sheet, metadata, output, phase="calibration")


def test_evaluation_audit_uses_stratum_weights_and_checks_immutable_text(tmp_path):
    sheet = tmp_path / "evaluation.csv"
    rows = [
        {
            "item_id": "G0001",
            "phase": "evaluation",
            "sentence": "Claim A",
            "label_a": "unsupported",
            "label_b": "unsupported",
            "label_final": "unsupported",
        },
        {
            "item_id": "G0002",
            "phase": "evaluation",
            "sentence": "Claim B",
            "label_a": "unsupported",
            "label_b": "unsupported",
            "label_final": "unsupported",
        },
    ]
    pd.DataFrame(rows).to_csv(sheet, index=False)
    metadata = tmp_path / "evaluation_sampling.json"
    metadata.write_text(
        json.dumps(
            {
                "pools": {"accepted": 100, "rejected": 20},
                "items": [
                    {
                        "item_id": "G0001",
                        "phase": "evaluation",
                        "accepted": True,
                        "gates": {
                            "citation": {"passed": True},
                            "provenance": {"passed": True},
                            "entailment": {"passed": True},
                            "label": {"passed": True},
                            "number": {"passed": True},
                        },
                    },
                    {
                        "item_id": "G0002",
                        "phase": "evaluation",
                        "accepted": False,
                        "gates": {
                            "citation": {"passed": True},
                            "provenance": {"passed": True},
                            "entailment": {"passed": False},
                            "label": {"passed": True},
                            "number": {"passed": True},
                        },
                    },
                ],
            }
        )
    )
    output = tmp_path / "evaluation.json"
    score_audit(sheet, metadata, output, phase="evaluation")
    result = json.loads(output.read_text())
    assert result["estimated_precision"] == 0
    assert result["estimated_unsupported_recall"] == pytest.approx(1 / 6)
    assert result["per_gate_unsupported"]["entailment"] == 1


def test_caption_cache_is_append_only_and_per_post_rationale_keeps_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CRISISMMD_ROOT", tmp_path)
    Image.new("RGB", (32, 32), (20, 40, 60)).save(tmp_path / "sample.png")
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame([{"sample_id": "sample-a", "dataset": "crisismmd", "image_path": "sample.png"}]).to_parquet(manifest)

    class Caption:
        model_name = "fixture-vlm"
        calls = 0

        def caption(self, image, prompt):
            self.calls += 1
            assert image.size == (32, 32) and "Do not guess" in prompt
            return "A wet road is visible."

    captioner = Caption()
    cache = tmp_path / "captions.parquet"
    generate_captions(["sample-a"], cache, captioner=captioner, manifest=manifest)
    generate_captions(["sample-a"], cache, captioner=captioner, manifest=manifest)
    assert captioner.calls == 1 and pd.read_parquet(cache).caption.tolist() == ["A wet road is visible."]
    assert CaptionCache(cache).get("sample-a")["model_generated"]

    good = Scripted([{"rationale": "The post says the bridge was damaged."}])
    result = explain_post(packet(), good, Scorer())
    assert result["rationale"] and result["prefix"].startswith("Classified as") and len(result["packet_sha256"]) == 64
    bad = Scripted([{"rationale": "Twenty bridges were damaged."}])
    result = explain_post(packet(), bad, Scorer())
    assert result["rationale"] is None and result["rejected_rationale"][0]["unsupported_numbers"] == ["20"]
