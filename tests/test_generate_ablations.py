"""The caption ablation changes G3 on saved sentences without changing other gates."""

from sakshi.generate.ablations import evaluate_window, summarize


class TextOnlyScorer:
    def verify(self, sentence, citations, packets):
        assert sentence == "Floodwater covers the road."
        assert citations == ["P1"]
        assert packets["P1"]["image_caption"] is None
        return {"entailment": {"passed": False}}


def test_text_only_g3_keeps_other_saved_gates_fixed():
    gates = {name: {"passed": True} for name in ("citation", "provenance", "entailment", "label", "number")}
    report = {
        "event": "hurricane_harvey",
        "window": {"start": "2017-08-27T00:00:00Z"},
        "input": {"pids": {"P1": "post-1"}},
        "sentences": [
            {"round": 0, "text": "Floodwater covers the road.", "citations": ["P1"], "gates": gates, "accepted": True}
        ],
    }
    packet = {"post-1": {"image_caption": {"text": "A flooded road."}}}
    result = evaluate_window(report, packet, TextOnlyScorer())
    assert result["full_all_gates_initial"] == result["full_all_gates_final"] == 1
    assert result["entailment_only_initial"] == 1
    assert result["text_only_g3_all_gates_initial"] == 0
    assert result["g3_decisions_changed"] == 1
    assert "| Overall | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 1 |" in summarize([result])
