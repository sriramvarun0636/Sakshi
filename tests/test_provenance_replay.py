import pandas as pd

from sakshi.provenance.replay import replay_event, summary_table
from sakshi.types import ProvenanceResult, ProvStatus


class FakeIndex:
    def __init__(self):
        self.manifest = pd.DataFrame(
            {
                "sample_id": ["later", "earlier", "other"],
                "dataset": ["crisismmd"] * 3,
                "event": ["hurricane_maria", "hurricane_maria", "hurricane_irma"],
                "posted_at": pd.to_datetime(["2017-09-22T00:00:00Z", "2017-09-21T00:00:00Z", "2017-09-10T00:00:00Z"]),
            }
        ).set_index("sample_id", drop=False)
        self.seen = []

    def check(self, sample_id):
        self.seen.append(sample_id)
        return ProvenanceResult(
            post_id=sample_id,
            status=ProvStatus.RECYCLED if sample_id == "later" else ProvStatus.NOVEL,
            tier=None,
            similarity=None,
            phash_distance=None,
            origin_post_id=None,
            origin_event=None,
            origin_first_seen=None,
            top_match_post_id=None,
            query_time=None,
            query_is_graphic=False,
            index_size=3,
        )


def test_replay_orders_posts_by_time_and_reports_all_statuses(tmp_path):
    index = FakeIndex()
    output = replay_event(index, "hurricane_maria", output_dir=tmp_path)
    frame = pd.read_parquet(output)
    assert index.seen == ["earlier", "later"]
    assert frame.status.tolist() == ["novel", "recycled"]
    markdown = summary_table({"hurricane_maria": frame})
    assert "| hurricane_maria | 2 |" in markdown
    assert "template_match" in markdown
