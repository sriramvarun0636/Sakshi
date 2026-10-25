import json

import numpy as np
import pandas as pd
from PIL import Image

from sakshi.data.features import EMBED_FILE, INDEX_FILE, TOKENS_FILE
from sakshi.provenance.index import ProvenanceIndex, build_index
from sakshi.types import ProvStatus


class Scorer:
    def score(self, embeddings):
        assert embeddings.shape[1] == 512
        return np.array([0.9, 0.1]) if len(embeddings) == 2 else np.array([0.1])


class LiveFeatures:
    def encode(self, images):
        embedding = np.zeros((len(images), 512), dtype=np.float32)
        embedding[:, 0] = 1
        return np.zeros((len(images), 50, 768), dtype=np.float32), embedding


class UniformScorer:
    def score(self, embeddings):
        return np.full(len(embeddings), 0.1, dtype=np.float32)


def test_index_aligns_sample_ids_and_derives_events(tmp_path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    token_map = np.lib.format.open_memmap(feature_dir / TOKENS_FILE, mode="w+", dtype=np.float16, shape=(2, 50, 768))
    token_map[:] = 0
    token_map.flush()
    del token_map
    embed_map = np.lib.format.open_memmap(feature_dir / EMBED_FILE, mode="w+", dtype=np.float32, shape=(2, 512))
    embed_map[:] = 0
    embed_map[:, 0] = 1
    embed_map.flush()
    del embed_map
    pd.DataFrame({"sample_id": ["b", "a"], "row": [0, 1], "ok": [True, True]}).to_parquet(feature_dir / INDEX_FILE)
    rows = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "dataset": ["cib", "cib"],
            "source": ["aidr_dt", "dmd"],
            "image_path": ["a.png", "b.png"],
            "tweet_id": ["123", None],
            "component": [1, 2],
            "event": [pd.NA, pd.NA],
            "folder": ["nepal_2015", pd.NA],
            "posted_at": pd.to_datetime(["2015-04-26T00:00:00Z", None]),
            "phash": pd.array([1, 2], dtype="UInt64"),
            "degenerate": [False, False],
            "width": [100, 200],
            "height": [80, 300],
        }
    )
    output = build_index(rows, feature_dir=feature_dir, output=tmp_path / "index.parquet", scorer=Scorer())
    index = pd.read_parquet(output)
    assert index.sample_id.tolist() == ["a", "b"]
    assert index.embed_row.tolist() == [1, 0]  # manifest and feature cache orders differ
    assert index.index_event.tolist() == ["aidr:nepal_2015", "dmd"]
    assert index.min_side.tolist() == [80, 200]
    assert index.is_graphic.tolist() == [True, False]
    onsets = json.loads((tmp_path / "onsets.json").read_text())
    assert onsets["aidr:nepal_2015"] == "2015-04-26T00:00:00+00:00"
    assert "hurricane_maria" in onsets
    manifest_path = tmp_path / "manifest.parquet"
    rows.to_parquet(manifest_path)
    provenance = ProvenanceIndex.load(output, feature_dir, manifest_path, graphic_scorer=Scorer(), live_featurizer=LiveFeatures())
    recycled = provenance.check("b", event="hurricane_maria", time=pd.Timestamp("2017-10-01T00:00:00Z"))
    assert recycled.status == ProvStatus.RECYCLED
    assert recycled.origin_post_id == "a" and recycled.index_size == 2
    assert provenance._query_from_sample("a", None, None).event == "aidr:nepal_2015"
    assert provenance.check().status == ProvStatus.NO_IMAGE
    prepared = provenance.prepare_query("b", event="hurricane_maria", time=pd.Timestamp("2017-10-01T00:00:00Z"))
    swept = provenance.candidates(prepared, min_tau_clip=0.8)
    assert provenance.check_query(prepared, candidates=swept, tau_clip=0.95) == recycled
    noisy = Image.fromarray(np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8))
    assert provenance.check(image=noisy).status == ProvStatus.SEEN_BEFORE
    provenance.graphic_scorer = UniformScorer()
    first = provenance.prepare_query(image=noisy, event="hurricane_maria", time=pd.Timestamp("2017-10-01T00:00:00Z"))
    second = provenance.prepare_query(image=noisy, event="hurricane_irma", time=pd.Timestamp("2017-09-01T00:00:00Z"))
    batched = provenance.prepare_image_batch([noisy, noisy], times=[first.time, second.time], events=[first.event, second.event])
    for individual, batch_item in zip((first, second), batched, strict=True):
        assert batch_item.post_id == individual.post_id
        assert batch_item.event == individual.event
        assert batch_item.time == individual.time
        assert batch_item.phash == individual.phash
        assert batch_item.is_graphic == individual.is_graphic
        assert np.array_equal(batch_item.embedding, individual.embedding)
