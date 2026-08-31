"""The review stage must propagate its freshly written reviewed_included.csv to
topic_model.doc_dataset, so the report recomputes its bibliographic networks from
THIS run's corpus — not the stale path Config.load auto-detected before the file
existed. Without it, the networks/connectivity/composite sections silently vanish
when the pre-existing path is empty or missing.
"""

import types

from pysyrev.pipeline import Pipeline


class _FakeExport:
    def __init__(self, path):
        self.included_docs = path


class _FakeReview:
    def __init__(self, path):
        self.export = _FakeExport(path)
        self.saved = False

    def run(self, dataset):
        return self

    def save(self):
        self.saved = True
        return self


def _config(topic_model):
    # Only the attributes the review branch of Pipeline.run touches.
    return types.SimpleNamespace(
        review=object(),
        topic_model=topic_model,
        topic_report=None,
    )


def test_review_propagates_included_docs_to_topic_model(monkeypatch):
    fresh = "/runs/2026-08-26T163601/reviewed_included.csv"
    import pysyrev.review as review_mod
    monkeypatch.setattr(review_mod.LLMReview, "from_config",
                        classmethod(lambda cls, cfg: _FakeReview(fresh)))

    tm = types.SimpleNamespace(doc_dataset="/runs/OLD/reviewed_included.csv")
    pipe = Pipeline(config=_config(tm))
    pipe.run(stages=["review"])

    assert pipe.review.saved
    assert tm.doc_dataset == fresh          # stale path overwritten with the fresh one


def test_propagation_is_noop_without_topic_model(monkeypatch):
    fresh = "/runs/latest/reviewed_included.csv"
    import pysyrev.review as review_mod
    monkeypatch.setattr(review_mod.LLMReview, "from_config",
                        classmethod(lambda cls, cfg: _FakeReview(fresh)))

    pipe = Pipeline(config=_config(topic_model=None))
    pipe.run(stages=["review"])             # must not raise when topic_model is absent
    assert pipe.review.saved
