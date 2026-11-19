"""A topic-model run that cannot start leaves nothing behind."""

from unittest import mock

import pytest

from pysyrev.topic_model import TopicModel


def test_no_dataset_raises_before_creating_a_run_dir(tmp_path):
    model = TopicModel(
        doc_dataset=None, allow_abbrev=False, distance="euclidean",
        bertopic_model=mock.Mock(), topic_distribution=mock.Mock(),
        nr_repr_docs=3, export_dir=str(tmp_path), n_neighbors=[5],
        n_components=[3])
    with pytest.raises(ValueError, match="reviewed_included.csv"):
        model.run()
    # An empty run directory would pass for the latest run with the report.
    assert list(tmp_path.iterdir()) == []
