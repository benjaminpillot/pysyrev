"""
High-level Pipeline class for running the pysyrev literature review pipeline
from Python or a Jupyter notebook.

Usage
-----
  from pysyrev import Pipeline

  # Full pipeline
  pipeline = Pipeline.from_config("config.yaml")
  pipeline.run()

  # Selected stages (in canonical order: bib → review → bib-network → topic-model)
  pipeline.run(stages=["bib", "review"])

  # Results are available as attributes
  pipeline.bib.dataset          # pd.DataFrame
  pipeline.review.included_docs # pd.DataFrame
  pipeline.network              # BibNetwork
  pipeline.topic                # TopicModel

Stage results persist on the instance between run() calls, so data is passed
in memory when stages are chained:

  pipeline.run(stages=["bib"])
  pipeline.run(stages=["review"])   # uses pipeline.bib.dataset automatically
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from pysyrev.bibdata import BibDataset
    from pysyrev.core.config import Config
    from pysyrev.network import BibNetwork
    from pysyrev.review import LLMReview
    from pysyrev.topic_model import TopicModel

ALL_STAGES = ['bib', 'review', 'bib-network', 'topic-model']


@dataclass
class Pipeline:
    """Literature review pipeline.

    Instantiate with :meth:`from_config`, then call :meth:`run`.
    """
    config:  'Config'
    bib:     Optional['BibDataset'] = field(default=None, init=False)
    review:  Optional['LLMReview']  = field(default=None, init=False)
    network: Optional['BibNetwork'] = field(default=None, init=False)
    topic:   Optional['TopicModel'] = field(default=None, init=False)

    @classmethod
    def from_config(cls, config: Union[str, 'Config']) -> 'Pipeline':
        """Create a Pipeline from a YAML config path or a Config object."""
        from pysyrev.core.config import Config
        if not isinstance(config, Config):
            config = Config.load(config)
        return cls(config=config)

    def run(self, stages: Optional[List[str]] = None) -> 'Pipeline':
        """Run one or more pipeline stages.

        Parameters
        ----------
        stages : list of str, optional
            Subset of stages to run. Defaults to all four stages in order.
            Valid values: 'bib', 'review', 'bib-network', 'topic-model'.
            Stages are always executed in canonical order regardless of the
            order they appear in the list.

        Returns
        -------
        self
            The Pipeline instance (for chaining).
        """
        if stages is None:
            stages = ALL_STAGES

        unknown = set(stages) - set(ALL_STAGES)
        if unknown:
            raise ValueError(
                f"Unknown stage(s) {unknown}. Valid: {ALL_STAGES}"
            )

        ordered = [s for s in ALL_STAGES if s in stages]

        if 'bib' in ordered:
            from pysyrev.bibdata import BibDataset
            self.bib = BibDataset.from_config(self.config.bib)

        if 'review' in ordered:
            from pysyrev.review import LLMReview
            self.review = LLMReview.from_config(self.config.review)
            dataset = self.bib.dataset if self.bib is not None else None
            self.review.run(dataset).save()

        if 'bib-network' in ordered:
            from pysyrev.network import BibNetwork
            self.network = BibNetwork.from_config(self.config.bib_network)
            dataset = self.review.included_docs if self.review is not None else None
            self.network.run(dataset)
            if self.config.bib_network.export is not None:
                self.network.save()

        if 'topic-model' in ordered:
            from pysyrev.topic_model import TopicModel
            self.topic = TopicModel.from_config(self.config.topic_model)
            dataset = self.review.included_docs if self.review is not None else None
            self.topic.run(dataset)

        return self
