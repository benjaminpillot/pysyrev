"""
LLM-based review pipeline.

LLMReview holds runtime objects ready to execute a multi-reviewer review on
a dataset. It mirrors TopicModel: pure runtime, no YAML knowledge, built
from a parsed ReviewConfig via `from_config`.

The bridge from ReviewConfig to the runtime layer also fabricates the
`input_description` string and propagates the inclusion/exclusion criteria
into each reviewer dict — these are presentation/runtime concerns, not
config concerns, so they live here rather than in config.py.
"""

import os
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Union

import nest_asyncio

from pysyrev.core.config import ReviewConfig, ReviewerConfig, ReviewExportConfig
from pysyrev.core.llm import (run_review, build_workflow_schema, build_reviewer,
                               Reviewer, REVIEW_SCORE)

REVIEWED_SUBSET: str = "reviewed_subset"


def _subset_filename(index):
    """Name for an intermediate per-batch file: reviewed_subset_<n>.csv"""
    return f"{REVIEWED_SUBSET}_{index}.csv"


# =============================================================================
# Helpers
# =============================================================================

def _input_description(text_inputs: List[str]) -> str:
    return f'article {text_inputs[0]}/{text_inputs[1]}/{text_inputs[2]}'


def _reviewer_kwargs(reviewer_config: ReviewerConfig,
                     review_config: ReviewConfig,
                     input_description: str) -> dict:
    """Build the dict consumed by `build_reviewer`. Combines reviewer-level
    fields with cross-section fields (criteria, input format, section defaults)."""
    return {
        **reviewer_config.__dict__,
        'inclusion_criteria':      review_config.inclusion_criteria,
        'exclusion_criteria':      review_config.exclusion_criteria,
        'input_description':       input_description,
        # reviewer-level overrides section-level; section-level overrides module default
        'max_retries':             (reviewer_config.max_retries
                                    or review_config.max_retries),
        'max_concurrent_requests': (reviewer_config.max_concurrent_requests
                                    or review_config.max_concurrent_requests),
        'items_per_call':          (reviewer_config.items_per_call
                                    or review_config.items_per_call),
    }


# =============================================================================
# Runtime model
# =============================================================================

@dataclass
class ReviewedDataset:
    total_docs:     pd.DataFrame = None
    included_docs:  pd.DataFrame = None


@dataclass
class LLMReview:
    """
    Runtime LLM review pipeline.

    Attributes mirror what ReviewConfig describes, plus the few derived
    runtime objects (reviewers, workflow_schema). Use `from_config` to
    build it from a parsed ReviewConfig — direct construction is also
    supported for tests or programmatic use.
    """
    text_inputs:       List[str]
    reviewers:         List[Reviewer]
    workflow_schema:   List[dict]
    export:            ReviewExportConfig
    decision_rule:     str
    batch_size:        int
    api_pause:         float
    sample_size:       Union[int, None]
    doc_dataset:       Union[str, None]
    _reviewed_dataset: ReviewedDataset = field(default_factory=ReviewedDataset, init=False, repr=False)

    # ---- bridge from configuration --------------------------------------

    @classmethod
    def from_config(cls, config: ReviewConfig) -> 'LLMReview':
        """Build an LLMReview from a parsed ReviewConfig."""
        config.export.resolve()
        input_description = _input_description(config.text_inputs)
        reviewers = [
            build_reviewer(**_reviewer_kwargs(rc, config, input_description))
            for rc in config.reviewers
        ]
        workflow_schema = build_workflow_schema(
            config.workflow,
            reviewers,
            config.text_inputs,
            config.decision_rule,
        )

        return cls(
            decision_rule   = config.decision_rule,
            text_inputs     = config.text_inputs,
            reviewers       = reviewers,
            workflow_schema = workflow_schema,
            export          = config.export,
            batch_size      = config.batch_size,
            api_pause       = config.api_pause,
            sample_size     = config.sample_size,
            doc_dataset     = config.doc_dataset,
        )

    # ---- runtime --------------------------------------------------------

    def run(self, dataset=None):
        """Execute the review. If *dataset* is None, load from ``doc_dataset``."""
        if dataset is None:
            if not self.doc_dataset:
                raise ValueError(
                    "No dataset provided: pass a DataFrame to run() or set "
                    "doc_dataset in the review section of your config."
                )
            dataset = pd.read_csv(self.doc_dataset)
        nest_asyncio.apply()
        if self.export.cache_dir:
            subset_file_fn = lambda n: os.path.join(
                self.export.cache_dir,
                _subset_filename(n),
            )
        else:
            subset_file_fn = None
        reviewed_ds = run_review(
            dataset,
            self.workflow_schema,
            self.decision_rule,
            self.batch_size,
            self.sample_size,
            self.api_pause,
            subset_file_fn,
        )
        self._reviewed_dataset.total_docs = reviewed_ds
        self._reviewed_dataset.included_docs = reviewed_ds.loc[reviewed_ds[REVIEW_SCORE] > 3, :]

        return self

    def save(self):
        self._reviewed_dataset.total_docs.to_csv(self.export.total_docs, index=False)
        self._reviewed_dataset.included_docs.to_csv(self.export.included_docs, index=False)
        return self

    @property
    def included_docs(self) -> pd.DataFrame:
        if self._reviewed_dataset.included_docs is None:
            raise ValueError("Review has not been run yet — call run() first.")
        return self._reviewed_dataset.included_docs
