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
from dataclasses import dataclass
from typing import List, Union

import nest_asyncio
from lattereview.agents import TitleAbstractReviewer

from pysyrev.core.config import ReviewConfig, ReviewerConfig
from pysyrev.core.llm import run_review, build_workflow_schema, build_reviewer


API_PAUSE = 30


def _output_filename(base, run_name=None, index=None):
    """Build an output filename: <base>[_<run_name>][_<index>].csv"""
    parts = [base]
    if run_name:
        parts.append(run_name)
    if index is not None:
        parts.append(str(index))
    return '_'.join(parts) + '.csv'


# =============================================================================
# Helpers — moved out of __post_init__ for clarity.
# =============================================================================

def _input_description(text_inputs: List[str]) -> str:
    """Format the input description string consumed by reviewers.
    Kept as a helper so the format choice is one-liner-changeable."""
    return f'article {text_inputs[0]}/{text_inputs[1]}/{text_inputs[2]}'


def _reviewer_kwargs(reviewer_config: ReviewerConfig,
                     review_config: ReviewConfig,
                     input_description: str) -> dict:
    """Build the dict consumed by `build_reviewer`. Combines what is in the
    reviewer YAML entry with the cross-section fields (criteria, input
    format) provided at the review level."""
    return {
        **reviewer_config.__dict__,
        'inclusion_criteria': review_config.inclusion_criteria,
        'exclusion_criteria': review_config.exclusion_criteria,
        'input_description': input_description,
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
    decision_rule:   str
    text_inputs:     List[str]
    reviewers:       List[TitleAbstractReviewer]
    workflow_schema: List[dict]
    export_to:       str
    batch_size:      int
    api_pause:       float
    sample_size:     Union[int, None] = None   # None when sampling is disabled
    run_name:        Union[str, None] = None   # None → default filenames

    _reviewed_dataset = ReviewedDataset()

    # ---- bridge from configuration --------------------------------------

    @classmethod
    def from_config(cls, config: ReviewConfig) -> 'LLMReview':
        """
        Build an LLMReview from a parsed ReviewConfig.

        Loads the environment file, builds reviewers from their YAML entries
        (combined with the review-level criteria), and assembles the
        workflow schema.
        """
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
            export_to       = config.export_to,
            batch_size      = config.batch_size,
            api_pause       = config.api_pause,
            sample_size     = config.sample_size,
            run_name        = config.run_name,
        )

    # ---- runtime --------------------------------------------------------

    def run(self, dataset):
        """Execute the review on `dataset`. Output directory and batching
        parameters come from the config-supplied attributes."""
        nest_asyncio.apply()
        subset_file_fn = lambda n: os.path.join(
            self.export_to,
            _output_filename('reviewed_subset', self.run_name, index=n),
        )
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
        self._reviewed_dataset.included_docs = reviewed_ds.loc[reviewed_ds["final_score"] > 3, :]

        return self

    def save(self):

        out_file = lambda x : os.path.join(
            self.export_to,
            _output_filename(f'reviewed_dataset_{x}', self.run_name),
        )

        self._reviewed_dataset.total_docs.to_csv(out_file("total_docs"), index=False)
        self._reviewed_dataset.included_docs.to_csv(out_file("included_docs"), index=False)

        return self
