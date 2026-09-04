"""
Rounds, votes and entry points — the review as the config describes it.

A workflow is a list of rounds; each round screens what the previous one left
unsettled, with its own reviewers and its own inclusion window. Once every
round has spoken, :func:`compute_final_score` turns their evaluations into one
decision per document, by majority or by mean.
"""

import asyncio
import os.path
import time
from functools import partial
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm_sync

from pysyrev.core.llm.batch_run import BatchRun, _review_round_deferred
from pysyrev.core.llm.reviewer import REVIEW_SCORE, _review_all


def build_workflow_schema(workflow, reviewers, text_inputs, decision_rule):
    workflow_schema = [{} for _ in range(len(workflow))]
    eval_keys_before = []   # _evaluation columns from previous rounds (used for filter)

    for round_idx, dict_from in enumerate(workflow):
        round_name     = dict_from["round"]
        reviewer_names = dict_from["reviewers"]

        eval_keys      = [f"round-{round_name}_{name}_evaluation" for name in reviewer_names]
        reasoning_keys = [f"round-{round_name}_{name}_reasoning"  for name in reviewer_names]

        workflow_schema[round_idx]["round"]       = round_name
        workflow_schema[round_idx]["reviewers"]   = [r for r in reviewers if r.name in reviewer_names]
        workflow_schema[round_idx]["text_inputs"] = list(text_inputs)

        if eval_keys_before:
            workflow_schema[round_idx]["filter"] = partial(
                eval_filter_func,
                eval_keys     = list(eval_keys_before),
                decision_rule = decision_rule,
            )

        eval_keys_before += eval_keys
        text_inputs = text_inputs + eval_keys + reasoning_keys  # new list, not in-place

    return workflow_schema


_warned_missing_inputs: set = set()


def _warn_missing_text_inputs(dataset: pd.DataFrame, text_inputs: List[str],
                              round_name: str) -> None:
    """Report text_inputs that match no column, instead of dropping them mutely.

    ``_build_texts`` skips any column it cannot find, which is what makes a
    round's later evaluation/reasoning inputs work. The same leniency hides a
    plain config typo: ask for `keywords` when the corpus column is
    `author_keywords` and the field is simply never sent to the reviewers, with
    nothing in the logs to say so.

    Warned once per round: the workflow re-runs for every checkpoint chunk, and
    a warning repeated fifty times is a warning nobody reads.
    """
    missing = [c for c in text_inputs if c not in dataset.columns]
    key = (round_name, tuple(missing))
    if missing and key not in _warned_missing_inputs:
        _warned_missing_inputs.add(key)
        print(f"[round {round_name}] text_inputs {missing} match no column in "
              f"the dataset and are not sent to the reviewers. Available: "
              f"{', '.join(sorted(dataset.columns))}")


def _build_texts(dataset: pd.DataFrame, text_inputs: List[str]) -> List[str]:
    texts = []
    for _, row in dataset.iterrows():
        parts = [f"{col}: {row[col]}"
                 for col in text_inputs
                 if col in row.index and pd.notna(row[col])]
        texts.append("\n".join(parts))
    return texts


async def _run_workflow(dataset: pd.DataFrame, workflow_schema: list,
                        batch_run: Optional[BatchRun] = None) -> pd.DataFrame:
    result = dataset.copy()
    for round_schema in workflow_schema:
        round_name  = round_schema["round"]
        reviewers   = round_schema["reviewers"]
        text_inputs = round_schema["text_inputs"]
        filter_fn   = round_schema.get("filter")

        if filter_fn:
            mask = result.apply(filter_fn, axis=1)
        else:
            mask = pd.Series(True, index=result.index)
        subset_idx = result.index[mask]

        _warn_missing_text_inputs(result, text_inputs, round_name)
        texts = _build_texts(result.loc[subset_idx], text_inputs)

        if batch_run is not None and reviewers and texts:
            # One deferred submission for the whole round, all reviewers at once.
            round_evals = await _review_round_deferred(
                reviewers, texts, round_name, batch_run)
        else:
            round_evals = {r.name: await _review_all(texts, r) for r in reviewers}

        for reviewer in reviewers:
            eval_col      = f"round-{round_name}_{reviewer.name}_evaluation"
            reasoning_col = f"round-{round_name}_{reviewer.name}_reasoning"
            evals = round_evals[reviewer.name]
            result.loc[subset_idx, eval_col]      = [e["evaluation"] for e in evals]
            result.loc[subset_idx, reasoning_col] = [e["reasoning"]  for e in evals]

    print("\n====== Finished reviewing ======\n")
    return result


# ── Score computation ──────────────────────────────────────────────────────

def compute_final_score(decision_rule, scores: pd.DataFrame):
    if decision_rule == "mean":
        return scores.mean(axis=1)
    else:  # majority
        final_scores = scores.median(axis=1)
        final_scores[final_scores == 3] = scores.iloc[final_scores == 3, -1]
        return final_scores


def eval_filter_func(row, eval_keys, decision_rule):
    score = np.asarray([int(row[eval_key])
                        for eval_key in eval_keys
                        if pd.notna(row.get(eval_key))])

    if decision_rule == "mean":
        if score.mean() == 3.:
            return True
    elif decision_rule == "majority":
        if np.count_nonzero(score > 3) == np.count_nonzero(score < 3):
            return True
        if np.count_nonzero(score == 3) / score.size >= .5:
            return True
    return False


# ── Entry points ───────────────────────────────────────────────────────────

def process_full(dataset, workflow_schema, batch_run: Optional[BatchRun] = None):
    return asyncio.run(_run_workflow(dataset, workflow_schema, batch_run))


def process_per_batch(dataset, workflow_schema, batch_size, pause, subset_file_fn):
    subsets = [dataset.iloc[i: min(i + batch_size, len(dataset)), :]
               for i in range(0, len(dataset), batch_size)]

    results = []
    for idx, subset in tqdm_sync(enumerate(subsets), total=len(subsets),
                                 desc="Batch", unit="batch"):
        if subset_file_fn is not None:
            subset_file = subset_file_fn(idx)
            if not os.path.exists(subset_file):
                reviewed_subset = asyncio.run(_run_workflow(subset, workflow_schema))
                reviewed_subset.to_csv(subset_file, index=False)
                time.sleep(pause)
            results.append(pd.read_csv(subset_file))
        else:
            reviewed_subset = asyncio.run(_run_workflow(subset, workflow_schema))
            results.append(reviewed_subset)
            time.sleep(pause)

    return pd.concat(results, ignore_index=True)


def run_review(dataset, workflow_schema, decision_rule,
               batch_size, sample_size, pause, subset_file_fn=None,
               batch_run: Optional[BatchRun] = None,
               sample_seed: Optional[int] = None):

    def ds_eval_keys():
        return [f"round-{s['round']}_{r.name}_evaluation"
                for s in workflow_schema for r in s["reviewers"]]

    if sample_size:
        # A seed pins the subset, so two runs differ by what the config changed
        # rather than by which articles they happened to draw.
        dataset = dataset.sample(sample_size, random_state=sample_seed)

    if batch_run is not None:
        # Checkpoint chunking is deliberately bypassed here. A deferred run
        # waits once per submission, so chunking a 4 600-document corpus into
        # nineteen pieces buys nineteen waits of up to 24 h for no saving —
        # and the batch is already the checkpoint, since its results stay
        # retrievable for weeks and its id is on disk.
        if batch_size and batch_size < len(dataset):
            print(f"[batch] batch_size={batch_size} is ignored in deferred-batch "
                  f"mode: each round is submitted in one go and the batch id "
                  f"itself is the restart point.")
        reviewed_dataset = process_full(dataset, workflow_schema, batch_run)
    elif batch_size and batch_size < len(dataset):
        reviewed_dataset = process_per_batch(dataset, workflow_schema, batch_size,
                                             pause, subset_file_fn)
    else:
        reviewed_dataset = process_full(dataset, workflow_schema)

    reviewed_dataset[REVIEW_SCORE] = compute_final_score(
        decision_rule, reviewed_dataset.loc[:, ds_eval_keys()]
    )
    return reviewed_dataset
