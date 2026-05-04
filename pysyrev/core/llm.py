import asyncio
import os.path
import time
from functools import partial

import numpy as np
import pandas as pd
from lattereview.agents import TitleAbstractReviewer
from lattereview.providers import LiteLLMProvider, OpenAIProvider, OllamaProvider
from lattereview.workflows import ReviewWorkflow

MAX_RETRIES = 2
MAX_CONCURRENT_REQUESTS = 30


def build_reviewer(name, provider, model_id,
                   host, reasoning, max_tokens,
                   temperature, reasoning_effort,
                   backstory, additional_context,
                   inclusion_criteria, exclusion_criteria,
                   input_description,
                   **kwargs):

    if provider == "litellm":
        llm_provider = LiteLLMProvider
    elif provider == "open-ai":
        llm_provider = OpenAIProvider
    else:
        llm_provider = OllamaProvider

    max_retries = kwargs["max_retries"] if "max_retries" in kwargs.keys() else MAX_RETRIES
    max_concurrent_requests = kwargs["max_concurrent_requests"] if \
        "max_concurrent_requests" in kwargs.keys() else MAX_CONCURRENT_REQUESTS

    return TitleAbstractReviewer(max_retries=max_retries,
                                 provider=llm_provider(model=model_id, host=host),
                                 name=name,
                                 backstory=backstory,
                                 inclusion_criteria=inclusion_criteria,
                                 exclusion_criteria=exclusion_criteria,
                                 input_description=input_description,
                                 reasoning=reasoning,
                                 max_concurrent_requests=max_concurrent_requests,
                                 model_args={"max_tokens": max_tokens,
                                             "temperature": temperature,
                                             "reasoning_effort": reasoning_effort},
                                 additional_context=additional_context)


def build_workflow_schema(workflow, reviewers, text_inputs, decision_rule):

    workflow_schema = [{} for _ in range(len(workflow))]  # This is the right form to avoid
                                                          # pointing to the same dictionary n times !!!
    eval_round_before = []

    for n, dict_from in enumerate(workflow):
        eval_keys = [f"round-{dict_from["round"]}_{reviewer_name}_output"
                     for reviewer_name in dict_from["reviewers"]]
        workflow_schema[n]["round"] = dict_from["round"]
        workflow_schema[n]["reviewers"] = [reviewer for reviewer in reviewers
                                           if reviewer.name in dict_from["reviewers"]]
        workflow_schema[n]["text_inputs"] = [input_ for input_ in text_inputs]  # warning : list is mutable !
                                                                                # Use comprehension list instead
        text_inputs += eval_keys

        if eval_round_before:
            workflow_schema[n]["filter"] = partial(eval_filter_func,
                                                   eval_keys=[eval_key for eval_key in eval_round_before],
                                                   decision_rule=decision_rule)

        eval_round_before += eval_keys

    return workflow_schema


def eval_filter_func(row, eval_keys, decision_rule):
    score = np.asarray([int(row[eval_key]["evaluation"])
                        for eval_key in eval_keys])

    if decision_rule == "mean":
        if score.mean() == 3.:
            return True
    elif decision_rule == "majority":
        if np.count_nonzero(score > 3) == np.count_nonzero(score < 3):
            return True
        if np.count_nonzero(score == 3) / score.size > .5:
            return True
    return False


def process_full(dataset, workflow_schema):

    # reviewed_dataset = asyncio.run(review(dataset, workflow_schema))
    # reviewed_dataset.to_csv(out_file, index=False)
    #
    # return 0
    return asyncio.run(review(dataset, workflow_schema))


def process_per_batch(dataset, workflow_schema,
                      batch_size, pause, subset_file_fn):

    subsets = [dataset.iloc[i: min(i + batch_size, len(dataset)), :]
               for i in range(0, len(dataset), batch_size)]

    for n, subset in enumerate(subsets):
        subset_file = subset_file_fn(n)
        if not os.path.exists(subset_file):
            reviewed_subset = asyncio.run(review(subset, workflow_schema))
            reviewed_subset.to_csv(subset_file, index=False)
            time.sleep(pause)  # Limit API nb of requests per min

    # Merge all subset files into the final output file.
    reviewed_dataset = pd.concat(
        [pd.read_csv(subset_file_fn(n)) for n in range(len(subsets))],
        ignore_index=True,
    )

    return reviewed_dataset
    # reviewed_dataset.to_csv(out_file, index=False)
    #
    # return 0


async def review(dataset, workflow_schema):

    review_workflow = ReviewWorkflow(workflow_schema=workflow_schema)
    updated_dataset = await review_workflow(dataset)

    print(f"\n====== Finished reviewing ======\n")
    print(f"\nTotal cost: {review_workflow.get_total_cost()}")
    print("-" * 100)

    return updated_dataset


def run_review(dataset, workflow_schema,
               batch_size, sample_size, pause, subset_file_fn=None):

    if sample_size:
        dataset = dataset.sample(sample_size)

    if batch_size and batch_size < len(dataset):
         return process_per_batch(dataset,
                                  workflow_schema,
                                  batch_size,
                                  pause,
                                  subset_file_fn)
    else:
        return process_full(dataset,
                            workflow_schema)

    # TODO
    #  reviewed_dataset["final_score"] =
