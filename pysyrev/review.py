from typing import List

import nest_asyncio
from dotenv import load_dotenv
from lattereview.agents import TitleAbstractReviewer

from pysyrev.core.llm import run_review, build_workflow_schema, build_reviewer


API_PAUSE = 30


class LLMReview:

    decision_rule : str
    env : str
    reviewers : List[TitleAbstractReviewer]
    text_inputs : List[str]
    workflow : List[dict]
    workflow_schema : List[dict] = None

    def __init__(self, reviewers, workflow, text_inputs,
                 env_file, decision_rule="majority"):
        """ Initialize class instance

        Parameters
        ----------
        reviewers: List[dict]
            List of reviewers as dict
        workflow: List[dict]
            List of dict corresponding to each round of the review
            (See LatteReview doc)
        text_inputs: List[str]
            List of column names in dataset for review
            (e.g. "abstract", "title", "keywords")
        env_file: str
            path to environment file
            (where are stored API keys, etc.)
        decision_rule: str
            Decision rule for including or excluding document
            ("mean" or "majority")
        """
        self.decision_rule = decision_rule
        self.text_inputs = text_inputs
        self._build_reviewers(reviewers)
        self._build_workflow_schema(workflow)
        self._load_env(env_file)

    def _build_reviewers(self, reviewers):
        """ Build reviewers for review

        Parameters
        ----------
        reviewers: List[dict]
            each reviewer is a dict with the following keys:
            - name : str
            - reasoning : str ("brief", "cot")
            - model_id : str
            - host : str
            - provider : str ("litellm", "open-ai", "ollama")
            - max_tokens : int
            - temperature : float
            - reasoning_effort : str ("low", "medium", "high")
            - inclusion_criteria : str
            - exclusion_criteria : str
            - input_description : str
            - backstory : str
            - additional_context : str

        Returns
        -------

        """
        self.reviewers = []
        for reviewer in reviewers:
            self.reviewers.append(build_reviewer(**reviewer))

        return self

    def _build_workflow_schema(self, workflow):
        """

        Parameters
        ----------
        workflow: List[dict]

        Returns
        -------

        """
        self.workflow_schema = build_workflow_schema(workflow,
                                                     self.reviewers,
                                                     self.text_inputs,
                                                     self.decision_rule)

        return self

    def _load_env(self, env):
        """ Load environment file

        Returns
        -------

        """
        self.env = env

        load_dotenv(env)

        return self

    def run(self,
            dataset,
            out_dir,
            batch_size=None,
            sample_size=None,
            pause=API_PAUSE):
        """

        Parameters
        ----------
        dataset: pandas.Dataframe
        out_dir: str
            path to directory for storing outputs
        batch_size: int
        sample_size: int
        pause: int
            time.sleep for limiting nb of API requests per min

        Returns
        -------

        """
        nest_asyncio.apply()

        return run_review(dataset,
                          self.workflow_schema,
                          out_dir,
                          batch_size,
                          sample_size,
                          pause)
