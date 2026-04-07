# from pysyrev.bib_dset import WosDataset, OpenAlexDataset

# test = WosDataset(bibfile="/home/benjaminpillot/Documents/PRO/"
#                           "ABM_LITERATURE_REVIEW/WoS/DECEMBER_2025/savedrecs_all.bib")
# test = OpenAlexDataset(bibfile="/home/benjaminpillot/Documents/PRO/"
#                                "ABM_LITERATURE_REVIEW/OpenAlex/DECEMBER_2025/open_alex.csv")

from pysyrev.review import LLMReview
from pysyrev.config import Config
import pandas as pd

full_dataset = pd.read_csv("~/Documents/PRO/ABM_LITERATURE_REVIEW/NOTEBOOKS"
                           "/merged_wos_openalex_with_abstract_no_duplicate.csv")

config = Config.load("pysyrev/config_examples/config_abm.yaml")
reviewers = config.review.reviewers
workflow = config.review.workflow
text_inputs = config.review.text_inputs
env_file = config.review.env
llmreview = LLMReview(reviewers, workflow, text_inputs, env_file)
llmreview.run(full_dataset,
              config.review.export_to,
              sample_size=config.review.sample_size)