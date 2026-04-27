import pandas as pd

from pysyrev.bibdata import WosDataset, OpenAlexDataset
from pysyrev.config import Config

# ds = pd.read_csv("/home/benjaminpillot/Documents/PRO/"
#                  "ABM_LITERATURE_REVIEW/OpenAlex/DECEMBER_2025/open_alex.csv",
#                  low_memory=False,
#                  lineterminator='\n')
# ds_oa = ds.iloc[0:1000]
# oa = OpenAlexDataset(bibfile="/home/benjaminpillot/Documents/PRO/"
#                              "ABM_LITERATURE_REVIEW/OpenAlex/DECEMBER_2025/open_alex.csv")
#
# wos = WosDataset(bibfile="/home/benjaminpillot/Documents/PRO/"
#                          "ABM_LITERATURE_REVIEW/WoS/DECEMBER_2025/savedrecs_all.bib")
# wos_oa = wos.merge([oa]).clean_and_drop().extract_documents("article",
#                                                             2020,
#                                                             10,
#                                                             "english"
#                                                             )
# wos_oa.to_csv("/home/benjaminpillot/Documents/PRO/ABM_LITERATURE_REVIEW/PYSYREV/wos_oa_articles.csv")

# from pysyrev.review import LLMReview
# from pysyrev.config import Config
# import pandas as pd
#
# full_dataset = pd.read_csv("~/Documents/PRO/ABM_LITERATURE_REVIEW/NOTEBOOKS"
#                            "/merged_wos_openalex_with_abstract_no_duplicate.csv")
#
config = Config.load("pysyrev/config_examples/config_abm.yaml")
reviewers = config.review.reviewers
# workflow = config.review.workflow
# text_inputs = config.review.text_inputs
# env_file = config.review.env
# llmreview = LLMReview(reviewers, workflow, text_inputs, env_file)
# llmreview.run(full_dataset,
#               config.review.export_to,
#               sample_size=config.review.sample_size)