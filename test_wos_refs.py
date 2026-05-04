from pysyrev.bibdata import BibDataset, WosDataset
from pysyrev.core.config import Config

config = Config.load("pysyrev/config_examples/config_abm.yaml")
wos_bib = WosDataset.from_config(config.bib.wos)
# bib = BibDataset.from_config(config.bib)
wos_bib.clean_and_drop().resolve_references()

wos_bib.to_csv("/home/benjaminpillot/Documents/PRO/ABM_LITERATURE_REVIEW/PYSYREV/wos_oa.csv")