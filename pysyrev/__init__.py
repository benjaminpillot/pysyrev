from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pysyrev")
except PackageNotFoundError:
    __version__ = "unknown"

from pysyrev.pipeline import Pipeline, ALL_STAGES
from pysyrev.bibdata import BibDataset, WosDataset, OpenAlexDataset, ScopusDataset, PubmedDataset
from pysyrev.topic_model import TopicModel
from pysyrev.network import BibNetwork
from pysyrev.topic_report import TopicReport
from pysyrev.review import LLMReview, ReviewedDataset
from pysyrev.download import PaperDownloader