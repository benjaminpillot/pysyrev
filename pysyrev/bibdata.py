from thefuzz import fuzz

from pysyrev.core.bib import fetch_citations, generate_bib, generate_oa_bib, merge_bibs, extract_documents, \
    check_bib_dataset
from pysyrev.core.utils import clean_doi, clean_abstract, has_abstract


class BibDataset:

    _db = None
    _bib_dataset = None
    _pbx_probe = None

    def __init__(self, bibfile=None, bib_dataset=None):
        """

        Parameters
        ----------
        bibfile: str
            path to bib file (.csv, .bib, etc.)
        bib_dataset: pandas.DataFrame
        """
        try:
            self.generate_bib(bibfile)
        except TypeError:
            self._bib_dataset = check_bib_dataset(bib_dataset)


    def clean_and_drop(self):
        """ Clean DOI and abstract columns, drop no-abstract rows

        Returns
        -------

        """
        # CLEAN
        self._bib_dataset.doi = self._bib_dataset.doi.apply(clean_doi)
        self._bib_dataset.abstract = self._bib_dataset.abstract.apply(clean_abstract)

        # Does it have abstract ? DROP no-abstract rows
        has_abstract_idx = self._bib_dataset.abstract.apply(has_abstract)
        self._bib_dataset.drop(self._bib_dataset.index[~has_abstract_idx],
                               axis=0,
                               inplace=True)

        # Reset index
        self._bib_dataset.reset_index(inplace=True)

        return self

    def extract_documents(self, document_type, year=1900,
                          nb_citations=0, language="en",
                          scorer=fuzz.partial_token_sort_ratio,
                          score_cutoff=90):
        """ Create sub bib dataset through metadata selection

        Parameters
        ----------
        document_type: str or list[str]
            list of valid document types (article, review, etc.)
        year: int or tuple[int, int]
            min year threshold or range (min, max)
        nb_citations: int or tuple[int, int]
            min nb of citations threshold or range (min, max)
        language: str or list[str]
        scorer: function
        score_cutoff: int

        Returns
        -------

        """
        if isinstance(document_type, str):
            document_type = [document_type]
        return self.__class__(bib_dataset=extract_documents(self._bib_dataset,
                                                            document_type,
                                                            language,
                                                            year,
                                                            nb_citations,
                                                            scorer,
                                                            score_cutoff))

    def fetch_abstracts(self):
        """ Use online APIs to retrieve abstracts

        Returns
        -------

        """
        #TODO
        pass

    def fetch_citations(self):
        """ Fetch citation count through
        Semantic Scholar or CrossRef

        Returns
        -------

        """
        try:
            return fetch_citations(self.doi)
        except AttributeError:
            raise ValueError("Bibliography has not been generated yet: no DOI available")

    def generate_bib(self, bibfile, del_duplicated=True):
        """ Generate bib using a custom version of pbx_probe

        Parameters
        ----------
        bibfile: str or bytes or os.PathLike
        del_duplicated: bool

        Returns
        -------

        """
        self._pbx_probe, self._bib_dataset = generate_bib(bibfile,
                                                          db = self._db,
                                                          del_duplicated=del_duplicated)
        # self._bib_dataset = self._pbx_probe.data

        return self

    def merge(self, others, doi_similarity=100, title_similarity=98):
        """ Merge dataset with other(s)

        Parameters
        ----------
        others: List[BibDataset]
        doi_similarity: int
            FuzzyWuzzy similarity threshold
        title_similarity: int
            FuzzyWuzzy similarity threshold

        Returns
        -------

        """
        datasets = [self._bib_dataset] + [other.dataset for other in others]

        return self.__class__(merge_bibs(datasets,
                                         doi_similarity,
                                         title_similarity))

    def sample(self, size=100, random_state=None):
        """ Sample dataset at random

        Parameters
        ----------
        size: int
            Sample size
        random_state: int
            Seed for random number generator

        Returns
        -------
        new instance of BibDataset

        """
        return self.__class__(bib_dataset=self._bib_dataset.sample(n=size,
                                                                   random_state=random_state,
                                                                   axis=0,
                                                                   ignore_index=True))

    def to_csv(self, file_name):
        """ Write bib to csv file

        Returns
        -------

        """
        self.dataset.to_csv(file_name)

    @property
    def doi(self):
        return self._bib_dataset.doi

    @property
    def citation_count(self):
        return self._bib_dataset.cited_by

    @property
    def dataset(self):
        return self._bib_dataset


class WosDataset(BibDataset):

    _db = "wos"


class OpenAlexDataset(BibDataset):

    _db = "scopus"

    def generate_bib(self, bibfile, **kwargs):

        new_bib_file = generate_oa_bib(bibfile)

        super().generate_bib(new_bib_file, **kwargs)


class ScopusDataset(BibDataset):

    _db = "scopus"

    pass


class PubmedDataset(BibDataset):

    _db = "pubmed"

    pass