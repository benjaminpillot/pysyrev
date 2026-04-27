from rapidfuzz import fuzz

from pysyrev.core.bib import fetch_citations, generate_bib, generate_oa_bib, extract_documents, \
    check_bib_dataset
from pysyrev.core.merge_bibs import merge_bibs
from pysyrev.core.clean import clean_doi, clean_abstracts
from typing import Iterable

import pandas as pd


class BibDataset:

    _db = None
    _bib_dataset = None
    # _pbx_probe = None

    def __init__(self, bibfile=None, bib_dataset=None):
        """

        Parameters
        ----------
        bibfile: pandas.DataFrame OR str
            path to bib file (.csv, .bib, etc.)
            or Pandas DataFrame
        bib_dataset: pandas.DataFrame
            Already processed bib dataset
        """
        try:
            self.generate_bib(bibfile)
        except TypeError:
            self._bib_dataset = check_bib_dataset(bib_dataset)


    def clean_and_drop(self,
                       min_signals_to_reject: int=2,
                       extra_garbage_phrases: Iterable[str] = (),
                       use_langdetect: bool = False,
                       ):
        """ Clean DOI and abstract columns, drop no-abstract rows

        Parameters
        ----------
        min_signals_to_reject
        extra_garbage_phrases
        use_langdetect

        Returns
        -------

        """
        # CLEAN
        self._bib_dataset.doi = self._bib_dataset.doi.apply(clean_doi)
        self._bib_dataset.abstract = clean_abstracts(self._bib_dataset.abstract,
                                                     min_signals_to_reject,
                                                     extra_garbage_phrases,
                                                     use_langdetect)

        # Does it have abstract ? DROP no-abstract rows
        self._bib_dataset.drop(self._bib_dataset.index[pd.isna(self._bib_dataset.abstract)],
                               axis=0,
                               inplace=True)

        # Reset index and drop index col
        self._bib_dataset.reset_index(inplace=True,
                                      drop=True)

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
        year: int or float
            min publication year
        nb_citations: int or float
            min citation count
        language: str or list[str]
        scorer: function
        score_cutoff: int

        Returns
        -------

        """
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

    def generate_bib(self, bibfile, del_duplicated=True, verbose=True):
        """ Generate bib using a custom version of pbx_probe

        Parameters
        ----------
        bibfile: str or bytes or os.PathLike
        del_duplicated: bool
        verbose:
            print command outputs

        Returns
        -------

        """
        self._bib_dataset = generate_bib(bibfile,
                                         db = self._db,
                                         del_duplicated=del_duplicated,
                                         print_log=verbose)
        # self._bib_dataset = self._pbx_probe.data

        return self

    def merge(self,
              others,
              title_similarity: int = 98,
              ngram_size: int = 3,
              max_candidates_per_row: int = 200,
              scorer = fuzz.token_set_ratio):
        """ Merge dataset with other(s) and remove duplicates

        Parameters
        ----------
        others: List[BibDataset]
        title_similarity: int
            FuzzyWuzzy similarity threshold
        ngram_size : int
            Word n-gram size for the blocking index. Larger = fewer but stricter
            candidates (3 is a reasonable choice for scientific titles).
        max_candidates_per_row : int
            Upper bound on the shortlist size per query. Prevents pathological
            cases where very common n-grams pull in thousands of candidates.
        scorer : callable
            rapidfuzz scorer used to compare shortlisted candidates
            (e.g. ``rapidfuzz.fuzz.token_set_ratio`` or ``fuzz.WRatio``).

        Returns
        -------

        """
        datasets = [self._bib_dataset] + [other.dataset for other in others]

        return self.__class__(bib_dataset=merge_bibs(datasets,
                                                     title_similarity_threshold=title_similarity,
                                                     ngram_size=ngram_size,
                                                     max_candidates_per_row=max_candidates_per_row,
                                                     scorer=scorer))

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

    def to_csv(self,
               file_name,
               sep=",",
               index=False):
        """ Write bib to csv file

        Parameters
        ----------
        file_name
        sep
        index
            Write row names

        Returns
        -------

        """
        self.dataset.to_csv(file_name,
                            sep=sep,
                            index=index)

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