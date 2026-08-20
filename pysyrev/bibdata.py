import warnings
from functools import partial

from rapidfuzz import fuzz

from pysyrev.core.api import OpenAlexClient, WosClient
from pysyrev.core.bib import (fetch_citations, generate_bib, generate_oa_bib,
                               extract_documents, check_bib_dataset)
from pysyrev.core.mappers import from_openalex_result, from_wos_result
from pysyrev.core.references import (resolve_references as _resolve_references,
                                     flag_shared_unresolved_references as _flag_unresolved,
                                     complete_reference_keys as _complete_reference_keys,
                                     resolve_ids_via_openalex as _resolve_ids_via_openalex)
from pysyrev.core.config import BibConfig, OpenAlexSourceConfig, WosSourceConfig
from pysyrev.core.merge_bibs import merge_bibs
from pysyrev.core.clean import clean_doi, clean_abstracts
from pysyrev.core.completion import complete_abstracts_from_wos
from typing import Iterable, List

import pandas as pd

_SCORER_MAP = {
    "partial_token_sort_ratio": fuzz.partial_token_sort_ratio,
    "token_set_ratio":          fuzz.token_set_ratio,
    "partial_ratio":            fuzz.partial_ratio,
    "WRatio":                   fuzz.WRatio,
    "ratio":                    fuzz.ratio,
}


class BibDataset:

    _db = None
    _bib_dataset = None
    _cross_id_map: dict  # {dropped_id: kept_id} built during merge

    # Completion strategies available for this source, keyed by the config
    # sub-block (attribute on the source config) that enables them. Each value
    # is ``(provider_label, factory)`` where ``factory(cfg)`` turns that config
    # block into a ``(dataframe, show_progress) -> int`` completer. Empty on the
    # base class; subclasses declare the providers they support.
    _COMPLETERS: dict = {}

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
        self._cross_id_map = {}
        if bibfile is not None:
            self.generate_bib(bibfile)
        elif bib_dataset is not None:
            self._bib_dataset = check_bib_dataset(bib_dataset)
        else:
            raise ValueError("Either bibfile or bib_dataset must be provided.")


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

    def extract_documents(self, include_document_type=None, year=1900,
                          nb_citations=0,
                          language="english",
                          scorer=fuzz.partial_token_sort_ratio,
                          score_cutoff=90,
                          exclude_document_type=None):
        """ Create sub bib dataset through metadata selection

        Parameters
        ----------
        include_document_type: str or list[str] or None
            document types to include (fuzzy-matched); None keeps all
        year: int or float
            min publication year
        nb_citations: int or float
            min citation count
        language: str or list[str] or None
            None means no language filter (keep all)
        scorer: callable
        score_cutoff: int
        exclude_document_type: str or list[str] or None
            document types to exclude (fuzzy-matched); takes priority over inclusion

        Returns
        -------

        """
        return self._propagate_to(
            self.__class__(bib_dataset=extract_documents(self._bib_dataset,
                                                         include_document_type,
                                                         language,
                                                         year,
                                                         nb_citations,
                                                         scorer,
                                                         score_cutoff,
                                                         exclude_doc_type=exclude_document_type))
        )

    def flag_shared_unresolved_references(self):
        """Add a 'shared_unresolved_references' column.

        For each document, the column contains the unresolved references
        that appear in at least one other document in the dataset — useful
        as edges for co-citation network analysis on unresolved refs.
        Requires resolve_references() to have been called first.
        """
        self._bib_dataset = _flag_unresolved(self._bib_dataset)

        return self

    def fetch_abstracts(self):
        """ Use online APIs to retrieve abstracts

        Returns
        -------

        """
        #TODO
        pass

    def complete_abstracts(self, completer, provider="completion", verbose=True):
        """Fill missing abstracts using a completion strategy.

        Generic over the provider: *completer* is any callable
        ``(dataframe, show_progress) -> int`` that fills the ``abstract`` column
        of records left bare by the primary source and returns how many it
        recovered (e.g. ``complete_abstracts_from_wos`` with its provider
        params bound). No other column should be modified — the source's
        references/IDs used for coupling networks must be preserved. Must be
        called before :meth:`clean_and_drop`, which drops no-abstract rows.

        Parameters
        ----------
        completer : callable
            Completion strategy, ``(dataframe, show_progress) -> int``.
        provider : str
            Human-readable provider name, used only in the log line.
        verbose : bool
            Show progress and print how many abstracts were recovered.

        Returns
        -------
        self
        """
        recovered = completer(self._bib_dataset, show_progress=verbose)
        if verbose:
            print(f"{provider} completion: recovered {recovered} missing abstract(s)")
        return self

    def apply_completions(self, source_config, verbose=True):
        """Apply every completion sub-block present on *source_config*.

        Iterates this class's ``_COMPLETERS`` registry: for each declared
        provider whose config sub-block is set, build its completer and run it.
        Absent sub-blocks are skipped. Returns self so it chains after
        :meth:`from_config`.
        """
        for attr, (provider, factory) in self._COMPLETERS.items():
            cfg = getattr(source_config, attr, None)
            if cfg:
                self.complete_abstracts(factory(cfg), provider=provider, verbose=verbose)
        return self

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

        merged_df, cross_id_map = merge_bibs(datasets,
                                             title_similarity_threshold=title_similarity,
                                             ngram_size=ngram_size,
                                             max_candidates_per_row=max_candidates_per_row,
                                             scorer=scorer)
        instance = self.__class__(bib_dataset=merged_df)
        instance._cross_id_map = cross_id_map
        return instance

    def resolve_references(self,
                           fuzzy_score_cutoff: int = 90,
                           ngram_size: int = 3,
                           max_candidates: int = 50,
                           scorer=fuzz.token_set_ratio):
        """Resolve raw references to internal document IDs.

        Adds two columns to the dataset:

        ``reference_ids``
            Internal doc IDs of resolved references ('; '-joined), or None.
        ``unresolved_references``
            Raw reference strings that found no match ('; '-joined), or None.

        Parameters
        ----------
        fuzzy_score_cutoff : int
            Minimum rapidfuzz score (0-100) to accept a fuzzy title match.
            Pass 100 to disable fuzzy matching entirely.
        ngram_size : int
            Word n-gram size for the blocking index.
        max_candidates : int
            Maximum candidates per query in the blocking phase.
        scorer : callable
            rapidfuzz scorer for fuzzy title comparison.

        """
        self._bib_dataset = _resolve_references(
            self._bib_dataset,
            cross_id_map=self._cross_id_map,
            fuzzy_score_cutoff=fuzzy_score_cutoff,
            ngram_size=ngram_size,
            max_candidates=max_candidates,
            scorer=scorer,
        )
        return self

    def complete_reference_keys(self, cache_path=None, resolvers=None,
                                resolve_external=True):
        """Add a ``reference_keys`` column: references remapped to canonical
        DOI keys (a source-agnostic coupling basis).

        See :func:`pysyrev.core.references.complete_reference_keys`. *resolvers*
        maps an id-scheme name to an ``ids -> {id: doi}`` callable. Mutates the
        underlying dataset in place and returns ``self``.
        """
        self._bib_dataset = _complete_reference_keys(
            self._bib_dataset,
            cache_path=cache_path,
            resolvers=resolvers,
            resolve_external=resolve_external,
        )
        return self

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
        return self._propagate_to(
            self.__class__(bib_dataset=self._bib_dataset.sample(n=size,
                                                                random_state=random_state,
                                                                axis=0,
                                                                ignore_index=True))
        )

    def to_csv(self,
               file_name,
               sep=",",
               index=False):
        """ Write bib to csv file

        Parameters
        ----------
        file_name: str
        sep: str
        index: bool
            Write row names

        Returns
        -------

        """
        self.dataset.to_csv(file_name,
                            sep=sep,
                            index=index)

    # ---- Protected methods -------------------------------------------------
    def _propagate_to(self, instance):
        instance._cross_id_map = self._cross_id_map
        return instance

    # ---- bridge from configuration -----------------------------------------

    @classmethod
    def from_config(cls, config: BibConfig) -> 'BibDataset':
        """Build a BibDataset from all sources declared in a BibConfig.

        Pipeline: load sources → merge → clean → extract (if include_doc_type set)
        → resolve references (if enabled).  All parameters are driven by the
        config; see CleanConfig, ExtractConfig, MergeConfig, and
        ResolveReferencesConfig for defaults.
        """
        by_source: dict = {}

        if config.open_alex:
            # Completion sub-blocks (e.g. wos_completion) enrich the OpenAlex
            # records before merge/clean, so abstracts OpenAlex left bare are
            # recovered rather than dropped as no-abstract rows. Only the
            # abstract is touched; OpenAlex IDs are preserved. Which providers
            # run is declared in OpenAlexDataset._COMPLETERS.
            open_alex = OpenAlexDataset.from_config(config.open_alex)
            open_alex.apply_completions(config.open_alex)
            by_source['open_alex'] = open_alex
        if config.wos:
            by_source['wos'] = WosDataset.from_config(config.wos)
        if config.scopus:
            by_source['scopus'] = ScopusDataset(bibfile=config.scopus)
        if config.pubmed:
            by_source['pubmed'] = PubmedDataset(bibfile=config.pubmed)

        if not by_source:
            raise ValueError("No bib source is configured — set at least one of "
                             "wos, open_alex, scopus, or pubmed in the config.")

        cfg_merge = config.merge

        # Order the loaded sources by the configured merge priority. The first
        # one becomes the merge's primary dataset (it wins duplicates and keeps
        # its IDs); sources not listed in `priority` keep their declaration order
        # after the listed ones. See MergeConfig.priority.
        ordered = [name for name in cfg_merge.priority if name in by_source]
        ordered += [name for name in by_source if name not in ordered]
        datasets: List[BibDataset] = [by_source[name] for name in ordered]
        merged = (
            datasets[0] if len(datasets) == 1
            else datasets[0].merge(
                datasets[1:],
                title_similarity      = cfg_merge.title_similarity,
                ngram_size            = cfg_merge.ngram_size,
                max_candidates_per_row= cfg_merge.max_candidates_per_row,
                scorer                = _SCORER_MAP[cfg_merge.scorer],
            )
        )

        cfg_clean = config.clean
        merged = merged.clean_and_drop(
            min_signals_to_reject = cfg_clean.min_signals_to_reject,
            extra_garbage_phrases = cfg_clean.extra_garbage_phrases or (),
            use_langdetect        = cfg_clean.use_langdetect,
        )

        # Canonical reference keys: remap every reference onto a shared DOI space
        # so bibliographic coupling/co-citation hold across the merged sources,
        # independent of merge order. Intra-corpus and DOI-bearing references map
        # for free; extra-corpus opaque ids are resolved per id scheme by the
        # resolvers built below (OpenAlex today) and cached to disk.
        cfg_rk = config.reference_keys
        if cfg_rk is not None:
            resolvers = cls._reference_key_resolvers(config) if cfg_rk.resolve_external else {}
            merged = merged.complete_reference_keys(
                cache_path=cfg_rk.cache, resolvers=resolvers,
                resolve_external=cfg_rk.resolve_external)

        # resolve_references runs before extract_documents so that:
        #   * references are resolved against the full
        #     cleaned dataset, maximizing the number of
        #     resolvable targets.
        cfg_rr = config.resolve_references
        if cfg_rr.enabled:
            api_sources = [
                name for name, src in (('wos', config.wos), ('open_alex', config.open_alex))
                if isinstance(src, (WosSourceConfig, OpenAlexSourceConfig))
                and src.source == 'api'
            ]
            if api_sources:
                warnings.warn(
                    f"resolve_references is enabled but the following sources use "
                    f"the API ({', '.join(api_sources)}), which does not return "
                    f"references inline. The 'references' column will be empty for "
                    f"those records and resolution will produce no matches. "
                    f"Switch to source: file to get references.",
                    UserWarning,
                    stacklevel=2,
                )
            merged = merged.resolve_references(
                fuzzy_score_cutoff = cfg_rr.fuzzy_score_cutoff,
                ngram_size         = cfg_rr.ngram_size,
                max_candidates     = cfg_rr.max_candidates,
                scorer             = _SCORER_MAP[cfg_rr.scorer],
            )

            if cfg_rr.flag_unresolved:
                merged = merged.flag_shared_unresolved_references()

        cfg_extract = config.extract
        merged = merged.extract_documents(
            cfg_extract.include_doc_type,
            year                  = cfg_extract.year,
            nb_citations          = cfg_extract.nb_citations,
            language              = cfg_extract.language,
            scorer                = _SCORER_MAP[cfg_extract.scorer],
            score_cutoff          = cfg_extract.score_cutoff,
            exclude_document_type = cfg_extract.exclude_doc_type,
        )

        if config.export:
            config.export.resolve()
            merged.to_csv(config.export.dataset)

        return merged

    @staticmethod
    def _reference_key_resolvers(config: BibConfig) -> dict:
        """Build the ``{id_scheme: ids -> {id: doi}}`` resolvers for
        :meth:`complete_reference_keys` from the configured sources.

        Each id scheme that can resolve extra-corpus ids to DOIs registers here
        from its own credentials. OpenAlex is the only scheme implemented today;
        further sources (Scopus, PubMed…) would add their resolver the same way.
        A scheme whose credentials are missing is simply absent — those ids then
        keep their native token.
        """
        resolvers: dict = {}
        oa = config.open_alex
        if isinstance(oa, OpenAlexSourceConfig) and oa.api is not None:
            client = OpenAlexClient(api_key=oa.api.api_key, email=oa.api.email)
            resolvers['openalex'] = partial(_resolve_ids_via_openalex, client=client)
        return resolvers

    @classmethod
    def _from_source_config(cls, config) -> 'BibDataset':
        """Template: branch on source type and return a new instance.

        * ``source: file`` — delegates to the regular constructor.
        * ``source: api``  — calls :meth:`_from_api_config` (subclass hook)
          to produce a DEFAULT_FIELDS DataFrame, then wraps it in the constructor.
        """
        if config.source == 'file':
            return cls(bibfile=config.file)
        return cls(bib_dataset=cls._from_api_config(config.api))

    @classmethod
    def _from_api_config(cls, api_config) -> pd.DataFrame:
        """Hook: query the source API and return a DEFAULT_FIELDS DataFrame.

        Must be overridden by subclasses that support ``source: api``.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not support API source. "
            "Override _from_api_config or set `source: file` in the config."
        )

    # ---- properties --------------------------------------------------------

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

    @classmethod
    def from_config(cls, config: WosSourceConfig) -> 'WosDataset':
        return cls._from_source_config(config)

    @classmethod
    def _from_api_config(cls, api_config) -> pd.DataFrame:
        client = WosClient(
            api_key      = api_config.api_key,
            session_file = (f"{api_config.cache_dir}/session.json"
                            if api_config.cache_dir else None),
        )
        return from_wos_result(client.search(query=api_config.query))


class OpenAlexDataset(BibDataset):

    _db = "scopus"

    # Abstract-completion providers available for OpenAlex-sourced records:
    # config sub-block -> (label, factory building a (df, show_progress) -> int
    # completer). Add a provider here to enable it — from_config needs no change.
    _COMPLETERS = {
        "wos_completion": ("WoS", lambda cfg: partial(
            complete_abstracts_from_wos,
            api_key=cfg.api_key,
            session_file=f"{cfg.cache_dir}/session.json" if cfg.cache_dir else None,
        )),
    }

    @classmethod
    def from_config(cls, config: OpenAlexSourceConfig) -> 'OpenAlexDataset':
        return cls._from_source_config(config)

    @classmethod
    def _from_api_config(cls, api_config) -> pd.DataFrame:
        client = OpenAlexClient(
            api_key      = api_config.api_key,
            email        = api_config.email,
            session_file = (f"{api_config.cache_dir}/session.json"
                            if api_config.cache_dir else None),
        )
        return from_openalex_result(
            client.search(query=api_config.query, filters=api_config.filters)
        )

    def generate_bib(self, bibfile, **kwargs):

        new_bib_file = generate_oa_bib(bibfile)

        super().generate_bib(new_bib_file, **kwargs)


class ScopusDataset(BibDataset):

    _db = "scopus"

    pass


class PubmedDataset(BibDataset):

    _db = "pubmed"

    pass