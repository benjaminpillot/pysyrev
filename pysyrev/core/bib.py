import tempfile
from typing import OrderedDict

import numpy as np
import pandas as pd

from crossref.restful import Works
from semanticscholar import SemanticScholar
from thefuzz import process
from tqdm import tqdm

from pysyrev.pbx.pbx_custom import pbx_probe


FROM_OA = {"keywords.display_name": "author_keywords",
           "cited_by_count": "cited_by",
           "publication_year": "year",
           "primary_location.source.display_name": "journal",
           "type": "document_type",
           "referenced_works": "references"}

DEFAULT_FIELDS = {"id": str,
                  "abstract": str,
                  "journal": str,
                  "title": str,
                  "author_keywords": str,
                  "doi": str,
                  "language": str,
                  "document_type": str,
                  "references" : str,
                  "year": float,
                  "cited_by": float}


def check_bib_dataset(dataset):

    valid_key = all([key in dataset.columns for key, _ in DEFAULT_FIELDS.items()])

    if valid_key:
        valid_format = all([dataset[key].dtype == val for key, val in DEFAULT_FIELDS.items()])
        if valid_format:
            return dataset
        else:
            pass

    raise ValueError("Dataset is not valid")


def extract_documents(dataset, doc_type, language,
                      year, nb_citations, scorer, score_cutoff):

    dset = []
    new_dataset = dataset[(dataset.year >= year) & (dataset.cited_by >= nb_citations)]
    for d_type in doc_type:
        best_matches = process.extractBests(d_type,
                                            new_dataset.document_type,
                                            scorer=scorer,
                                            score_cutoff=score_cutoff,
                                            limit=len(new_dataset.document_type))
        dset.append(new_dataset.loc[[key for _, _, key in best_matches], :])

    return pd.concat(dset, ignore_index=True)



def fetch_citations(doi):
    # First, import the client from semanticscholar module

    works = Works()

    # You'll need an instance of the client to request data from the API
    sch = SemanticScholar()
    citation_count = []

    for doi_ in tqdm(doi, desc="Fetch citations"):
        if doi_.lower() != "unknown":
            try:
                paper = sch.get_paper(doi_)
                citation_count.append(paper.citationCount)
            except:
                try:
                    paper = works.doi(doi_)
                    citation_count.append(paper["is-referenced-by-count"])
                except:
                    citation_count.append(np.nan)
        else:
            citation_count.append(np.nan)

    return citation_count


def generate_bib(bibfile, db, del_duplicated):

    pbx = pbx_probe(file_bib=bibfile,
                    db=db,
                    del_duplicated=del_duplicated)
    columns = []
    for col, type_ in DEFAULT_FIELDS.items():
        columns.append(pbx.data[col].astype(type_))

    return pbx, pd.concat(columns, axis=1)

def generate_oa_bib(bibfile):

    dataframe = pd.read_csv(bibfile,
                            low_memory=False,
                            lineterminator='\n')
    dataframe.rename(FROM_OA,
                     inplace=True,
                     axis="columns")
    _, new_bib_file = tempfile.mkstemp(suffix=".csv", dir=tempfile.tempdir)
    dataframe.to_csv(new_bib_file, sep=",")

    return new_bib_file


def merge_bibs(datasets,
               doi_similarity_threshold,
               title_similarity_threshold):

    def compare_idx(dataset, title_score, doi_score, title_idx, doi_idx):
        if doi_score >= doi_similarity_threshold:
            return dataset.index[doi_idx]
        else:
            if title_score >= title_similarity_threshold:
                return dataset.index[title_idx]
            else:
                return np.nan

    main_ds = datasets[0]

    for n, _ in enumerate(datasets):

        if n < len(datasets) - 1:

            other_ds = datasets[n+1]

            similarity = []
            doi_similarity = []
            other_titles = {idx: title for idx, title in enumerate(other_ds.title)}
            other_doi = {idx: doi for idx, doi in enumerate(other_ds.doi)}

            for (doi, title) in zip(main_ds.doi, main_ds.title):
                try:
                    similarity.append(process.extractOne(title, other_titles))
                    doi_similarity.append(process.extractOne(doi, other_doi))
                except KeyError:
                    pass

            comparison = OrderedDict({"title_similarity": [s[1] for s in similarity],
                                      "doi_similarity": [s[1] for s in doi_similarity],
                                      "title_idx": [s[2] for s in similarity],
                                      "doi_idx": [s[2] for s in doi_similarity]})

            corr_idx = [compare_idx(other_ds, *var) for var in zip(*comparison.values())]
            other_ds = other_ds.drop([idx for idx in corr_idx if ~np.isnan(idx)])

            main_ds = pd.concat([main_ds,
                                 other_ds],
                                ignore_index=True)

    # _, new_bib_file = tempfile.mkstemp(suffix=".csv", dir=tempfile.tempdir)
    # main_ds.to_csv(new_bib_file, sep=",")

    return main_ds