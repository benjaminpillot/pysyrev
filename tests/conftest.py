from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysyrev.core.config import (
    BibNetworkSectionConfig,
    PaperSelectionConfig,
    ReportMetaConfig,
    ReportSectionsConfig,
    TemporalSectionConfig,
    TopicCharacteristicsConfig,
    TopicReportConfig,
    TopicsSectionConfig,
    TopicSimilarityConfig,
)

_TESTS_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Constants shared by synthetic-data fixtures
# ─────────────────────────────────────────────────────────────────────────────
_N_DOCS   = 30
_N_TOPICS = 3
_N_REPR   = 5


# =============================================================================
# PDF engine fixtures (existing)
# =============================================================================

@pytest.fixture
def minimal_report_data():
    return {
        "meta": {
            "title": "Test Report",
            "author": "Pytest",
            "date": "2026-05-18",
            "version": "1.0",
            "subtitle": "Automated test suite",
            "summary": "A minimal report generated during testing.",
        },
        "sections": [
            {
                "title": "Section 1",
                "blocks": [
                    {"type": "paragraph",     "text": "Hello world."},
                    {"type": "rich_text",     "text": "<b>Bold</b> and <i>italic</i>."},
                    {"type": "spacer"},
                    {"type": "line"},
                    {"type": "bullets",       "items": ["Bullet A", "Bullet B"]},
                    {"type": "numbered_list", "items": ["First", "Second"]},
                    {"type": "key_value",     "items": [{"key": "Key", "value": "Value"}]},
                    {"type": "callout",       "title": "Note", "text": "A test callout."},
                    {"type": "code",          "text": "x = 1 < 2\ny = x & True"},
                    {"type": "subsection",    "title": "Sub", "blocks": [
                        {"type": "paragraph", "text": "Nested paragraph."},
                    ]},
                    {"type": "page_break"},
                ],
            }
        ],
    }


@pytest.fixture
def tiny_corpus():
    """A small corpus of scientific-flavoured sentences for NLP tests."""
    return [
        "Agent-based models simulate the actions of autonomous agents.",
        "Cellular automata are discrete dynamic systems used in complex systems.",
        "Social simulation explores emergent behaviour in multi-agent systems.",
        "Epidemiological models track the spread of infectious disease in populations.",
        "Network analysis reveals the structure of social and biological systems.",
        "Complexity science studies how simple rules give rise to complex phenomena.",
        "Computational social science applies algorithms to understand human behaviour.",
        "Game theory analyses strategic interactions between rational decision makers.",
        "System dynamics models feedback loops and time delays in complex systems.",
        "Individual-based models capture heterogeneity across agents in a population.",
        "Evolutionary algorithms use selection and mutation to optimise solutions.",
    ]


# =============================================================================
# Synthetic topic-model data fixtures
# =============================================================================

@pytest.fixture
def tiny_bertopic_results():
    """Synthetic bertopic_results DataFrame (~30 docs, 3 topics, all metadata cols)."""
    rng   = np.random.default_rng(42)
    probs = rng.dirichlet(np.ones(_N_TOPICS), size=_N_DOCS)
    topics = np.argmax(probs, axis=1).tolist()
    topics[0] = topics[1] = -1          # 2 outliers

    # Fixed years so year-filter tests are deterministic
    years = [2010 + (i % 14) for i in range(_N_DOCS)]   # 2010–2023
    cited_by = [float(20 * (i % 15 + 1)) for i in range(_N_DOCS)]

    doc_types = ["review" if i % 6 == 0 else "article" for i in range(_N_DOCS)]
    titles    = [
        f"Review of topic modelling approaches {i}" if i % 8 == 0
        else f"Study on complex adaptive systems {i}"
        for i in range(_N_DOCS)
    ]

    return pd.DataFrame({
        "Document": [f"preprocessed doc {i}" for i in range(_N_DOCS)],
        "ID":       range(_N_DOCS),
        "Topic":    topics,
        "year":     years,
        "cited_by": cited_by,
        "document_type": doc_types,
        "title":    titles,
        "doi":      [f"10.1000/test.{i:04d}" for i in range(_N_DOCS)],
        **{f"topic#{j}": probs[:, j] for j in range(_N_TOPICS)},
    })


@pytest.fixture
def tiny_topic_info():
    """Synthetic topic_info DataFrame (3 topics, repr_doc columns as CSV-loaded strings)."""
    rows = []
    for tid in range(_N_TOPICS):
        rows.append({
            "Topic":       tid,
            "Name":        f"topic_{tid}_name",
            "Count":       10,
            "Representation": str([f"keyword_{tid}_{k}" for k in range(5)]),
            # repr_doc columns stored as stringified lists (mimics CSV round-trip)
            "repr_doc_title":    str([f"Representative paper {tid}-{i}" for i in range(_N_REPR)]),
            "repr_doc_year":     str([float(2018 + i) for i in range(_N_REPR)]),
            "repr_doc_cited_by": str([float(50 * (i + 1)) for i in range(_N_REPR)]),
            "repr_doc_document_type": str(
                ["review" if i % 2 == 0 else "article" for i in range(_N_REPR)]
            ),
            "repr_doc_doi":     str([f"10.2000/{tid}.{i}" for i in range(_N_REPR)]),
            "repr_doc_abstract": str([f"Abstract for topic {tid} paper {i}." for i in range(_N_REPR)]),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_best_results():
    """Synthetic best_results DataFrame (one selected model)."""
    return pd.DataFrame({
        "hdbscan":              ["2_2"],
        "umap":                 ["5_3"],
        "nb_topics":            [_N_TOPICS],
        "entropy":              [1.2],
        "diversity":            [0.85],
        "nb_outliers":          [2],
        "purity":               [0.72],
        "distance":             [0.28],
        "distance_with_purity": [0.24],
        "u_mass":               [-0.5],
    })


@pytest.fixture
def report_cfg():
    """TopicReportConfig with all sections enabled and sensible test defaults."""
    return TopicReportConfig(
        meta=ReportMetaConfig(
            title="Unit-test report",
            author="pytest",
            date_format="%Y",
            version="0.0",
        ),
        sections=ReportSectionsConfig(
            topics=TopicsSectionConfig(n_repr_docs_per_topic=3),
            bib_network=BibNetworkSectionConfig(),
            temporal=TemporalSectionConfig(
                variants=["absolute", "normalized", "weighted"]
            ),
            topic_characteristics=TopicCharacteristicsConfig(
                n_top_cited_per_topic=3, n_top_cited_global=10
            ),
            topic_similarity=TopicSimilarityConfig(clustering=True, dendrogram=True),
            paper_selection=PaperSelectionConfig(
                min_year=2015,
                proportion_per_topic=0.5,
                export_annex=False,
            ),
        ),
    )


# =============================================================================
# Integration-test fixtures (session-scoped, marked integration)
# =============================================================================

@pytest.fixture(scope="session")
def bib_dataset():
    """Full bib_dataset CSV from tests/data/ — shared across integration tests."""
    return pd.read_csv(_TESTS_DIR / "data" / "bib_dataset.csv")


@pytest.fixture
def reviewed_dataset_path():
    """Path to a reviewed-style dataset with raw references (drives the report's
    network panels, which are recomputed from `references`)."""
    return str(_TESTS_DIR / "data" / "bib_dataset.csv")


@pytest.fixture(scope="session")
def topic_model_outputs(tmp_path_factory, bib_dataset):
    """Run TopicModel with tiny params (all-MiniLM-L6-v2). Session-scoped — runs once."""
    from bertopic.vectorizers import ClassTfidfTransformer
    from pysyrev.topic_model import (
        BertopicModel, HdbscanModel, TopicDistribution, TopicModel, UmapModel,
    )

    tmp = tmp_path_factory.mktemp("topic_model")
    ctfidf       = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
    umap_model   = UmapModel(min_dist=0.0, metric="cosine", low_memory=False, random_state=42)
    hdbscan_model = HdbscanModel(
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
    )
    bertopic_model = BertopicModel(
        hdbscan_model=hdbscan_model,
        umap_model=umap_model,
        ctfidf_model=ctfidf,
        transformer_model="all-MiniLM-L6-v2",
        calculate_probabilities=True,
        n_gram_range="bigram",
        language="english",
    )
    topic_dist = TopicDistribution(window=4, stride=1, min_similarity=0.1, batch_size=100)
    model = TopicModel(
        doc_dataset=None,
        allow_abbrev=False,
        distance="euclidean",
        bertopic_model=bertopic_model,
        topic_distribution=topic_dist,
        nr_repr_docs=3,
        export_dir=str(tmp),
        n_neighbors=[5],
        n_components=[3],
        min_topic_size_range=[2, 4],
        min_sample_range=[2, 2],
        topic_size_step=1,
        min_sample_step=1,
        keep_n_results=3,
        ranking_scorer="u_mass",
        purity_scorer="c_v",
        run_name="test_run",
    )
    model.run(bib_dataset, show_progress=False)
    return {"run_dir": str(tmp / "test_run")}
