"""
"How to read this section" boxes — one at the top of each report section.

Each says what the section's results are and how they were computed, in a few
sentences, and follows the settings the section was built with so it never
describes a computation that did not run.
"""

_TITLE = "How to read this section"


def _box(*sentences) -> dict:
    return {"type": "callout", "title": _TITLE,
            "text": " ".join(s for s in sentences if s)}


def topics_guide(ranked: bool, labelled: bool) -> dict:
    repr_docs = (
        "<br/><b>Representative documents</b> are those whose wording is closest to "
        "the topic's: the cosine similarity (<b>Sim.</b>) between the document's "
        "c-TF-IDF vector and the topic's — BERTopic's own measure, computed here "
        "on every document of the topic, most typical first. It is a lexical "
        "measure: it says nothing about citations or influence, which the paper "
        "selection section weighs."
        if ranked else
        "<br/><b>Representative documents</b> are the few the topic model kept: "
        "those whose c-TF-IDF vector is closest to the topic's, picked by "
        "BERTopic from a sample of the topic's documents and listed unordered."
    )
    return _box(
        "Topics are found by BERTopic: each document's title and abstract is "
        "embedded by a language model, the embeddings are reduced with UMAP and "
        "clustered with HDBSCAN. Documents that fit no cluster are outliers and "
        "are left out of every topic table.",
        "<b>Keywords</b> are the topic's top n-grams by c-TF-IDF — frequent in "
        "this topic, rare in the others.",
        ("<b>Labels</b> are written by an LLM from those keywords and a few "
         "representative documents." if labelled else ""),
        repr_docs,
    )


def networks_guide(cfg) -> dict:
    return _box(
        "Two views of the corpus' reference lists.",
        "<br/>• <b>Bibliographic coupling</b> links two corpus documents that cite the "
        "same works — strength = shared references / √(references of each). It "
        "groups papers that build on the same basis today: research fronts.",
        "<br/>• <b>Co-citation</b> links two cited works that corpus documents cite "
        f"together (works cited by at least {cfg.cocitation.min_ref_freq} "
        "documents) — strength = co-citing documents / √(citations of each). It "
        "reveals the intellectual base the field stands on.",
        "<br/>In both networks, communities are found with the Leiden algorithm "
        "(resolutions swept, the partition with the best modularity kept); "
        "communities too small to report form an unassigned tail. A node's "
        "<b>strength</b> is the sum of its link weights. The layout is "
        "force-directed on the same weights, and only each node's "
        "k strongest links are drawn (backbone).",
        "<br/><b>Community sub-themes</b> are the TF-IDF terms that set each "
        "community apart. The <b>community × topic mapping</b> crosses coupling "
        "communities with BERTopic topics, and the <b>connectivity</b> panels "
        "score how strongly topics and communities are coupled to each other, "
        "against what chance would give.",
    )


def actors_guide(cfg) -> dict:
    sort_text = {
        "papers": "the number of papers they sign (full counting: a paper counts "
                  "once for each of its actors)",
        "fractional": "fractional counting (each paper is worth 1, split equally "
                      "among its actors of the same level)",
        "citations": "the citations received by the papers they sign",
    }[cfg.sort_by]
    net = cfg.network
    network_text = (
        "In each <b>collaboration network</b>, two actors are linked when they "
        "sign papers together; the link strength is normalised "
        f"({net.normalize}) for the layout and the Leiden communities, so that "
        "big producers do not attract every link. Only actors with at least "
        f"{net.min_papers} papers are drawn (the {net.max_nodes} most productive "
        f"at most); node area follows {net.size_by}, edge width the number of "
        "co-signed papers."
        if net.enabled else "")
    return _box(
        f"Rankings order actors by {sort_text}; the table also gives the other "
        "counts, the share of the papers they sign and their active years. A "
        "country or institution is credited once per paper, however many of its "
        "authors sign it.",
        network_text,
    )


def temporal_guide() -> dict:
    return _box(
        "How each topic's output evolves over the years. The per-year curves "
        "count the documents a topic publishes each year; a rolling mean "
        "smooths them, and the share of the year shows a topic's weight in the "
        "field's output regardless of the field's overall growth.",
        "The cumulative views add the years up: in absolute numbers, as a share "
        "of the topic's own documents (when each topic took off), and weighted "
        "by BERTopic's topic probabilities (a document counts towards every "
        "topic it partly belongs to).",
    )


def topic_characteristics_guide() -> dict:
    return _box(
        "The size and the citation impact of each topic. Citations are raw "
        "counts from the bibliographic sources: they favour older papers, which "
        "had more time to be cited. Mean citations correct for the topic's "
        "size; the sum over a topic's most cited documents shows whether its "
        "impact rests on a few landmark papers; and the topic distribution of "
        "the corpus' most cited documents shows which topics hold the field's "
        "most visible work.",
    )


def topic_similarity_guide() -> dict:
    return _box(
        "How close the topics are in content. Every document carries a "
        "probability for each topic (BERTopic's approximate topic "
        "distribution); two topics are similar when they claim the same "
        "documents — the cosine similarity of their probability columns. A "
        "dendrogram, when shown, groups the topics by Ward hierarchical "
        "clustering and sets the order of the heatmap.",
    )


def paper_selection_guide(cfg, composite_used: bool) -> dict:
    budget = (
        f"A reading budget of {cfg.budget} papers is shared among the topics "
        f"({'by the square root of' if cfg.allocation == 'sqrt' else 'in proportion to'} "
        f"their number of eligible papers, at least {cfg.min_per_topic} each)."
        if cfg.budget else
        f"Each topic contributes {cfg.proportion_per_topic:.0%} of its papers.")
    scope = f"Only papers published from {cfg.min_year} on are eligible. {budget}"

    if composite_used:
        comp = cfg.composite
        relevance = ("citations per year" if comp.relevance_mode == "cpy" else
                     "the mean of citations per year and raw citations")
        aggregate = {
            "mean": "their (weighted) mean — a weak axis can be made up by "
                    "strong ones",
            "gmean": "their geometric mean — a weak axis drags the score down",
            "chebyshev": "their minimum (Chebyshev) — a paper scores only as "
                         "high as its weakest axis, so all-round papers win",
        }[comp.aggregate]
        weights = (f" Axes are weighted {list(comp.weights)} (centrality, "
                   "typicality, relevance)." if comp.weights else "")
        ranking = (
            "Papers are ranked by a <b>three-axis score</b>, each axis a "
            "percentile (0–1) within the paper's bibliographic-coupling "
            "community, so papers are compared with their own sub-literature:",
            "<br/>• <b>Centrality (Cent.)</b> — the paper's position in the coupling "
            "network: mean of the percentiles of its PageRank and of its "
            "weighted degree. High = it shares many references with the rest "
            "of its community.",
            f"<br/>• <b>Relevance (Rel.)</b> — citation impact: {relevance}, counted "
            "over complete years only, so a recent paper is not penalised for "
            "the year still running. Papers of the current year have no "
            "complete year: their relevance is n.d. and they are scored on the "
            "two other axes.",
            "<br/>• <b>Typicality (Typ.)</b> — how prototypical the paper's wording is "
            "of its community: the cosine similarity between its TF-IDF vector "
            "and the community's mean vector.",
            f"<br/>The <b>Score</b> combines the three axes by {aggregate}.{weights} "
            "A paper outside the coupling network (no reference shared with the "
            "corpus) has no score and comes last.",
        )
    else:
        ranking = ({
            "citations": "Papers are ranked by their raw citation count.",
            "random": "Papers are drawn at random (a baseline to compare "
                      "the other rankings with).",
            "coupling": "Papers are ranked by their strength in the "
                        "bibliographic-coupling network: how much of their "
                        "references they share with the rest of the corpus.",
            "co_citation": "Papers are ranked by their strength in the "
                           "co-citation network: how often the corpus cites "
                           "them together with other works.",
            "composite": "The three-axis score needs the coupling network, "
                         "which could not be built: papers are ranked by their "
                         "raw citation count instead.",
        }[cfg.selection_by],)

    reviews = (
        "<br/><b>Reviews</b> are added beside the ranking, detected by their "
        "document type and/or title and ranked on citations alone; the most "
        "recent ones keep a guaranteed place, since they have had no time to be "
        "cited. The <b>Selection</b> column says why each paper is listed."
        if cfg.review.enabled else "")
    return _box(
        "A reading list: the papers of each topic most worth reading first.",
        scope, *ranking, reviews,
    )


def overview_guide() -> dict:
    return _box(
        "The topic model is chosen among many fits (UMAP and HDBSCAN parameter "
        "grid). Each fit is scored on four criteria — <b>entropy</b> (how evenly "
        "documents spread over the topics), <b>coherence</b> (how often a "
        "topic's top words occur together in the documents), the number of "
        "documents assigned to a topic (the rest are <b>outliers</b>) and "
        "<b>diversity</b> (the share of distinct words among the topics' top "
        "words). The fits are ranked by their distance to the ideal on all "
        "four, and the selected one is described below. <b>Purity</b> = "
        "coherence × diversity. <b>HDBSCAN parameters</b> read "
        "min_topic_size_min_samples: the smallest number of documents a topic "
        "may hold, then how conservative the clustering is.",
    )
