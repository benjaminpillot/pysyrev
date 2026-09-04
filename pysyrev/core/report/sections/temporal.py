"""Temporal dynamics: how each topic's output evolves over the years."""

import pandas as pd

from pysyrev.core.report.sections.common import _topic_cols, _wo_outliers


def _build_temporal_section(bertopic_results, nb_topics, temporal_cfg, section_n):
    """Section — number of documents per topic over time."""
    import plotly.express as px

    if "year" not in bertopic_results.columns:
        return None

    br_wo = _wo_outliers(bertopic_results).copy()
    br_wo = br_wo.dropna(subset=["year"])
    br_wo["year"]    = br_wo["year"].astype(int)
    br_wo["nb_docs"] = 1

    variants = temporal_cfg.variants
    sub_blocks = []

    # -- absolute --
    if "absolute" in variants:
        fig = px.histogram(
            br_wo.sort_values("Topic"),
            x="year", y="nb_docs", color="Topic",
            title="Number of documents per topic over time (cumulative absolute)",
            nbins=int(br_wo["year"].max() - br_wo["year"].min() + 1),
            cumulative=True,
        )
        fig.update_layout(width=900, height=550)
        sub_blocks.append({
            "type":   "subsection",
            "title":  "Absolute",
            "blocks": [{"type": "plotly", "figure": fig,
                        "caption": "Cumulative total documents per topic per year."}],
        })

    # -- normalized --
    if "normalized" in variants:
        topic_sizes = br_wo.groupby("Topic")["nb_docs"].transform("sum")
        br_norm = br_wo.copy()
        br_norm["normalized"] = 1.0 / topic_sizes
        fig = px.histogram(
            br_norm.sort_values("Topic"),
            x="year", y="normalized", color="Topic",
            title="Proportion of each topic's documents over time (cumulative normalized)",
            nbins=int(br_wo["year"].max() - br_wo["year"].min() + 1),
            cumulative=True,
        )
        fig.update_layout(width=900, height=550)
        sub_blocks.append({
            "type":   "subsection",
            "title":  "Normalized (within-topic proportion)",
            "blocks": [{"type": "plotly", "figure": fig,
                        "caption": "Cumulative share of each topic's own documents per year."}],
        })

    # -- weighted (BERTopic approximate distribution) --
    if "weighted" in variants:
        tcols = _topic_cols(br_wo)
        if tcols:
            id_year = br_wo[["year"]].copy()
            df_long = pd.concat([id_year, br_wo[tcols]], axis=1)
            df_long = df_long.melt(id_vars=["year"], var_name="topic", value_name="weight")
            df_long["topic_num"] = df_long["topic"].str.extract(r"(\d+)").astype(int)
            df_agg = (
                df_long
                .groupby(["year", "topic_num"], as_index=False)["weight"]
                .sum()
                .rename(columns={"weight": "weighted_count"})
            )
            fig = px.histogram(
                df_agg.sort_values("topic_num"),
                x="year", y="weighted_count", color="topic_num",
                title="Weighted number of documents per topic over time (cumulative)",
                nbins=int(br_wo["year"].max() - br_wo["year"].min() + 1),
                cumulative=True,
            )
            fig.update_layout(width=900, height=550)
            sub_blocks.append({
                "type":   "subsection",
                "title":  "Weighted (BERTopic approximate distribution)",
                "blocks": [{"type": "plotly", "figure": fig,
                            "caption": "Cumulative sum of BERTopic topic-probability weights per year."}],
            })

    if not sub_blocks:
        return None
    return {"title": f"{section_n}. Temporal dynamics", "blocks": sub_blocks}
