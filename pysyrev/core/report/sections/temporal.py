"""Temporal dynamics: how each topic's output evolves over the years."""

import pandas as pd

from pysyrev.core.report.sections.common import (_docs_per_topic_per_year,
                                                 _short_label, _topic_cols,
                                                 _topic_label, _wo_outliers)


def _legend_name(topic_id, topic_labels) -> str:
    """The topic as the legend names it: its id first, so a curve can be read
    back against the topic table, then as much of its label as fits."""
    label = _topic_label(topic_id, topic_labels)
    if label == f"Topic {topic_id}":
        return label
    return f"{topic_id}. {_short_label(label, max_len=34)}"


def _long_counts(counts, topic_labels):
    """A year × topic count frame melted to the long form px.line wants."""
    long = (
        counts.reset_index()
        .melt(id_vars="year", var_name="Topic", value_name="value")
    )
    long["Topic"] = long["Topic"].astype(int)
    long["Topic name"] = long["Topic"].map(
        lambda tid: _legend_name(int(tid), topic_labels)
    )
    return long.sort_values(["Topic", "year"])


def _year_tick_step(years) -> int:
    """A tick every year on a short span, every 2 or 5 on a long one."""
    span = int(max(years)) - int(min(years))
    for step in (1, 2, 5, 10):
        if span / step <= 20:
            return step
    return 10


def _trajectory_figure(counts, topic_labels, y_title, title):
    """One line per topic over the corpus years.

    Topics are told apart by colour *and* by marker shape, on a colour-blind
    safe palette: a review often has more topics than a palette has distinct
    hues, and the report is read on paper as often as on screen.
    """
    import plotly.express as px

    fig = px.line(
        _long_counts(counts, topic_labels),
        x="year", y="value", color="Topic name", symbol="Topic name",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={"year": "Year", "value": y_title},
        title=title,
    )
    fig.update_traces(line={"width": 2.5}, marker={"size": 8})
    fig.update_layout(
        width=900, height=550,
        template="plotly_white",
        font={"size": 15},
        title={"font": {"size": 18}},
        hovermode="x unified",
        legend={"title": None, "orientation": "h", "yanchor": "top",
                "y": -0.16, "xanchor": "left", "x": 0, "font": {"size": 13}},
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(dtick=_year_tick_step(counts.index))
    fig.update_yaxes(rangemode="tozero")
    return fig


def _build_trajectory_blocks(br_wo, topic_labels, smoothing_window):
    """Per-year curves: has a topic accelerated, plateaued, or fallen away?

    The cumulative charts of the other variants only ever rise, so a topic that
    has stopped publishing looks the same there as one still going: the curve
    just flattens, and a flattening is easy to miss beside topics still
    climbing. These are the year-by-year derivatives of those curves.
    """
    counts = _docs_per_topic_per_year(br_wo)
    if counts is None or counts.empty or len(counts.index) < 2:
        return []

    window = max(1, int(smoothing_window))
    blocks = [{
        "type": "paragraph",
        "text": ("Documents published each year, rather than accumulated: a "
                 "rising line is a topic speeding up, a falling one a topic "
                 "losing ground. The last year of a corpus is usually still "
                 "being indexed, so read its final point as incomplete rather "
                 "than as a collapse."),
    }]

    blocks.append({
        "type": "plotly",
        "figure": _trajectory_figure(
            counts, topic_labels, "Documents",
            "Documents per topic and per year"),
        "caption": ("Annual document count per topic. Years in which a topic "
                    "published nothing are drawn at zero, not skipped."),
        "filename_prefix": "temporal_trajectory",
    })

    if window > 1 and len(counts.index) >= window:
        smoothed = counts.rolling(window=window, center=True, min_periods=1).mean()
        blocks.append({
            "type": "plotly",
            "figure": _trajectory_figure(
                smoothed, topic_labels, "Documents (rolling mean)",
                f"Documents per topic and per year — {window}-year rolling mean"),
            "caption": (f"The same counts under a centred {window}-year rolling "
                        "mean, which trades the exact yearly value for a "
                        "readable trend."),
            "filename_prefix": "temporal_trajectory_smoothed",
        })

    # Share of the year's output: a topic can publish more every year and still
    # be losing ground, if the corpus as a whole is growing faster than it is.
    totals = counts.sum(axis=1)
    share  = counts.div(totals.where(totals > 0), axis=0) * 100.0
    if window > 1 and len(share.index) >= window:
        share = share.rolling(window=window, center=True, min_periods=1).mean()
        title_note   = f" — {window}-year rolling mean"
        caption_note = f", under a centred {window}-year rolling mean"
    else:
        title_note = caption_note = ""

    blocks.append({
        "type": "plotly",
        "figure": _trajectory_figure(
            share, topic_labels, "Share of the year (%)",
            f"Share of each year's documents per topic{title_note}"),
        "caption": (f"Each topic as a percentage of the documents published that "
                    f"year{caption_note}. It separates a topic's own momentum "
                    f"from the growth of the corpus: a flat line is a topic "
                    f"growing exactly as fast as the field, a falling one a "
                    f"topic the field is leaving behind."),
        "filename_prefix": "temporal_share",
    })
    return blocks


def _build_temporal_section(bertopic_results, nb_topics, temporal_cfg, section_n,
                            topic_labels=None):
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

    # -- trajectory (per-year, not cumulative) --
    if "trajectory" in variants:
        traj_blocks = _build_trajectory_blocks(
            br_wo, topic_labels, getattr(temporal_cfg, "smoothing_window", 3))
        if traj_blocks:
            sub_blocks.append({
                "type":   "subsection",
                "title":  "Trajectories (per year)",
                "blocks": traj_blocks,
            })

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
