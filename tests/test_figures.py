"""Tests for the shared matrix-heatmap layout (pysyrev.core.report.figures)."""

import numpy as np
import pytest

from pysyrev.core.report.figures import (matrix_heatmap, plot_connectivity_matrix,
                                  plot_crosstab_heatmap, plot_similarity_matrix)


def _trace(fig):
    return fig.data[0]


# =============================================================================
# Generic layout
# =============================================================================

class TestMatrixHeatmap:

    def test_non_finite_cells_are_left_unannotated(self):
        z = np.array([[np.nan, 0.5], [0.5, np.inf]])
        fig = matrix_heatmap(z, ["a", "b"], ["a", "b"], text_format=lambda v: f"{v:.2f}")
        assert _trace(fig).text == (["", "0.50"], ["0.50", ""])

    def test_no_annotations_without_a_formatter(self):
        fig = matrix_heatmap(np.eye(2), ["a", "b"], ["a", "b"])
        assert _trace(fig).text is None

    def test_size_follows_the_matrix_shape(self):
        wide = matrix_heatmap(np.zeros((2, 6)), list("abcdef"), list("ab"))
        tall = matrix_heatmap(np.zeros((6, 2)), list("ab"), list("abcdef"))
        assert wide.layout.width > tall.layout.width
        assert tall.layout.height > wide.layout.height

    def test_size_floors_keep_a_tiny_matrix_readable(self):
        fig = matrix_heatmap(np.zeros((1, 1)), ["a"], ["a"])
        assert (fig.layout.width, fig.layout.height) == (460, 420)

    def test_reads_like_a_table(self):
        # Columns labelled on top, rows running top-to-bottom.
        fig = matrix_heatmap(np.eye(2), ["a", "b"], ["a", "b"])
        assert fig.layout.xaxis.side == "top"
        assert fig.layout.yaxis.autorange == "reversed"

    def test_upper_bound_is_left_to_the_data(self):
        fig = matrix_heatmap(np.full((2, 2), 0.2), ["a", "b"], ["a", "b"])
        assert _trace(fig).zmin == 0.0 and _trace(fig).zmax is None


# =============================================================================
# Every panel shares the house style
# =============================================================================

class TestHouseStyle:

    @pytest.fixture
    def figures(self):
        M = np.array([[1.0, 0.25], [0.25, 2.0]])
        S = np.array([[np.nan, 0.4], [0.4, np.nan]])
        C = np.array([[5.0, 0.0], [1.0, 4.0]])
        return {
            "connectivity": plot_connectivity_matrix(M, ["T0", "T1"], baseline=0.3),
            "similarity":   plot_similarity_matrix(S, ["T0", "T1"]),
            "crosstab":     plot_crosstab_heatmap(C, ["C0", "C1"], ["T0", "T1"]),
        }

    def test_same_colour_scale(self, figures):
        scales = {name: _trace(fig).colorscale for name, fig in figures.items()}
        assert len(set(scales.values())) == 1

    def test_same_axis_convention(self, figures):
        for fig in figures.values():
            assert fig.layout.xaxis.side == "top"
            assert fig.layout.xaxis.tickangle == -30
            assert fig.layout.yaxis.autorange == "reversed"
            assert fig.layout.plot_bgcolor == "white"

    def test_every_cell_is_annotated(self, figures):
        for name, fig in figures.items():
            assert _trace(fig).text is not None, name

    def test_each_panel_keeps_its_own_units(self, figures):
        titles = {name: _trace(fig).colorbar.title.text for name, fig in figures.items()}
        assert titles == {"connectivity": "mean coupling",
                          "similarity": "cosine",
                          "crosstab": "docs"}


# =============================================================================
# Panel-specific behaviour
# =============================================================================

class TestPanels:

    def test_connectivity_reports_the_baseline_in_its_title(self):
        fig = plot_connectivity_matrix(np.eye(2), ["a", "b"], baseline=0.314)
        assert "0.31" in fig.layout.title.text

    def test_connectivity_without_baseline_has_no_title(self):
        fig = plot_connectivity_matrix(np.eye(2), ["a", "b"])
        assert fig.layout.title.text is None

    def test_similarity_blanks_the_diagonal(self):
        S = np.array([[np.nan, 0.4], [0.4, np.nan]])
        fig = plot_similarity_matrix(S, ["T0", "T1"])
        assert _trace(fig).text == (["", "0.40"], ["0.40", ""])

    def test_crosstab_annotates_counts_as_integers_and_blanks_zeros(self):
        fig = plot_crosstab_heatmap(np.array([[5.0, 0.0]]), ["C0"], ["T0", "T1"])
        assert _trace(fig).text == (["5", ""],)

    def test_crosstab_axis_titles(self):
        fig = plot_crosstab_heatmap(np.array([[1.0]]), ["C0"], ["T0"],
                                    row_title="Community", col_title="Topic")
        assert fig.layout.yaxis.title.text == "Community"
        assert fig.layout.xaxis.title.text == "Topic"


# =============================================================================
# Backwards compatibility
# =============================================================================

def test_networks_package_still_exports_its_matrix_panels():
    from pysyrev.core.networks import (plot_connectivity_matrix as pc,
                                       plot_crosstab_heatmap as px)
    assert pc is plot_connectivity_matrix and px is plot_crosstab_heatmap
