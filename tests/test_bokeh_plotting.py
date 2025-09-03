"""
Tests for baltic_bokeh plotting functions.
"""

import pytest
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

import baltic_bokeh
from baltic_bokeh.bokeh import prepare_bokeh_data

class TestMetadataColors:
    """Test metadata color preparation."""

    def test_generate_leaf_colours_missing_column(self, sample_tree, sample_metadata):
        """Test color preparation with missing column."""

        ### this should raise assertion error
        with pytest.raises(AssertionError):
            data = prepare_bokeh_data(
                tree=sample_tree, hover_data=None, df_metadata=sample_metadata, color_column="nonexistent_column", color_discrete_map=None
            )
        

class TestplotRectangularTree:
    """Test plotRectangularTree function."""

    def test_plot_tree_basic(self, sample_tree):
        """Test basic tree plotting."""
        p = baltic_bokeh.plotRectangularTree(sample_tree, plot_width=400, plot_height=400)

        assert p is not None
        assert hasattr(p, "renderers")
        # Should have created a figure
        assert p.width == 400
        assert p.height == 400

    def test_plot_tree_custom_figure(self, sample_tree):
        """Test plotting on existing figure."""
        existing_fig = figure(width=300, height=300)
        p = baltic_bokeh.plotRectangularTree(sample_tree, p=existing_fig)

        assert p is existing_fig
        assert p.width == 300
        assert p.height == 300

    def test_plot_tree_connection_types(self, sample_tree):
        """Test different connection types."""
        for conn_type in ["baltic", "direct"]:
            p = baltic_bokeh.plotRectangularTree(sample_tree, connection_type=conn_type)
            assert p is not None

    def test_plot_tree_invalid_connection_type(self, sample_tree):
        """Test invalid connection type raises error."""
        with pytest.raises(AssertionError):
            baltic_bokeh.plotRectangularTree(sample_tree, connection_type="invalid")


class TestplotRectangularPoints:
    """Test plotRectangularPoints function."""

    def test_plot_points_basic(self, sample_tree):
        """Test basic point plotting."""
        p = baltic_bokeh.plotRectangularPoints(sample_tree, plot_width=400, plot_height=400)

        assert p is not None
        assert p.width == 400
        assert p.height == 400

    def test_plot_points_with_metadata(self, sample_tree, sample_metadata, color_map):
        """Test point plotting with metadata."""
        p = baltic_bokeh.plotRectangularPoints(
            sample_tree,
            df_metadata=sample_metadata,
            color_column="taxon",
            color_discrete_map=color_map,
            hover_data=["location", "country"],
        )

        assert p is not None
        # Should have hover tools
        hover_tools = [tool for tool in p.tools if isinstance(tool, HoverTool)]
        assert len(hover_tools) > 0

    def test_plot_points_size_and_color(self, sample_tree):
        """Test point plotting with custom size and color."""
        p = baltic_bokeh.plotRectangularPoints(sample_tree, size=12)
        assert p is not None

class TestPlotCircularTree:
    """Test plotCircularTree function."""

    def test_plot_circular_tree_basic(self, sample_tree):
        """Test basic circular tree plotting."""
        p = baltic_bokeh.plotCircularTree(sample_tree, plot_width=400, plot_height=400)

        assert p is not None
        assert p.width == 400
        assert p.height == 400
        # Circular plots should have axes hidden
        assert all(not axis.visible for axis in p.axis)
        assert not p.xgrid.visible
        assert not p.ygrid.visible

    def test_plot_circular_tree_with_params(self, sample_tree):
        """Test circular tree with custom parameters."""
        p = baltic_bokeh.plotCircularTree(
            sample_tree, circStart=0.25, circFrac=0.5, inwardSpace=0.1
        )
        assert p is not None


class TestPlotCircularPoints:
    """Test plotCircularPoints function."""

    def test_plot_circular_points_basic(self, sample_tree):
        """Test basic circular point plotting."""
        p = baltic_bokeh.plotCircularPoints(
            sample_tree, plot_width=400, plot_height=400
        )

        assert p is not None
        assert p.width == 400
        assert p.height == 400
        assert all(not axis.visible for axis in p.axis)

    def test_plot_circular_points_with_metadata(
        self, sample_tree, sample_metadata, color_map
    ):
        """Test circular points with metadata and hover."""
        p = baltic_bokeh.plotCircularPoints(
            sample_tree,
            df_metadata=sample_metadata,
            color_column="taxon",
            color_discrete_map=color_map,
            hover_data=["location", "country"],
        )

        assert p is not None
        # Should have hover tools
        hover_tools = [tool for tool in p.tools if isinstance(tool, HoverTool)]
        assert len(hover_tools) > 0


class TestAddText:
    """Test addText function."""

    def test_add_text_basic(self, sample_tree):
        """Test basic text addition."""
        p = figure(width=400, height=400)
        p_with_text = baltic_bokeh.addText(sample_tree, p)

        assert p_with_text is p
        # Should have called text method (though we can't easily verify the exact call)

    def test_add_text_custom_target_and_text(self, sample_tree):
        """Test text addition with custom target and text functions."""
        p = figure(width=400, height=400)
        p_with_text = baltic_bokeh.addText(
            sample_tree,
            p,
            target=lambda k: k.is_leaf(),
            text=lambda k: f"Node_{k.name}",
        )

        assert p_with_text is p


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_metadata(self, sample_tree):
        """Test handling of empty metadata."""
        empty_metadata = pd.DataFrame()

        with pytest.raises(AssertionError):
            baltic_bokeh.plotRectangularPoints(
                sample_tree, df_metadata=empty_metadata, color_column="nonexistent"
            )

    def test_missing_metadata_samples(self, sample_tree):
        """Test handling when tree samples not in metadata."""
        # Create metadata with different sample names
        metadata = pd.DataFrame(
            {"taxon": ["other1", "other2"], "location": ["place1", "place2"]},
            index=["X", "Y"],
        )

        p = baltic_bokeh.plotRectangularPoints(
            sample_tree, df_metadata=metadata, color_column="taxon"
        )
        assert p is not None