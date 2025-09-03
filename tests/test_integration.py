"""
Integration tests for baltic-bokeh package.
"""

import pytest
import os
import pandas as pd
import baltic as bt
import baltic_bokeh
from bokeh.plotting import figure


class TestIntegration:
    """Integration tests using example files from examples directory."""

    def test_full_workflow_circular(self, sample_tree, sample_metadata):
        """Test complete workflow with circular tree."""
        # Use fixtures
        tree = sample_tree
        metadata = sample_metadata

        # Define colors
        colors = {
            "gambiae": "#1f77b4",
            "coluzzii": "#ff7f0e",
            "arabiensis": "#2ca02c",
            "merus": "#d62728",
            "melas": "#9467bd",
        }

        # Create circular tree
        p = baltic_bokeh.plotCircularTree(tree, plot_width=600, plot_height=600)

        # Add points with metadata
        p = baltic_bokeh.plotCircularPoints(
            tree,
            p=p,
            df_metadata=metadata,
            color_column="taxon",
            color_discrete_map=colors,
            hover_data=["location", "country"],
        )

        # Verify the result
        assert p is not None
        assert p.width == 600
        assert p.height == 600
        assert len(p.renderers) > 0  # Should have some renderers

    def test_full_workflow_rectangular(self, sample_tree, sample_metadata):
        """Test complete workflow with rectangular tree."""
        # Use fixtures
        tree = sample_tree
        metadata = sample_metadata

        # Create rectangular tree
        p = baltic_bokeh.plotRectangularTree(
            tree,
            plot_width=800,
            plot_height=400,
        )

        # Add points
        p = baltic_bokeh.plotRectangularPoints(
            tree,
            p=p,
            df_metadata=metadata,
            color_column="taxon",
            hover_data=["location", "country"],
        )

        # Add text labels
        p = baltic_bokeh.addText(tree, p)

        # Verify the result
        assert p is not None
        assert p.width == 800
        assert p.height == 400
        assert len(p.renderers) > 0

    def test_multiple_connection_types(self, sample_tree):
        """Test all connection types work with real tree."""
        tree = sample_tree

        for conn_type in ["baltic", "direct"]:
            p = baltic_bokeh.plotRectangularTree(tree, connection_type=conn_type)
            assert p is not None
            assert len(p.renderers) > 0

    def test_without_metadata(self, sample_tree):
        """Test plotting without any metadata."""
        tree = sample_tree

        # Tree plot
        p1 = baltic_bokeh.plotRectangularTree(tree)
        assert p1 is not None

        # Points plot
        p2 = baltic_bokeh.plotRectangularPoints(tree)
        assert p2 is not None

        # Circular tree
        p3 = baltic_bokeh.plotCircularTree(tree)
        assert p3 is not None

        # Circular points
        p4 = baltic_bokeh.plotCircularPoints(tree)
        assert p4 is not None

    def test_chaining_operations(self, sample_tree, sample_metadata):
        """Test chaining multiple operations on the same figure."""
        tree = sample_tree
        metadata = sample_metadata

        # Start with base figure
        p = figure(width=600, height=600)
        original_renderer_count = len(p.renderers)

        # Chain operations
        p = baltic_bokeh.plotRectangularTree(tree, p=p)
        tree_renderer_count = len(p.renderers)

        p = baltic_bokeh.plotRectangularPoints(
            tree, p=p, df_metadata=metadata, color_column="taxon"
        )
        points_renderer_count = len(p.renderers)

        p = baltic_bokeh.addText(tree, p)
        text_renderer_count = len(p.renderers)

        # Verify that renderers were added at each step
        assert tree_renderer_count > original_renderer_count
        assert points_renderer_count > tree_renderer_count
        assert (
            text_renderer_count >= points_renderer_count
        )  # Text might not add renderers if no data

    def test_error_handling_invalid_files(self):
        """Test error handling with invalid files."""
        with pytest.raises(
            Exception
        ):  # Baltic should raise an exception for invalid files
            bt.loadNewick("nonexistent_file.newick")

    def test_large_metadata_columns(self, sample_tree):
        """Test with metadata containing many columns."""
        tree = sample_tree

        # Create metadata with many columns
        data = {}
        column_names = []
        for i in range(20):
            col_name = f"column_{i}"
            column_names.append(col_name)
            data[col_name] = [f"value_{i}_{j}" for j in range(5)]

        # Get sample names from tree
        sample_names = [k.name for k in tree.Objects if k.is_leaf()]
        metadata = pd.DataFrame(data, index=sample_names[: len(data["column_0"])])

        # Should handle many hover columns
        p = baltic_bokeh.plotRectangularPoints(
            tree,
            df_metadata=metadata,
            color_column="column_0",
            hover_data=column_names[:10],  # First 10 columns
        )

        assert p is not None
