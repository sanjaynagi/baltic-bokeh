"""
Tests for baltic_bokeh plotting functions.
"""

import pytest
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from unittest.mock import Mock, patch

import baltic_bokeh
from baltic_bokeh.bokeh import _prepare_metadata_colors


class TestMetadataColors:
    """Test metadata color preparation."""
    
    def test_prepare_metadata_colors_with_metadata(self, sample_tree, sample_metadata, color_map):
        """Test color preparation with valid metadata."""
        color_result = _prepare_metadata_colors(
            sample_tree, sample_metadata, 'species', color_map
        )
        
        # Check that colors are assigned
        assert len(color_result) > 0
        
        # Check that leaf nodes get colors from metadata
        leaves = [k for k in sample_tree.Objects if k.is_leaf()]
        for leaf in leaves:
            if hasattr(leaf, 'name') and leaf.name in sample_metadata.index:
                species = sample_metadata.loc[leaf.name, 'species']
                expected_color = color_map.get(species, 'gray')
                assert color_result.get(leaf) == expected_color
    
    def test_prepare_metadata_colors_without_metadata(self, sample_tree):
        """Test color preparation without metadata."""
        color_result = _prepare_metadata_colors(sample_tree, None, None, None)
        
        # Should default to black for all objects
        leaves = [k for k in sample_tree.Objects if k.is_leaf()]
        for leaf in leaves:
            assert color_result.get(leaf) == 'black'
    
    def test_prepare_metadata_colors_missing_column(self, sample_tree, sample_metadata):
        """Test color preparation with missing column."""
        color_result = _prepare_metadata_colors(
            sample_tree, sample_metadata, 'nonexistent_column', None
        )
        
        # Should default to black when column doesn't exist
        leaves = [k for k in sample_tree.Objects if k.is_leaf()]
        for leaf in leaves:
            assert color_result.get(leaf) == 'black'


class TestPlotTree:
    """Test plotTree function."""
    
    def test_plot_tree_basic(self, sample_tree):
        """Test basic tree plotting."""
        p = baltic_bokeh.plotTree(sample_tree, plot_width=400, plot_height=400)
        
        assert p is not None
        assert hasattr(p, 'renderers')
        # Should have created a figure
        assert p.width == 400
        assert p.height == 400
    
    def test_plot_tree_with_metadata(self, sample_tree, sample_metadata, color_map):
        """Test tree plotting with metadata coloring."""
        p = baltic_bokeh.plotTree(
            sample_tree,
            metadata_df=sample_metadata,
            color_column='species',
            color_discrete_map=color_map,
            plot_width=500,
            plot_height=500
        )
        
        assert p is not None
        assert p.width == 500
        assert p.height == 500
    
    def test_plot_tree_custom_figure(self, sample_tree):
        """Test plotting on existing figure."""
        existing_fig = figure(width=300, height=300)
        p = baltic_bokeh.plotTree(sample_tree, p=existing_fig)
        
        assert p is existing_fig
        assert p.width == 300
        assert p.height == 300
    
    def test_plot_tree_connection_types(self, sample_tree):
        """Test different connection types."""
        for conn_type in ['baltic', 'direct', 'elbow']:
            p = baltic_bokeh.plotTree(sample_tree, connection_type=conn_type)
            assert p is not None
    
    def test_plot_tree_invalid_connection_type(self, sample_tree):
        """Test invalid connection type raises error."""
        with pytest.raises(AssertionError):
            baltic_bokeh.plotTree(sample_tree, connection_type='invalid')


class TestPlotPoints:
    """Test plotPoints function."""
    
    def test_plot_points_basic(self, sample_tree):
        """Test basic point plotting."""
        p = baltic_bokeh.plotPoints(sample_tree, plot_width=400, plot_height=400)
        
        assert p is not None
        assert p.width == 400
        assert p.height == 400
    
    def test_plot_points_with_metadata(self, sample_tree, sample_metadata, color_map):
        """Test point plotting with metadata."""
        p = baltic_bokeh.plotPoints(
            sample_tree,
            metadata_df=sample_metadata,
            color_column='species',
            color_discrete_map=color_map,
            hover_data=['location', 'year']
        )
        
        assert p is not None
        # Should have hover tools
        hover_tools = [tool for tool in p.tools if isinstance(tool, HoverTool)]
        assert len(hover_tools) > 0
    
    def test_plot_points_custom_target(self, sample_tree):
        """Test point plotting with custom target function."""
        # Plot only leaves (default behavior) since internal nodes may not have names
        p = baltic_bokeh.plotPoints(sample_tree, target=lambda k: k.is_leaf())
        assert p is not None
    
    def test_plot_points_size_and_color(self, sample_tree):
        """Test point plotting with custom size and color."""
        p = baltic_bokeh.plotPoints(sample_tree, size=12, colour='red')
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
            sample_tree,
            circStart=0.25,
            circFrac=0.5,
            inwardSpace=0.1
        )
        assert p is not None
    
    def test_plot_circular_tree_with_metadata(self, sample_tree, sample_metadata, color_map):
        """Test circular tree with metadata coloring."""
        p = baltic_bokeh.plotCircularTree(
            sample_tree,
            metadata_df=sample_metadata,
            color_column='species',
            color_discrete_map=color_map
        )
        assert p is not None


class TestPlotCircularPoints:
    """Test plotCircularPoints function."""
    
    def test_plot_circular_points_basic(self, sample_tree):
        """Test basic circular point plotting."""
        p = baltic_bokeh.plotCircularPoints(sample_tree, plot_width=400, plot_height=400)
        
        assert p is not None
        assert p.width == 400
        assert p.height == 400
        assert all(not axis.visible for axis in p.axis)
    
    def test_plot_circular_points_with_metadata(self, sample_tree, sample_metadata, color_map):
        """Test circular points with metadata and hover."""
        p = baltic_bokeh.plotCircularPoints(
            sample_tree,
            metadata_df=sample_metadata,
            color_column='species',
            color_discrete_map=color_map,
            hover_data=['location', 'year']
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
            text=lambda k: f"Node_{k.name}"
        )
        
        assert p_with_text is p


class TestPackageImports:
    """Test package-level imports."""
    
    def test_import_all_functions(self):
        """Test that all functions are importable."""
        import baltic_bokeh
        
        expected_functions = [
            'plotTree',
            'plotPoints', 
            'plotCircularTree',
            'plotCircularPoints',
            'addText'
        ]
        
        for func_name in expected_functions:
            assert hasattr(baltic_bokeh, func_name)
            assert callable(getattr(baltic_bokeh, func_name))
    
    def test_version_info(self):
        """Test version information is available."""
        import baltic_bokeh
        
        assert hasattr(baltic_bokeh, '__version__')
        assert hasattr(baltic_bokeh, '__author__')
        assert hasattr(baltic_bokeh, '__all__')
        
        # Check __all__ contains expected functions
        assert set(baltic_bokeh.__all__) == {
            'plotTree', 'plotPoints', 'plotCircularTree', 
            'plotCircularPoints', 'addText'
        }


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_metadata(self, sample_tree):
        """Test handling of empty metadata."""
        empty_metadata = pd.DataFrame()
        
        p = baltic_bokeh.plotPoints(
            sample_tree,
            metadata_df=empty_metadata,
            color_column='nonexistent'
        )
        assert p is not None
    
    def test_missing_metadata_samples(self, sample_tree):
        """Test handling when tree samples not in metadata."""
        # Create metadata with different sample names
        metadata = pd.DataFrame({
            'species': ['other1', 'other2'],
            'location': ['place1', 'place2']
        }, index=['X', 'Y'])
        
        p = baltic_bokeh.plotPoints(
            sample_tree,
            metadata_df=metadata,
            color_column='species'
        )
        assert p is not None
    
    def test_none_values_in_metadata(self, sample_tree, sample_metadata):
        """Test handling of None/NaN values in metadata."""
        # Add some NaN values
        metadata_with_nan = sample_metadata.copy()
        metadata_with_nan.loc['A', 'species'] = None
        
        p = baltic_bokeh.plotPoints(
            sample_tree,
            metadata_df=metadata_with_nan,
            color_column='species'
        )
        assert p is not None