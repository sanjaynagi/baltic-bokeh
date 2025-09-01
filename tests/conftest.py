"""
Test fixtures and configuration for baltic-bokeh tests.
"""

import pytest
import pandas as pd
import baltic as bt
import os
from unittest.mock import Mock


@pytest.fixture
def sample_tree():
    """Load sample Baltic tree from examples directory."""
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
    tree_file = os.path.join(examples_dir, "example.newick")
    tree = bt.loadNewick(tree_file, absoluteTime=False)
    return tree


@pytest.fixture
def sample_metadata():
    """Load sample metadata DataFrame from examples directory."""
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
    metadata_file = os.path.join(examples_dir, "example_metadata.tsv")
    return pd.read_csv(metadata_file, sep="\t", index_col=0)


@pytest.fixture
def color_map():
    """Sample color mapping for testing."""
    return {
        "gambiae": "#1f77b4",
        "coluzzii": "#ff7f0e",
        "arabiensis": "#2ca02c",
        "merus": "#d62728",
        "melas": "#9467bd",
    }


@pytest.fixture
def mock_bokeh_figure():
    """Mock bokeh figure for testing."""
    mock_fig = Mock()
    mock_fig.multi_line = Mock()
    mock_fig.scatter = Mock()
    mock_fig.text = Mock()
    mock_fig.add_tools = Mock()
    return mock_fig
