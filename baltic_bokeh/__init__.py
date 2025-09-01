"""
Baltic-Bokeh: Interactive phylogenetic tree visualization

This package provides interactive Bokeh-based plotting functionality for Baltic phylogenetic trees,
offering enhanced interactivity and metadata integration support compared to static matplotlib plots.
"""

from .bokeh import plotTree, plotPoints, plotCircularTree, plotCircularPoints, addText

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "plotTree",
    "plotPoints",
    "plotCircularTree",
    "plotCircularPoints",
    "addText",
]
