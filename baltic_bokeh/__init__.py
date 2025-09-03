"""
Baltic-Bokeh: Interactive phylogenetic tree visualization

This package provides interactive Bokeh-based plotting functionality for Baltic phylogenetic trees,
offering enhanced interactivity and metadata integration support compared to static matplotlib plots.
"""

from .bokeh import plotTree, plotRectangularTree, plotRectangularPoints, plotCircularTree, plotCircularPoints, plot_bokeh_scatter, addText

__version__ = "0.1.0"
__author__ = "Sanjay C Nagi"
__email__ = "sanjay.c.nagi@gmail.com"

__all__ = [
    "plot_bokeh_scatter",
    "plotTree",
    "plotRectangularTree",
    "plotRectangularPoints",
    "plotCircularTree",
    "plotCircularPoints",
    "addText",
]
