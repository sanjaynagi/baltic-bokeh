"""
Baltic-Bokeh: Interactive phylogenetic tree visualization

This package provides interactive Bokeh-based plotting functionality for Baltic phylogenetic trees,
offering enhanced interactivity and metadata integration support compared to static matplotlib plots.
"""

from .bokeh import plotCircular, plotRectangular, plotRectangularTree, plotRectangularPoints, plotCircularTree, plotCircularPoints, addText

__version__ = "0.1.0"
__author__ = "Sanjay C Nagi"
__email__ = "sanjay.c.nagi@gmail.com"

__all__ = [
    "plotRectangularTree",
    "plotRectangularPoints",
    "plotCircularTree",
    "plotCircularPoints",
    "addText",
]
