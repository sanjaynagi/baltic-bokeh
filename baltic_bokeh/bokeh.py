"""
Bokeh plotting functions for Baltic phylogenetic trees.

This module provides interactive bokeh-based plotting functionality
equivalent to the matplotlib functions but with enhanced interactivity
and metadata integration support.
"""

import math
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category10, Set3
import numpy as np

def plotRectangular(
    tree,
    df_metadata=None,
    color_column=None,
    color_discrete_map=None,
    size=15,
    connection_type='baltic',
    hover_data=None,
    plot_width=600,
    plot_height=600
):
    """
    Plot a rectangular phylogenetic tree with optional metadata-based coloring 
    and interactive hover tooltips using Bokeh.

    This function combines the base tree structure (via `plotRectangularTree`) 
    with scatter points for leaves or nodes (via `plotRectangularPoints`), 
    allowing integration of metadata for coloring and interactivity.

    Parameters
    ----------
    tree : object
        A phylogenetic tree object with `.Objects` containing nodes/branches.
    df_metadata : pandas.DataFrame, optional
        Metadata dataframe indexed by sample names. Used for coloring and hover info.
    color_column : str, optional
        Column in `df_metadata` to use for coloring points.
    color_discrete_map : dict, optional
        Custom mapping of metadata values to colors, e.g. {"A": "red", "B": "blue"}.
        If not provided, a default Bokeh palette is used.
    size : int, default=15
        Size of scatter points representing tree tips/nodes.
    connection_type : {"baltic", "direct"}, default="baltic"
        Type of branch connection:
        - "baltic": horizontal-vertical style (classic phylogenetic tree)
        - "direct": straight-line connections between parent and child
    hover_data : list of str, optional
        List of additional metadata columns to display in hover tooltips.
    plot_width : int, default=600
        Width of the Bokeh plot in pixels.
    plot_height : int, default=600
        Height of the Bokeh plot in pixels.

    Returns
    -------
    bokeh.plotting.figure
        A Bokeh figure object containing the interactive rectangular tree plot.

    Examples
    --------
    >>> p = plotRectangular(
    ...     tree,
    ...     df_metadata=metadata,
    ...     color_column="host",
    ...     hover_data=["location", "date"]
    ... )
    >>> from bokeh.io import show
    >>> show(p)
    """
    p = plotRectangularTree(
        tree, connection_type=connection_type, plot_width=plot_width, plot_height=plot_height
    )

    p = plotRectangularPoints(
        tree,
        p=p,
        df_metadata=df_metadata,
        color_column=color_column,
        color_discrete_map=color_discrete_map,
        size=size,
        hover_data=hover_data,
        plot_width=plot_width,
        plot_height=plot_height
    )

    return p


def plotCircular(
    tree,
    df_metadata=None,
    color_column=None,
    color_discrete_map=None,
    size=15,
    hover_data=None,
    plot_width=600,
    plot_height=600
):
    """
    Plot a circular (radial) phylogenetic tree with optional metadata-based 
    coloring and interactive hover tooltips using Bokeh.

    This function combines the circular tree layout (via `plotCircularTree`) 
    with scatter points for leaves (via `plotCircularPoints`), allowing 
    integration of metadata for coloring and interactivity.

    Parameters
    ----------
    tree : object
        A phylogenetic tree object with `.Objects` containing nodes/branches.
    df_metadata : pandas.DataFrame, optional
        Metadata dataframe indexed by sample names. Used for coloring and hover info.
    color_column : str, optional
        Column in `df_metadata` to use for coloring points.
    color_discrete_map : dict, optional
        Custom mapping of metadata values to colors, e.g. {"A": "red", "B": "blue"}.
        If not provided, a default Bokeh palette is used.
    size : int, default=15
        Size of scatter points representing tree tips.
    hover_data : list of str, optional
        List of additional metadata columns to display in hover tooltips.
    plot_width : int, default=600
        Width of the Bokeh plot in pixels.
    plot_height : int, default=600
        Height of the Bokeh plot in pixels.

    Returns
    -------
    bokeh.plotting.figure
        A Bokeh figure object containing the interactive circular tree plot.

    Examples
    --------
    >>> p = plotCircular(
    ...     tree,
    ...     df_metadata=metadata,
    ...     color_column="host",
    ...     hover_data=["location", "date"]
    ... )
    >>> from bokeh.io import show
    >>> show(p)
    """
    
    p = plotCircularTree(
        tree, plot_width=plot_width, plot_height=plot_height
    )

    p = plotCircularPoints(
        tree,
        p=p,
        df_metadata=df_metadata,
        color_column=color_column,
        color_discrete_map=color_discrete_map,
        size=size,
        hover_data=hover_data,
        plot_width=plot_width,
        plot_height=plot_height
    )

    return p


def plotRectangularTree(
    tree_obj,
    p=None,
    connection_type='baltic',
    x_attr=lambda k: k.x,
    y_attr=lambda k: k.y,
    width=2,
    plot_width=800,
    plot_height=600
):
    """
    Plot the tree using Bokeh with black lines.

    Parameters:
    tree_obj: The tree object
    p (bokeh.plotting.figure or None): Bokeh figure to plot on. If None, creates new figure.
    connection_type (str): Connection type ('baltic', 'direct'). Default 'baltic'.
    x_attr (function or None): Function for x-coordinates. Default uses branch.x.
    y_attr (function or None): Function for y-coordinates. Default uses branch.y.
    width (int): Line width. Default 2.
    plot_width (int): Width of the plot if creating new figure
    plot_height (int): Height of the plot if creating new figure

    Returns:
    bokeh.plotting.figure: The bokeh figure with tree plot
    """
    if p is None:
        p = figure(
            width=plot_width,
            height=plot_height,
            title="Phylogenetic Tree",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )

    assert connection_type in [
        "baltic",
        "direct",
    ], f'Unrecognised drawing type "{connection_type}"'

    # Collect line segments
    xs = []
    ys = []

    for k in tree_obj.Objects:
        x = x_attr(k)
        xp = x_attr(k.parent) if k.parent else x
        y = y_attr(k)

        if connection_type == "baltic":
            # Horizontal branch to parent
            xs.extend([xp, x])
            ys.extend([y, y])

            # Vertical connector for internal nodes
            if k.is_node():
                yl, yr = y_attr(k.children[0]), y_attr(k.children[-1])
                xs.extend([x, x])
                ys.extend([yl, yr])

        elif connection_type == "direct":
            yp = y_attr(k.parent) if k.parent else y
            xs.extend([xp, x])
            ys.extend([yp, y])

    # Plot line segments
    if xs and ys:
        line_data = {"xs": [], "ys": []}

        i = 0
        while i < len(xs) - 1:
            line_xs = [xs[i], xs[i + 1]]
            line_ys = [ys[i], ys[i + 1]]

            line_data["xs"].append(line_xs)
            line_data["ys"].append(line_ys)
            i += 2

        if line_data["xs"]:
            source = ColumnDataSource(line_data)
            p.multi_line("xs", "ys", color="black", line_width=width, source=source)

    return p


def plotRectangularPoints(
    tree_obj,
    p=None,
    x_attr=None,
    y_attr=None,
    size=10,
    df_metadata=None,
    color_column=None,
    color_discrete_map=None,
    alpha=1,
    plot_width=800,
    plot_height=600,
    hover_data=None
):
    """
    Plot points on the tree with interactive features and metadata support.

    Parameters:
    tree_obj: The tree object
    p (bokeh.plotting.figure or None): Bokeh figure to plot on. If None, creates new figure.
    x_attr (function or None): Function for x-coordinates. Default uses branch.x.
    y_attr (function or None): Function for y-coordinates. Default uses branch.y.
    size (int or function or None): Point size. Default 8.
    df_metadata (pd.DataFrame or None): DataFrame with metadata for coloring
    color_column (str or None): Column name in df_metadata for coloring
    color_discrete_map (dict or None): Custom color mapping {value: color}
    plot_width (int): Width of the plot if creating new figure
    plot_height (int): Height of the plot if creating new figure
    hover_data (list or None): Additional columns from metadata to show on hover

    Returns:
    bokeh.plotting.figure: The bokeh figure with points
    """
    if p is None:
        p = figure(
            width=plot_width,
            height=plot_height,
            title="Phylogenetic Tree Points",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )

    if x_attr is None:
        x_attr = lambda k: k.x
    if y_attr is None:
        y_attr = lambda k: k.y

    # Collect point data
    data, hover_data = prepare_bokeh_data(tree_obj, hover_data, df_metadata, color_column, color_discrete_map)
    for k in tree_obj.Objects:
        if k.is_leaf():
            data["x"].append(x_attr(k))
            data["y"].append(y_attr(k))

    return plot_bokeh_scatter(p, data, hover_data, size, alpha)

def plotCircularTree(
    tree_obj,
    p=None,
    x_attr=lambda k: k.x,
    y_attr=lambda k: k.y,
    width=2,
    circStart=0.0,
    circFrac=1.0,
    inwardSpace=0.0,
    normaliseHeight=None,
    precision=15,
    plot_width=800,
    plot_height=800
):
    """
    Plot the tree in a circular layout using Bokeh with black lines.

    Parameters:
    tree_obj: The tree object
    p (bokeh.plotting.figure or None): Bokeh figure to plot on. If None, creates new figure.
    x_attr (function or None): Function for x-coordinates. Default uses branch.x.
    y_attr (function or None): Function for y-coordinates. Default uses branch.y.
    width (int): Line width. Default 2.
    circStart (float): Starting angle in fractions of 2*pi. Default 0.0.
    circFrac (float): Fraction of full circle to use. Default 1.0.
    inwardSpace (float): Space to leave in middle. Default 0.0.
    normaliseHeight (function or None): Height normalization function.
    precision (int): Number of points for curved segments. Default 15.
    plot_width (int): Width of the plot if creating new figure
    plot_height (int): Height of the plot if creating new figure

    Returns:
    bokeh.plotting.figure: The bokeh figure with circular tree
    """
    if p is None:
        p = figure(
            width=plot_width,
            height=plot_height,
            title="Circular Phylogenetic Tree",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.axis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False

    if inwardSpace < 0:
        inwardSpace -= tree_obj.treeHeight

    circ_s = circStart * math.pi * 2
    circ = circFrac * math.pi * 2

    allXs = list(map(x_attr, tree_obj.Objects))
    if normaliseHeight is None:
        normaliseHeight = lambda value: (value - min(allXs)) / (max(allXs) - min(allXs))

    linspace = lambda start, stop, n: (
        list(start + ((stop - start) / (n - 1)) * i for i in range(n))
        if n > 1
        else [stop]
    )

    # Collect line segments for circular tree
    line_data = {"xs": [], "ys": []}

    for k in tree_obj.Objects:
        x = normaliseHeight(x_attr(k) + inwardSpace)
        xp = (
            normaliseHeight(x_attr(k.parent) + inwardSpace)
            if k.parent and k.parent.parent
            else x
        )
        y = y_attr(k)

        y = circ_s + circ * y / tree_obj.ySpan
        X = math.sin(y)
        Y = math.cos(y)

        # Radial branch
        line_data["xs"].append([X * xp, X * x])
        line_data["ys"].append([Y * xp, Y * x])

        # Curved connector for internal nodes
        if k.is_node():
            yl, yr = y_attr(k.children[0]), y_attr(k.children[-1])
            yl = circ_s + circ * yl / tree_obj.ySpan
            yr = circ_s + circ * yr / tree_obj.ySpan
            ybar = linspace(yl, yr, precision)

            xs = [math.sin(yb) * x for yb in ybar]
            ys = [math.cos(yb) * x for yb in ybar]

            # Add curved segments
            for i in range(len(xs) - 1):
                line_data["xs"].append([xs[i], xs[i + 1]])
                line_data["ys"].append([ys[i], ys[i + 1]])

    if line_data["xs"]:
        source = ColumnDataSource(line_data)
        p.multi_line("xs", "ys", color="black", line_width=width, source=source)

    return p


def plotCircularPoints(
    tree_obj,
    p=None,
    x_attr=None,
    y_attr=None,
    size=15,
    circStart=0.0,
    circFrac=1.0,
    inwardSpace=0.0,
    normaliseHeight=None,
    df_metadata=None,
    color_column=None,
    color_discrete_map=None,
    alpha=1,
    plot_width=800,
    plot_height=800,
    hover_data=None
):
    """
    Plot points on a circular tree with interactive features and metadata support.
    """
    if p is None:
        p = figure(
            width=plot_width,
            height=plot_height,
            title="Circular Tree Points",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.axis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False

    if x_attr is None:
        x_attr = lambda k: k.x
    if y_attr is None:
        y_attr = lambda k: k.y

    data, hover_data = prepare_bokeh_data(tree_obj, hover_data, df_metadata, color_column, color_discrete_map)

    # add coordinates
    if inwardSpace < 0:
        inwardSpace -= tree_obj.treeHeight

    circ_s = circStart * math.pi * 2
    circ = circFrac * math.pi * 2

    allXs = list(map(x_attr, tree_obj.Objects))
    if normaliseHeight is None:
        normaliseHeight = lambda value: (value - min(allXs)) / (max(allXs) - min(allXs))

    for k in tree_obj.Objects:
        if k.is_leaf():
            x = normaliseHeight(x_attr(k) + inwardSpace)
            y = circ_s + circ * y_attr(k) / tree_obj.ySpan
            X = math.sin(y) * x
            Y = math.cos(y) * x
        
            data["x"].append(X)
            data["y"].append(Y)

    return plot_bokeh_scatter(p, data, hover_data, size, alpha)

def plot_bokeh_scatter(p, data, hover_data=None, size=8, alpha=0.8):
    """
    Add scatter points with hover tooltips to a Bokeh figure.
    
    Parameters:
    p (bokeh.plotting.figure): Bokeh figure to add scatter to
    data (dict): Dictionary containing 'x', 'y', 'names', 'colors' and other data
    color_column (str or None): Column name for color metadata
    df_metadata (pd.DataFrame or None): Metadata dataframe 
    hover_data (list or None): List of additional column names for hover tooltips
    size (int or function): Size of scatter points. Default 8.
    alpha (float): Alpha transparency of points. Default 0.8.
    
    Returns:
    bokeh.plotting.figure: The figure with scatter points added
    """
    source = ColumnDataSource(data)

    # Create hover tool
    hover_tooltips = []
    if hover_data:
        for col in hover_data:
            if col in data:
                hover_tooltips.append((col, f"@{col}"))

    # Plot points
    renderer = p.scatter(
        "x", "y", size=size, color="colors", source=source, alpha=alpha
    )

    hover = HoverTool(tooltips=hover_tooltips, renderers=[renderer])
    p.add_tools(hover)

    return p

from bokeh.palettes import Category10, Set3

def generate_leaf_colours(
    df_metadata, color_column, color_discrete_map=None, default_color="grey"
):
    """
    Helper function to prepare colors based on metadata DataFrame.

    Parameters:
    df_metadata (pd.DataFrame): DataFrame with metadata, indexed by sample names
    color_column (str): Column name to use for coloring
    color_discrete_map (dict or None): Custom color mapping {value: color}
    default_color (str): Fallback color for values not in the mapping

    Returns:
    list: List of colors corresponding to metadata entries
    """

    unique_values = df_metadata[color_column].dropna().unique()

    # Use provided color map or create default one
    if color_discrete_map:
        value_to_color = color_discrete_map.copy()
    else:
        # Use bokeh palettes for default colors
        if len(unique_values) <= 3:
            palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        elif len(unique_values) <= 10:
            palette = Category10[max(3, len(unique_values))]
        else:
            palette = Set3[12]

        value_to_color = {
            val: palette[i % len(palette)] for i, val in enumerate(unique_values)
        }

    # Assign colors, defaulting to grey if not found
    colours = [
        value_to_color.get(val, default_color)
        for val in df_metadata[color_column].values
    ]

    return colours

def prepare_bokeh_data(tree_obj, hover_data, df_metadata, color_column, color_discrete_map):
    # Collect point data
    data = {"x": [], "y": []}
    if hover_data is None and df_metadata is not None:
        hover_data = df_metadata.columns.to_list()
    elif hover_data is None and df_metadata is None:
        hover_data = []

    for h in hover_data:
        data[h] = df_metadata[h].values

    if df_metadata is not None and color_column is not None:
        assert color_column in df_metadata.columns, f"provided {color_column} as color_column not in metadata"
        data['colors'] = generate_leaf_colours(
            df_metadata, color_column, color_discrete_map
        )
    elif color_column is None or df_metadata is None:
        data['colors'] = np.repeat("black", np.sum([o.is_leaf() for o in tree_obj.Objects]))


    return data, hover_data


def addText(tree_obj, p, target=None, x_attr=None, y_attr=None, text=None, **kwargs):
    """
    Add text annotations to a bokeh tree plot.

    Note: Bokeh text rendering is different from matplotlib.
    This is a basic implementation that adds text labels.
    """
    if target is None:
        target = lambda k: k.is_leaf()
    if x_attr is None:
        x_attr = lambda k: k.x
    if y_attr is None:
        y_attr = lambda k: k.y
    if text is None:
        text = lambda k: k.name

    data = {"x": [], "y": [], "text": []}

    for k in filter(target, tree_obj.Objects):
        data["x"].append(x_attr(k))
        data["y"].append(y_attr(k))
        data["text"].append(text(k))

    if data["x"]:
        source = ColumnDataSource(data)
        p.text("x", "y", text="text", source=source, text_font_size="8pt")

    return p
