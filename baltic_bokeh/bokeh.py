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
from numba import njit

def plotTree(
    tree,
    type="rectangular",  # "rectangular", "circular", "unrooted"
    df_metadata=None,
    color_column=None,
    color_discrete_map=None,
    size=15,
    connection_type="baltic",  # only used for rectangular
    hover_data=None,
    marker="circle",
    marker_line_color="black",
    marker_line_width=1,
    output_backend="webgl",
    plot_width=800,
    plot_height=800,
):
    """
    Plot a phylogenetic tree (rectangular, circular, or unrooted) with optional
    metadata-based coloring and interactive hover tooltips using Bokeh.

    Parameters
    ----------
    tree : object
        A phylogenetic tree object with `.Objects` containing nodes/branches.
    type : {"rectangular", "circular", "unrooted"}, default="rectangular"
        Layout type for the tree.
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
        Only used for rectangular layout.
    hover_data : list of str, optional
        List of additional metadata columns to display in hover tooltips.
    marker : str, default="circle"
        Type of glyph marker for points. Options include:
        'circle', 'square', 'hex', 'dot', etc.
    marker_line_color : str, default="black"
        Outline color of glyph markers.
    marker_line_width : int, default=1
        Outline thickness of glyph markers.
    output_backend : str, default="webgl"
        Bokeh output backend. 'webgl' is recommended for better performance with many points.
    plot_width : int, default=800
        Width of the Bokeh plot in pixels.
    plot_height : int, default=800
        Height of the Bokeh plot in pixels.

    Returns
    -------
    bokeh.plotting.figure
        A Bokeh figure object containing the interactive tree plot.

    Examples
    --------
    >>> p = plotTree(
    ...     tree,
    ...     type="unrooted",
    ...     df_metadata=metadata,
    ...     color_column="host",
    ...     hover_data=["location", "date"]
    ... )
    >>> from bokeh.io import show
    >>> show(p)
    """
    type = type.lower()
    assert type in ["rectangular", "circular", "unrooted", "r", "c", "u"], f"Unknown tree type: {type}"

    if type == "rectangular" or type == 'r':
        # Draw edges
        p = plotRectangularTree(
            tree,
            connection_type=connection_type,
            plot_width=plot_width,
            plot_height=plot_height,
            output_backend=output_backend,
        )
        # Add points
        p = plotRectangularPoints(
            tree,
            p=p,
            df_metadata=df_metadata,
            color_column=color_column,
            color_discrete_map=color_discrete_map,
            size=size,
            hover_data=hover_data,
            output_backend=output_backend,
            marker=marker,
            marker_line_color=marker_line_color,
            marker_line_width=marker_line_width,
            plot_width=plot_width,
            plot_height=plot_height,
        )

    elif type == "circular" or type == 'c':
        # Draw edges
        p = plotCircularTree(
            tree,
            plot_width=plot_width,
            plot_height=plot_height,
            output_backend=output_backend,
        )
        # Add points
        p = plotCircularPoints(
            tree,
            p=p,
            df_metadata=df_metadata,
            color_column=color_column,
            color_discrete_map=color_discrete_map,
            size=size,
            hover_data=hover_data,
            marker=marker,
            marker_line_color=marker_line_color,
            marker_line_width=marker_line_width,
            output_backend=output_backend,
            plot_width=plot_width,
            plot_height=plot_height,
        )

    elif type == "unrooted" or type == 'u':
        # Draw edges
        p, df_nodes = plotUnrootedTree(
            tree,
            plot_width=plot_width,
            plot_height=plot_height,
            output_backend=output_backend,
        )
        # Add points (only leaves)
        p = plotUnrootedPoints(
            tree,
            p=p,
            df_metadata=df_metadata,
            color_column=color_column,
            color_discrete_map=color_discrete_map,
            size=size,
            hover_data=hover_data,
            marker=marker,
            marker_line_color=marker_line_color,
            marker_line_width=marker_line_width,
            output_backend=output_backend,
            plot_width=plot_width,
            plot_height=plot_height,
        )

    return p

def plotRectangularTree(
    tree,
    p=None,
    connection_type='baltic',
    x_attr=lambda k: k.x,
    y_attr=lambda k: k.y,
    width=2,
    plot_width=800,
    plot_height=600,
    output_backend='webgl'
):
    """
    Plot the tree using Bokeh with black lines.

    Parameters:
    tree: The tree object
    p (bokeh.plotting.figure or None): Bokeh figure to plot on. If None, creates new figure.
    connection_type (str): Connection type ('baltic', 'direct'). Default 'baltic'.
    x_attr (function or None): Function for x-coordinates. Default uses branch.x.
    y_attr (function or None): Function for y-coordinates. Default uses branch.y.
    width (int): Line width. Default 2.
    plot_width (int): Width of the plot if creating new figure
    plot_height (int): Height of the plot if creating new figure
    output_backend (str): Bokeh output backend. Default 'webgl' for better performance with many lines.

    Returns:
    bokeh.plotting.figure: The bokeh figure with tree plot
    """
    if p is None:
        p = figure(
            width=plot_width,
            height=plot_height,
            title="Phylogenetic Tree",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            output_backend=output_backend
        )

    assert connection_type in [
        "baltic",
        "direct",
    ], f'Unrecognised drawing type "{connection_type}"'

    # Collect line segments
    xs = []
    ys = []

    for k in tree.Objects:
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
    tree,
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
    marker="circle",
    marker_line_color="black",
    marker_line_width=1,
    hover_data=None,
    output_backend='webgl'
):
    """
    Plot points on the tree with interactive features and metadata support.

    Parameters:
    tree: The tree object
    p (bokeh.plotting.figure or None): Bokeh figure to plot on. If None, creates new figure.
    x_attr (function or None): Function for x-coordinates. Default uses branch.x.
    y_attr (function or None): Function for y-coordinates. Default uses branch.y.
    size (int or function or None): Point size. Default 8.
    df_metadata (pd.DataFrame or None): DataFrame with metadata for coloring
    color_column (str or None): Column name in df_metadata for coloring
    color_discrete_map (dict or None): Custom color mapping {value: color}
    alpha (float): Alpha transparency of points. Default 1.
    marker (str): Type of glyph marker. Options include:
                  'circle', 'square', 'triangle', 'diamond',
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
            output_backend=output_backend
        )

    if x_attr is None:
        x_attr = lambda k: k.x
    if y_attr is None:
        y_attr = lambda k: k.y

    # Collect point data
    data, hover_data = prepare_bokeh_data(tree, hover_data, df_metadata, color_column, color_discrete_map)
    for k in tree.Objects:
        if k.is_leaf():
            data["x"].append(x_attr(k))
            data["y"].append(y_attr(k))

    return plot_bokeh_scatter(p=p, data=data, hover_data=hover_data, size=size, alpha=alpha, marker=marker, marker_line_color=marker_line_color, marker_line_width=marker_line_width)

def plotCircularTree(
    tree,
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
    plot_height=800,
    output_backend='webgl'
):
    """
    Plot the tree in a circular layout using Bokeh with black lines.

    Parameters:
    tree: The tree object
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
            output_backend=output_backend
        )
        p.axis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False


    xs, ys, parent, is_node, left_child, right_child = extract_tree_arrays_trees(tree, x_attr, y_attr)

    seg_xs, seg_ys = compute_circular_segments(
        xs, ys, parent, is_node, left_child, right_child,
        tree.treeHeight, tree.ySpan,
        inwardSpace, circStart, circFrac, precision
    )

    if seg_xs:
        source = ColumnDataSource({"xs": seg_xs, "ys": seg_ys})
        p.multi_line("xs", "ys", color="black", line_width=width, source=source)

    return p


def plotCircularPoints(
    tree,
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
    marker="circle",
    marker_line_color="black",
    marker_line_width=1,
    hover_data=None,
    output_backend='webgl',
    plot_width=800,
    plot_height=800,
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
            output_backend=output_backend
        )
        p.axis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False

    if x_attr is None:
        x_attr = lambda k: k.x
    if y_attr is None:
        y_attr = lambda k: k.y

    data, hover_data = prepare_bokeh_data(tree, hover_data, df_metadata, color_column, color_discrete_map)

    xs, ys, is_leaf = extract_tree_arrays_points(tree, x_attr, y_attr)

    X, Y = compute_circular_coords(
        xs, ys, is_leaf,
        tree_height=tree.treeHeight,
        y_span=tree.ySpan,
        inwardSpace=inwardSpace,
        circStart=circStart,
        circFrac=circFrac
    )

    data["x"].extend(X.tolist())
    data["y"].extend(Y.tolist())

    return plot_bokeh_scatter(p=p, data=data, hover_data=hover_data, size=size, alpha=alpha, marker=marker,
                             marker_line_color=marker_line_color, marker_line_width=marker_line_width)


def plot_bokeh_scatter(
    p,
    data,
    hover_data=None,
    size=8,
    alpha=0.8,
    marker="circle",  # glyph type
    marker_line_color="black",  # outline color
    marker_line_width=1,        # outline thickness
):
    """
    Add scatter points with hover tooltips to a Bokeh figure.
    
    Parameters:
    p (bokeh.plotting.figure): Bokeh figure to add scatter to
    data (dict): Dictionary containing 'x', 'y', 'names', 'colors' and other data
    hover_data (list or None): List of additional column names for hover tooltips
    size (int or function): Size of scatter points. Default 8.
    alpha (float): Alpha transparency of points. Default 0.8.
    marker (str): Type of glyph marker. Options include:
                  'circle', 'square', 'triangle', 'diamond',
                  'hex', 'cross', 'x', 'dot', etc.
                  Default is 'circle'.
    line_color (str): Outline color of glyphs. Default 'black'.
    line_width (int): Outline thickness. Default 1.
    
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

    # Plot points with chosen marker and outline
    renderer = p.scatter(
        "x",
        "y",
        size=size,
        fill_color="colors",  # inside color
        line_color=marker_line_color,  # outline color
        line_width=marker_line_width,  # outline thickness
        source=source,
        alpha=alpha,
        marker=marker,
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

def prepare_bokeh_data(tree, hover_data, df_metadata, color_column, color_discrete_map):
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
        data['colors'] = np.repeat("black", np.sum([o.is_leaf() for o in tree.Objects]))


    return data, hover_data


def addText(tree, p, target=None, x_attr=None, y_attr=None, text=None, **kwargs):
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

    for k in filter(target, tree.Objects):
        data["x"].append(x_attr(k))
        data["y"].append(y_attr(k))
        data["text"].append(text(k))

    if data["x"]:
        source = ColumnDataSource(data)
        p.text("x", "y", text="text", source=source, text_font_size="8pt")

    return p




def extract_tree_arrays_points(tree, x_attr, y_attr):
    xs, ys, is_leaf = [], [], []
    for k in tree.Objects:
        xs.append(x_attr(k))
        ys.append(y_attr(k))
        is_leaf.append(k.is_leaf())
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64), np.array(is_leaf, dtype=np.bool_)

def extract_tree_arrays_trees(tree, x_attr, y_attr):
    n = len(tree.Objects)
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int32)       # -1 means no parent
    is_node = np.zeros(n, dtype=np.bool_)
    left_child = np.full(n, -1, dtype=np.int32)   # -1 means no child
    right_child = np.full(n, -1, dtype=np.int32)

    # Map each node object to its index
    index_map = {k: i for i, k in enumerate(tree.Objects)}

    for i, k in enumerate(tree.Objects):
        xs[i] = x_attr(k)
        ys[i] = y_attr(k)

        # Assign parent index if it exists in the map
        if k.parent is not None and k.parent in index_map:
            parent[i] = index_map[k.parent]

        # Assign children if it's an internal node
        if k.is_node():
            is_node[i] = True
            if k.children:
                left_child[i] = index_map.get(k.children[0], -1)
                right_child[i] = index_map.get(k.children[-1], -1)

    return xs, ys, parent, is_node, left_child, right_child

@njit
def compute_circular_segments(xs, ys, parent, is_node, left_child, right_child,
                              tree_height, y_span, inwardSpace,
                              circStart, circFrac, precision):
    n = len(xs)
    max_segments = n * (precision + 2)  # rough upper bound
    seg_xs = []
    seg_ys = []

    if inwardSpace < 0:
        inwardSpace -= tree_height

    circ_s = circStart * math.pi * 2.0
    circ = circFrac * math.pi * 2.0

    min_x = xs.min()
    max_x = xs.max()
    denom = max_x - min_x

    for i in range(n):
        x = (xs[i] + inwardSpace - min_x) / denom
        xp = x

        # If node has a parent, use its radius
        if parent[i] != -1:
            xp = (xs[parent[i]] + inwardSpace - min_x) / denom
        else:
            # Root or unrooted center: start from the origin
            xp = 0.0

        y = circ_s + circ * ys[i] / y_span
        X = math.sin(y)
        Y = math.cos(y)

        # Radial branch (now includes root → child)
        seg_xs.append([X * xp, X * x])
        seg_ys.append([Y * xp, Y * x])

        # Curved connector for internal nodes
        if is_node[i] and left_child[i] != -1 and right_child[i] != -1:
            yl = circ_s + circ * ys[left_child[i]] / y_span
            yr = circ_s + circ * ys[right_child[i]] / y_span

            for j in range(precision - 1):
                t1 = yl + (yr - yl) * j / (precision - 1)
                t2 = yl + (yr - yl) * (j + 1) / (precision - 1)
                seg_xs.append([math.sin(t1) * x, math.sin(t2) * x])
                seg_ys.append([math.cos(t1) * x, math.cos(t2) * x])

    return seg_xs, seg_ys

@njit
def compute_circular_coords(xs, ys, is_leaf, tree_height, y_span, inwardSpace, circStart, circFrac):
    n = len(xs)
    Xs = []
    Ys = []

    # Adjust inward space
    if inwardSpace < 0:
        inwardSpace -= tree_height

    circ_s = circStart * math.pi * 2.0
    circ = circFrac * math.pi * 2.0

    # Normalization
    min_x = xs.min()
    max_x = xs.max()
    denom = max_x - min_x

    for i in range(n):
        if is_leaf[i]:
            x = (xs[i] + inwardSpace - min_x) / denom
            y = circ_s + circ * ys[i] / y_span
            Xs.append(math.sin(y) * x)
            Ys.append(math.cos(y) * x)

    return np.array(Xs), np.array(Ys)

def extract_tree_arrays_equal_angle(tree):
    """
    Extract arrays for equal-angles layout from a Baltic tree.
    """
    n = len(tree.Objects)
    parent = np.full(n, -1, dtype=np.int32)
    children = [[] for _ in range(n)]
    branch_length = np.zeros(n, dtype=np.float64)
    is_leaf = np.zeros(n, dtype=np.bool_)
    names = []

    index_map = {k: i for i, k in enumerate(tree.Objects)}

    for i, k in enumerate(tree.Objects):
        names.append(getattr(k, "name", str(i)))
        branch_length[i] = getattr(k, "length", 1.0) or 1.0

        # Only assign parent if it's in the map
        if k.parent is not None and k.parent in index_map:
            parent[i] = index_map[k.parent]

        if k.is_leaf():
            is_leaf[i] = True
        else:
            # Only keep children that are in Objects
            children[i] = [index_map[c] for c in k.children if c in index_map]

    return parent, children, branch_length, is_leaf, names

from numba import njit

@njit
def count_leaves_numba(node, children_idx, children_ptr, is_leaf):
    """Count descendant leaves of a node (recursive)."""
    if is_leaf[node]:
        return 1
    total = 0
    for j in range(children_ptr[node], children_ptr[node+1]):
        child = children_idx[j]
        total += count_leaves_numba(child, children_idx, children_ptr, is_leaf)
    return total

@njit
def layout_equal_angle_numba(
    root,
    parent,
    children_idx,
    children_ptr,
    branch_length,
    is_leaf,
    center_x=0.0,
    center_y=0.0,
    arc_start=0.0,
    arc_stop=2*np.pi,
):
    n = len(parent)
    xs = np.zeros(n, dtype=np.float64)
    ys = np.zeros(n, dtype=np.float64)

    # We'll store edges in Python list (Numba supports append of tuples of floats)
    edges = []

    # Stack: fixed-size tuples
    stack = [(root, center_x, center_y, arc_start, arc_stop)]

    while stack:
        node, x, y, arc_s, arc_e = stack.pop()
        xs[node] = x
        ys[node] = y

        if not is_leaf[node]:
            # number of children
            n_children = children_ptr[node+1] - children_ptr[node]
            counts = np.zeros(n_children, dtype=np.int32)
            child_ids = np.zeros(n_children, dtype=np.int32)

            # compute leaf counts
            for k in range(n_children):
                child = children_idx[children_ptr[node] + k]
                child_ids[k] = child
                counts[k] = count_leaves_numba(child, children_idx, children_ptr, is_leaf)

            total_leaves = counts.sum()

            arc_size = arc_e - arc_s
            child_arc_start = arc_s
            for k in range(n_children):
                child = child_ids[k]
                c_count = counts[k]

                child_arc_size = arc_size * c_count / total_leaves
                child_arc_stop = child_arc_start + child_arc_size
                child_angle = child_arc_start + child_arc_size / 2

                dist = branch_length[child]
                child_x = x + dist * np.sin(child_angle)
                child_y = y + dist * np.cos(child_angle)

                edges.append((x, y, child_x, child_y))

                stack.append((child, child_x, child_y, child_arc_start, child_arc_stop))
                child_arc_start = child_arc_stop

    return xs, ys, edges

def layout_equal_angle_baltic(tree, center_x=0.0, center_y=0.0):
    parent, children, branch_length, is_leaf, names = extract_tree_arrays_equal_angle(tree)

    # Flatten children into CSR-like structure
    children_idx = np.array([c for sub in children for c in sub], dtype=np.int32)
    children_ptr = np.zeros(len(children)+1, dtype=np.int32)
    for i in range(len(children)):
        children_ptr[i+1] = children_ptr[i] + len(children[i])

    root = tree.Objects.index(tree.root)

    xs, ys, edges = layout_equal_angle_numba(
        root, parent, children_idx, children_ptr, branch_length, is_leaf
    )

    df_nodes = pd.DataFrame({
        "x": xs,
        "y": ys,
        "id": names,
        "is_leaf": is_leaf
    })
    df_edges = pd.DataFrame(edges, columns=["x0", "y0", "x1", "y1"])

    return df_nodes, df_edges

def plotUnrootedTree(
    tree,
    p=None,
    width=2,
    plot_width=800,
    plot_height=800,
    output_backend="webgl",
):
    """
    Plot an unrooted phylogenetic tree using the Equal Angles algorithm.
    """
    df_nodes, df_edges = layout_equal_angle_baltic(tree)

    if p is None:
        p = figure(
            width=plot_width,
            height=plot_height,
            title="Unrooted Phylogenetic Tree",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            output_backend=output_backend,
        )
        p.axis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False

    # Draw edges
    if not df_edges.empty:
        source_edges = ColumnDataSource(df_edges)
        p.segment(
            x0="x0",
            y0="y0",
            x1="x1",
            y1="y1",
            line_color="black",
            line_width=width,
            source=source_edges,
        )

    return p, df_nodes

def plotUnrootedPoints(
    tree,
    p=None,
    df_metadata=None,
    color_column=None,
    color_discrete_map=None,
    size=10,
    alpha=1,
    marker="circle",
    marker_line_color="black",
    marker_line_width=1,
    hover_data=None,
    output_backend="webgl",
    plot_width=800,
    plot_height=800,
):
    """
    Plot points on an unrooted tree with metadata coloring and hover tooltips.
    Only leaf nodes are plotted.
    """
    # Get layout
    df_nodes, df_edges = layout_equal_angle_baltic(tree)

    # Filter to leaves only
    df_leaves = df_nodes[df_nodes["is_leaf"]].copy()

    if p is None:
        p = figure(
            width=plot_width,
            height=plot_height,
            title="Unrooted Tree Points",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            output_backend=output_backend,
        )
        p.axis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False

    data, hover_data = prepare_bokeh_data(tree, hover_data, df_metadata, color_column, color_discrete_map)
    data['x'] = df_leaves["x"].values
    data['y'] = df_leaves['y'].values

    return plot_bokeh_scatter(
        p=p,
        data=data,
        hover_data=hover_data,
        size=size,
        alpha=alpha,
        marker=marker,
        marker_line_color=marker_line_color,
        marker_line_width=marker_line_width,
    )