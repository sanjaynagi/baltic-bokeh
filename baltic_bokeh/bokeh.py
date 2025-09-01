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


def _prepare_metadata_colors(tree_obj, metadata_df, color_column, color_discrete_map=None, target=None):
    """
    Helper function to prepare colors based on metadata DataFrame.
    
    Parameters:
    tree_obj: The tree object
    metadata_df (pd.DataFrame): DataFrame with metadata, indexed by sample names
    color_column (str): Column name to use for coloring
    color_discrete_map (dict or None): Custom color mapping {value: color}
    target (function or None): Function to filter which objects to color
    
    Returns:
    dict: Mapping from tree objects to colors
    """
    if target is None:
        target = lambda k: k.is_leaf()
    
    color_map = {}
    
    # Get unique values in the color column
    if metadata_df is not None and color_column in metadata_df.columns:
        unique_values = metadata_df[color_column].dropna().unique()
        
        # Use provided color map or create default one
        if color_discrete_map:
            value_to_color = color_discrete_map
        else:
            # Use bokeh palettes for default colors
            if len(unique_values) <= 3:
                palette = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Basic colors for small sets
            elif len(unique_values) <= 10:
                palette = Category10[max(3, len(unique_values))]
            else:
                palette = Set3[12]
            value_to_color = {val: palette[i % len(palette)] for i, val in enumerate(unique_values)}
        
        # Map tree objects to colors based on metadata
        for k in filter(target, tree_obj.Objects):
            if not hasattr(k, 'name') or k.name is None:
                color_map[k] = 'gray'  # Default for objects without names (internal nodes)
                continue
            sample_name = k.name
            if sample_name in metadata_df.index:
                value = metadata_df.loc[sample_name, color_column]
                if pd.notna(value) and value in value_to_color:
                    color_map[k] = value_to_color[value]
                else:
                    color_map[k] = 'gray'  # Default for missing values
            else:
                color_map[k] = 'gray'  # Default for samples not in metadata
    else:
        # No metadata or column provided, use default color
        for k in filter(target, tree_obj.Objects):
            color_map[k] = 'black'
    
    return color_map


def plotTree(tree_obj, p=None, connection_type=None, target=None, x_attr=None, y_attr=None, 
             width=None, colour=None, metadata_df=None, color_column=None, color_discrete_map=None,
             plot_width=800, plot_height=600, **kwargs):
    """
    Plot the tree using Bokeh with interactive features and metadata support.
    
    Parameters:
    tree_obj: The tree object
    p (bokeh.plotting.figure or None): Bokeh figure to plot on. If None, creates new figure.
    connection_type (str or None): Connection type ('baltic', 'direct', 'elbow'). Default 'baltic'.
    target (function or None): Function to select branches to plot. Default plots all.
    x_attr (function or None): Function for x-coordinates. Default uses branch.x.
    y_attr (function or None): Function for y-coordinates. Default uses branch.y.
    width (int or function or None): Line width. Default 2.
    colour (str or function or None): Line color. Default 'black'.
    metadata_df (pd.DataFrame or None): DataFrame with metadata for coloring
    color_column (str or None): Column name in metadata_df for coloring
    color_discrete_map (dict or None): Custom color mapping {value: color}
    plot_width (int): Width of the plot if creating new figure
    plot_height (int): Height of the plot if creating new figure
    **kwargs: Additional arguments
    
    Returns:
    bokeh.plotting.figure: The bokeh figure with tree plot
    """
    if p is None:
        p = figure(width=plot_width, height=plot_height, 
                  title="Phylogenetic Tree", tools="pan,wheel_zoom,box_zoom,reset,save")
    
    if target is None: target = lambda k: True
    if x_attr is None: x_attr = lambda k: k.x
    if y_attr is None: y_attr = lambda k: k.y
    if width is None: width = 2
    if connection_type is None: connection_type = 'baltic'
    
    assert connection_type in ['baltic', 'direct', 'elbow'], f'Unrecognised drawing type "{connection_type}"'
    
    # Prepare colors from metadata if provided
    if metadata_df is not None and color_column is not None:
        # Use leaf-only target for metadata coloring
        metadata_target = lambda k: k.is_leaf() 
        color_map = _prepare_metadata_colors(tree_obj, metadata_df, color_column, color_discrete_map, metadata_target)
        colour = lambda k: color_map.get(k, 'black')
    elif colour is None:
        colour = 'black'
    
    # Collect line segments
    xs = []
    ys = []
    colors = []
    
    for k in filter(target, tree_obj.Objects):
        x = x_attr(k)
        xp = x_attr(k.parent) if k.parent else x
        y = y_attr(k)
        
        try:
            color = colour(k) if callable(colour) else colour
        except (KeyError, AttributeError):
            color = 'gray'
        
        if connection_type == 'baltic':
            # Horizontal branch to parent
            xs.extend([xp, x])
            ys.extend([y, y])
            colors.extend([color, color])
            
            # Vertical connector for internal nodes
            if k.is_node():
                yl, yr = y_attr(k.children[0]), y_attr(k.children[-1])
                xs.extend([x, x])
                ys.extend([yl, yr])
                colors.extend([color, color])
                
        elif connection_type == 'direct':
            yp = y_attr(k.parent) if k.parent else y
            xs.extend([xp, x])
            ys.extend([yp, y])
            colors.extend([color, color])
            
        elif connection_type == 'elbow':
            yp = y_attr(k.parent) if k.parent else y
            # Create elbow connection: parent_x,parent_y -> parent_x,child_y -> child_x,child_y
            xs.extend([xp, xp, x])
            ys.extend([yp, y, y])
            colors.extend([color, color, color])
    
    # Plot line segments
    if xs and ys:
        # For bokeh, we need to plot line segments differently
        # Group consecutive points of same color and plot as multi_line
        line_data = {'xs': [], 'ys': [], 'colors': []}
        
        i = 0
        while i < len(xs) - 1:
            current_color = colors[i]
            line_xs = [xs[i], xs[i+1]]
            line_ys = [ys[i], ys[i+1]]
            
            line_data['xs'].append(line_xs)
            line_data['ys'].append(line_ys)
            line_data['colors'].append(current_color)
            i += 2
        
        if line_data['xs']:
            source = ColumnDataSource(line_data)
            p.multi_line('xs', 'ys', color='colors', line_width=width, source=source)
    
    return p


def plotPoints(tree_obj, p=None, x_attr=None, y_attr=None, target=None, size=None, colour=None,
               metadata_df=None, color_column=None, color_discrete_map=None,
               plot_width=800, plot_height=600, hover_data=None, **kwargs):
    """
    Plot points on the tree with interactive features and metadata support.
    
    Parameters:
    tree_obj: The tree object
    p (bokeh.plotting.figure or None): Bokeh figure to plot on. If None, creates new figure.
    x_attr (function or None): Function for x-coordinates. Default uses branch.x.
    y_attr (function or None): Function for y-coordinates. Default uses branch.y.
    target (function or None): Function to select branches. Default selects leaves.
    size (int or function or None): Point size. Default 8.
    colour (str or function or None): Point color. Default 'black'.
    metadata_df (pd.DataFrame or None): DataFrame with metadata for coloring
    color_column (str or None): Column name in metadata_df for coloring
    color_discrete_map (dict or None): Custom color mapping {value: color}
    plot_width (int): Width of the plot if creating new figure
    plot_height (int): Height of the plot if creating new figure
    hover_data (list or None): Additional columns from metadata to show on hover
    **kwargs: Additional arguments
    
    Returns:
    bokeh.plotting.figure: The bokeh figure with points
    """
    if p is None:
        p = figure(width=plot_width, height=plot_height,
                  title="Phylogenetic Tree Points", tools="pan,wheel_zoom,box_zoom,reset,save")
    
    if target is None: target = lambda k: k.is_leaf()
    if x_attr is None: x_attr = lambda k: k.x
    if y_attr is None: y_attr = lambda k: k.y
    if size is None: size = 8
    
    # Prepare colors from metadata if provided
    if metadata_df is not None and color_column is not None:
        color_map = _prepare_metadata_colors(tree_obj, metadata_df, color_column, color_discrete_map, target)
        colour = lambda k: color_map.get(k, 'gray')
    elif colour is None:
        colour = 'black'
    
    # Collect point data
    data = {'x': [], 'y': [], 'colors': [], 'names': []}
    
    # Add hover data columns if provided
    if hover_data and metadata_df is not None:
        for col in hover_data:
            if col in metadata_df.columns:
                data[col] = []
    
    for k in filter(target, tree_obj.Objects):
        data['x'].append(x_attr(k))
        data['y'].append(y_attr(k))
        data['names'].append(k.name)
        
        try:
            color = colour(k) if callable(colour) else colour
        except (KeyError, AttributeError):
            color = 'gray'
        data['colors'].append(color)
        
        # Add hover data
        if hover_data and metadata_df is not None:
            for col in hover_data:
                if col in metadata_df.columns and k.name in metadata_df.index:
                    value = metadata_df.loc[k.name, col]
                    data[col].append(str(value) if pd.notna(value) else 'N/A')
                elif col in data:
                    data[col].append('N/A')
    
    if data['x']:
        source = ColumnDataSource(data)
        
        # Create hover tool
        hover_tooltips = [("Name", "@names")]
        if color_column and metadata_df is not None:
            hover_tooltips.append((color_column, f"@{color_column}" if color_column in data else "N/A"))
        if hover_data:
            for col in hover_data:
                if col in data:
                    hover_tooltips.append((col, f"@{col}"))
        
        hover = HoverTool(tooltips=hover_tooltips)
        p.add_tools(hover)
        
        # Plot points
        point_size = size if callable(size) else size
        p.scatter('x', 'y', size=point_size, color='colors', source=source, alpha=0.8)
    
    return p


def plotCircularTree(tree_obj, p=None, target=None, x_attr=None, y_attr=None, width=None, colour=None,
                     circStart=0.0, circFrac=1.0, inwardSpace=0.0, normaliseHeight=None, precision=15,
                     metadata_df=None, color_column=None, color_discrete_map=None,
                     plot_width=800, plot_height=800, **kwargs):
    """
    Plot the tree in a circular layout using Bokeh.
    
    Parameters:
    tree_obj: The tree object
    p (bokeh.plotting.figure or None): Bokeh figure to plot on. If None, creates new figure.
    target (function or None): Function to select branches to plot. Default plots all.
    x_attr (function or None): Function for x-coordinates. Default uses branch.x.
    y_attr (function or None): Function for y-coordinates. Default uses branch.y.
    width (int or function or None): Line width. Default 2.
    colour (str or function or None): Line color. Default 'black'.
    circStart (float): Starting angle in fractions of 2*pi. Default 0.0.
    circFrac (float): Fraction of full circle to use. Default 1.0.
    inwardSpace (float): Space to leave in middle. Default 0.0.
    normaliseHeight (function or None): Height normalization function.
    precision (int): Number of points for curved segments. Default 15.
    metadata_df (pd.DataFrame or None): DataFrame with metadata for coloring
    color_column (str or None): Column name in metadata_df for coloring
    color_discrete_map (dict or None): Custom color mapping {value: color}
    plot_width (int): Width of the plot if creating new figure
    plot_height (int): Height of the plot if creating new figure
    **kwargs: Additional arguments
    
    Returns:
    bokeh.plotting.figure: The bokeh figure with circular tree
    """
    if p is None:
        p = figure(width=plot_width, height=plot_height,
                  title="Circular Phylogenetic Tree", tools="pan,wheel_zoom,box_zoom,reset,save")
        p.axis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False
    
    if target is None: target = lambda k: True
    if x_attr is None: x_attr = lambda k: k.x
    if y_attr is None: y_attr = lambda k: k.y
    if colour is None: colour = 'black'
    if width is None: width = 2
    
    if inwardSpace < 0: inwardSpace -= tree_obj.treeHeight
    
    # Prepare colors from metadata if provided
    if metadata_df is not None and color_column is not None:
        color_map = _prepare_metadata_colors(tree_obj, metadata_df, color_column, color_discrete_map, target)
        colour = lambda k: color_map.get(k, 'gray')
    
    circ_s = circStart * math.pi * 2
    circ = circFrac * math.pi * 2
    
    allXs = list(map(x_attr, tree_obj.Objects))
    if normaliseHeight is None: 
        normaliseHeight = lambda value: (value - min(allXs)) / (max(allXs) - min(allXs))
    
    linspace = lambda start, stop, n: list(start + ((stop - start) / (n - 1)) * i for i in range(n)) if n > 1 else [stop]
    
    # Collect line segments for circular tree
    line_data = {'xs': [], 'ys': [], 'colors': []}
    
    for k in filter(target, tree_obj.Objects):
        x = normaliseHeight(x_attr(k) + inwardSpace)
        xp = normaliseHeight(x_attr(k.parent) + inwardSpace) if k.parent and k.parent.parent else x
        y = y_attr(k)
        
        try:
            color = colour(k) if callable(colour) else colour
        except (KeyError, AttributeError):
            color = 'gray'
        
        y = circ_s + circ * y / tree_obj.ySpan
        X = math.sin(y)
        Y = math.cos(y)
        
        # Radial branch
        line_data['xs'].append([X * xp, X * x])
        line_data['ys'].append([Y * xp, Y * x])
        line_data['colors'].append(color)
        
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
                line_data['xs'].append([xs[i], xs[i + 1]])
                line_data['ys'].append([ys[i], ys[i + 1]])
                line_data['colors'].append(color)
    
    if line_data['xs']:
        source = ColumnDataSource(line_data)
        p.multi_line('xs', 'ys', color='colors', line_width=width, source=source)
    
    return p


def plotCircularPoints(tree_obj, p=None, x_attr=None, y_attr=None, target=None, size=None, colour=None,
                       circStart=0.0, circFrac=1.0, inwardSpace=0.0, normaliseHeight=None,
                       metadata_df=None, color_column=None, color_discrete_map=None,
                       plot_width=800, plot_height=800, hover_data=None, **kwargs):
    """
    Plot points on a circular tree with interactive features and metadata support.
    """
    if p is None:
        p = figure(width=plot_width, height=plot_height,
                  title="Circular Tree Points", tools="pan,wheel_zoom,box_zoom,reset,save")
        p.axis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False
    
    if target is None: target = lambda k: k.is_leaf()
    if x_attr is None: x_attr = lambda k: k.x
    if y_attr is None: y_attr = lambda k: k.y
    if size is None: size = 8
    
    if inwardSpace < 0: inwardSpace -= tree_obj.treeHeight
    
    # Prepare colors from metadata if provided
    if metadata_df is not None and color_column is not None:
        color_map = _prepare_metadata_colors(tree_obj, metadata_df, color_column, color_discrete_map, target)
        colour = lambda k: color_map.get(k, 'gray')
    elif colour is None:
        colour = 'black'
    
    circ_s = circStart * math.pi * 2
    circ = circFrac * math.pi * 2
    
    allXs = list(map(x_attr, tree_obj.Objects))
    if normaliseHeight is None:
        normaliseHeight = lambda value: (value - min(allXs)) / (max(allXs) - min(allXs))
    
    # Collect point data
    data = {'x': [], 'y': [], 'colors': [], 'names': []}
    
    # Add hover data columns if provided
    if hover_data and metadata_df is not None:
        for col in hover_data:
            if col in metadata_df.columns:
                data[col] = []
    
    for k in filter(target, tree_obj.Objects):
        x = normaliseHeight(x_attr(k) + inwardSpace)
        y = circ_s + circ * y_attr(k) / tree_obj.ySpan
        X = math.sin(y) * x
        Y = math.cos(y) * x
        
        data['x'].append(X)
        data['y'].append(Y)
        data['names'].append(k.name)
        
        try:
            color = colour(k) if callable(colour) else colour
        except (KeyError, AttributeError):
            color = 'gray'
        data['colors'].append(color)
        
        # Add hover data
        if hover_data and metadata_df is not None:
            for col in hover_data:
                if col in metadata_df.columns and k.name in metadata_df.index:
                    value = metadata_df.loc[k.name, col]
                    data[col].append(str(value) if pd.notna(value) else 'N/A')
                elif col in data:
                    data[col].append('N/A')
    
    if data['x']:
        source = ColumnDataSource(data)
        
        # Create hover tool
        hover_tooltips = [("Name", "@names")]
        if color_column and metadata_df is not None:
            hover_tooltips.append((color_column, f"@{color_column}" if color_column in data else "N/A"))
        if hover_data:
            for col in hover_data:
                if col in data:
                    hover_tooltips.append((col, f"@{col}"))
        
        hover = HoverTool(tooltips=hover_tooltips)
        p.add_tools(hover)
        
        # Plot points
        point_size = size if callable(size) else size
        p.scatter('x', 'y', size=point_size, color='colors', source=source, alpha=0.8)
    
    return p


def addText(tree_obj, p, target=None, x_attr=None, y_attr=None, text=None, **kwargs):
    """
    Add text annotations to a bokeh tree plot.
    
    Note: Bokeh text rendering is different from matplotlib. 
    This is a basic implementation that adds text labels.
    """
    if target is None: target = lambda k: k.is_leaf()
    if x_attr is None: x_attr = lambda k: k.x
    if y_attr is None: y_attr = lambda k: k.y
    if text is None: text = lambda k: k.name
    
    data = {'x': [], 'y': [], 'text': []}
    
    for k in filter(target, tree_obj.Objects):
        data['x'].append(x_attr(k))
        data['y'].append(y_attr(k))
        data['text'].append(text(k))
    
    if data['x']:
        source = ColumnDataSource(data)
        p.text('x', 'y', text='text', source=source, text_font_size="8pt")
    
    return p