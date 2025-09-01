#!/usr/bin/env python3
"""
Example usage of baltic-bokeh for interactive phylogenetic tree visualization.
"""

import baltic as bt
import baltic_bokeh
import pandas as pd
from bokeh.plotting import show, output_file


def main():
    # Load tree file
    tree = bt.loadNewick("./example.newick")

    # Load metadata
    metadata = pd.read_csv("./example_metadata.tsv", sep="\t", index_col=0)

    # Define color palette
    TAXON_COLORS = {
        "gambiae": "#1f77b4",
        "coluzzii": "#ff7f0e",
        "arabiensis": "#2ca02c",
        "merus": "#d62728",
        "melas": "#9467bd",
        "unassigned": "black",
    }

    # Create interactive circular tree plot
    p = baltic_bokeh.plotCircularTree(
        tree, metadata_df=None, color_column=None, plot_width=800, plot_height=800
    )

    # Add interactive points with metadata coloring
    p = baltic_bokeh.plotCircularPoints(
        tree,
        p=p,
        metadata_df=metadata,
        color_column="species",
        color_discrete_map=TAXON_COLORS,
        plot_width=800,
        plot_height=800,
        hover_data=metadata.columns.tolist(),
    )

    # Save and show the plot
    output_file("phylogenetic_tree.html")
    show(p)


if __name__ == "__main__":
    main()
