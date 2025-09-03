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
    tree = bt.loadNewick("./vgsc_focal.fasta.treefile")

    # Load metadata
    metadata = pd.read_csv("./vgsc_focal.metadata.tsv", sep="\t", index_col=0)

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
        tree, plot_width=800, plot_height=800
    )

    # Add interactive points with metadata coloring
    p = baltic_bokeh.plotCircularPoints(
        tree,
        p=p,
        df_metadata=metadata,
        color_column="taxon",
        color_discrete_map=TAXON_COLORS,
        plot_width=800,
        plot_height=800,
        hover_data=['taxon', 'country', 'year'],
    )

    # Save and show the plot
    output_file("phylogenetic_tree.html")
    show(p)


if __name__ == "__main__":
    main()
