import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import baltic as bt
    import baltic_bokeh as bt_bokeh

    import pandas as pd
    import numpy as np

    tree = bt.loadNewick("./examples/vgsc_focal.fasta.treefile")
    metadata = pd.read_csv("./examples/vgsc_focal.metadata.tsv", sep="\t")

    # tree = bt.loadNewick("./examples/example.newick")
    # metadata = pd.read_csv("./examples/example_metadata.tsv", sep="\t")#[:1000]

    import plotly.express as px

    TAXON_PALETTE = px.colors.qualitative.Vivid
    TAXON_COLORS = {
        "gambiae": TAXON_PALETTE[1],
        "coluzzii": TAXON_PALETTE[0],
        "arabiensis": TAXON_PALETTE[2],
        "merus": TAXON_PALETTE[3],
        "melas": TAXON_PALETTE[4],
        "quadriannulatus": TAXON_PALETTE[5],
        "fontenillei": TAXON_PALETTE[6],
        "gcx1": TAXON_PALETTE[7],
        "gcx2": TAXON_PALETTE[8],
        "gcx3": TAXON_PALETTE[9],
        "gcx4": TAXON_PALETTE[10],
        "bissau":TAXON_PALETTE[7],
        "pwani":TAXON_PALETTE[9],
        "unassigned": "black",
    }
    return TAXON_COLORS, bt_bokeh, metadata, tree


@app.cell
def _(TAXON_COLORS, bt_bokeh, metadata, tree):
    p = bt_bokeh.plotTree(
        tree,
        type='u',
        df_metadata=metadata,
        color_column="taxon",
        color_discrete_map=TAXON_COLORS,
        size=5,
        plot_width=1200,
        plot_height=1200,
        output_backend='webgl',
        marker='hex',
        marker_line_color='white',
        marker_line_width=0,
        hover_data=['taxon', 'country']
    )
    return (p,)


@app.cell
def _(p):
    p
    return


@app.cell
def _():
    # 750 secs for webgl - full focal vgsc

    # 0.05 secs for 200 sample vgsc , webgl and canvas

    # 0.48 secs canvas 1000, 0.39 webgl
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
