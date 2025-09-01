import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import baltic as bt
    import baltic_bokeh as bt_bokeh
    import pandas as pd

    tree = bt.loadNewick("./examples/example.newick")
    metadata = pd.read_csv("./examples/example_metadata.tsv", sep="\t")

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
        "unassigned": "black",
    }

    # Interactive plot with metadata coloring
    p = bt_bokeh.plotTree(
        tree, metadata_df=None, color_column=None, plot_width=600, plot_height=600
    )

    p = bt_bokeh.plotPoints(
        tree,
        p=p,
        metadata_df=metadata,
        color_column="location",
        # color_discrete_map=TAXON_COLORS,
        size=15,
        plot_width=600,
        plot_height=600,
        hover_data=["species", "country"],
    )
    return metadata, p


@app.cell
def _(p):
    p
    return


@app.cell
def _(metadata):
    metadata
    return


if __name__ == "__main__":
    app.run()
