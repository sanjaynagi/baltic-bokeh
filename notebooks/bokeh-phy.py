import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import baltic as bt
    import baltic_bokeh as bt_bokeh
    import pandas as pd

    tree = bt.loadNewick("./examples/vgsc_focal.fasta.treefile")
    metadata = pd.read_csv("./examples/vgsc_focal.metadata.tsv", sep="\t")

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

    # Interactive plot with metadata coloring
    p = bt_bokeh.plotCircular(
        tree,
        df_metadata=metadata,
        color_column="taxon",
        color_discrete_map=TAXON_COLORS,
        size=5,
        plot_width=1200,
        plot_height=1200,
        hover_data=["sample_id", "taxon", "country"],
    )
    return


app._unparsable_cell(
    r"""
     p
    """,
    name="_"
)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
