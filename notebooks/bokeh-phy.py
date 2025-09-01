import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import baltic as bt
    from baltic import bokeh as bt_bokeh
    import pandas as pd

    tree = bt.loadNewick('./vgsc_focal.fasta.treefile')
    metadata = pd.read_csv('./vgsc-metadata.tsv', sep="\t")

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
        "unassigned": "black"
    }

    # Interactive plot with metadata coloring
    p = bt_bokeh.plotCircularTree(tree,
                        metadata_df=None,
                        color_column=None,
                        plot_width=1000, plot_height=1000
    )

    p = bt_bokeh.plotCircularPoints(tree, p=p,
                          metadata_df=metadata,
                          color_column='taxon',
                          color_discrete_map=TAXON_COLORS,
                          plot_width=1000, plot_height=1000,
                          hover_data=metadata.columns.to_list()
    )

    return (p,)


@app.cell
def _(p):
    p
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
