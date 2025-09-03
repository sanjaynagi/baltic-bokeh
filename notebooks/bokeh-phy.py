import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    # import numpy as np

    # def fast_drawTree(self, order=None, width_function=None, pad_nodes=None, verbose=False):
    #     if order is None:
    #         order = self.traverse_tree(include_condition=lambda k: k.is_leaflike())
    #         if verbose:
    #             print("Drawing tree in pre-order")
    #     else:
    #         if verbose:
    #             print("Drawing tree with provided order")

    #     name_order = {x.name: i for i, x in enumerate(order)}
    #     assert len(name_order) == len(order), "Non-unique names present in tree"

    #     if width_function is None:
    #         skips = np.array([1 if x.is_leaflike() else x.width + 1 for x in order], dtype=np.float64)
    #     else:
    #         skips = np.array(list(map(width_function, order)), dtype=np.float64)

    #     heights, is_leaf, parent, child_ptrs, child_index, index_map = extract_tree_arrays_for_draw(self)
    #     order_idx = np.array([index_map[x] for x in order], dtype=np.int32)

    #     xs, ys = compute_draw_coords(heights, is_leaf, parent, child_ptrs, child_index, order_idx, skips)

    #     # Assign back
    #     for i, k in enumerate(self.Objects):
    #         k.x = xs[i]
    #         k.y = ys[i]

    #     # Handle padding (fallback to original logic)
    #     if pad_nodes is not None:
    #         for n in pad_nodes:
    #             idx = sorted([name_order[lf] for lf in n.leaves]) if n.is_node() else [order.index(n)]
    #             for i, k in enumerate(order):
    #                 if i < idx[0]:
    #                     k.y += pad_nodes[n]
    #                 if (i - 1) < idx[-1]:
    #                     k.y += pad_nodes[n]

    #         all_ys = [k.y for k in self.Objects if k.y is not None]
    #         minY = min(all_ys)
    #         for k in self.getExternal():
    #             k.y -= minY - 0.5

    #     yvalues = [k.y for k in self.Objects]
    #     self.ySpan = max(yvalues) - min(yvalues) + min(yvalues) * 2

    #     if self.root.is_node():
    #         self.root.x = min([q.x - q.length for q in self.root.children if q.x is not None])
    #         children_y_coords = [q.y for q in self.root.children if q.y is not None]
    #         self.root.y = sum(children_y_coords) / float(len(children_y_coords))
    #     else:
    #         self.root.x = self.root.length

    #     return self

    # def extract_tree_arrays_for_draw(tree):
    #     n = len(tree.Objects)
    #     heights = np.empty(n, dtype=np.float64)
    #     is_leaf = np.zeros(n, dtype=np.bool_)
    #     parent = np.full(n, -1, dtype=np.int32)
    #     index_map = {k: i for i, k in enumerate(tree.Objects)}

    #     # Collect children in flat form
    #     child_index = []
    #     child_ptrs = np.zeros(n + 1, dtype=np.int32)

    #     for i, k in enumerate(tree.Objects):
    #         heights[i] = k.height
    #         if k.is_leaflike():
    #             is_leaf[i] = True
    #         if k.parent and k.parent in index_map:
    #             parent[i] = index_map[k.parent]

    #         # record children
    #         child_ptrs[i] = len(child_index)
    #         if k.is_node():
    #             for ch in k.children:
    #                 child_index.append(index_map[ch])
    #     child_ptrs[n] = len(child_index)

    #     child_index = np.array(child_index, dtype=np.int32)

    #     return heights, is_leaf, parent, child_ptrs, child_index, index_map

    # from numba import njit
    # @njit
    # def compute_draw_coords(heights, is_leaf, parent, child_ptrs, child_index, order_idx, skips):
    #     n = len(heights)
    #     xs = np.empty(n, dtype=np.float64)
    #     ys = np.empty(n, dtype=np.float64)

    #     # Leaves first
    #     for i, node_idx in enumerate(order_idx):
    #         if is_leaf[node_idx]:
    #             xs[node_idx] = heights[node_idx]
    #             ys[node_idx] = skips[i:].sum() - skips[i] / 2.0

    #     # Internal nodes: post-order traversal
    #     for node_idx in range(n):
    #         if not is_leaf[node_idx]:
    #             start = child_ptrs[node_idx]
    #             end = child_ptrs[node_idx + 1]
    #             if end > start:
    #                 child_y = 0.0
    #                 count = 0
    #                 for j in range(start, end):
    #                     ch = child_index[j]
    #                     child_y += ys[ch]
    #                     count += 1
    #                 xs[node_idx] = heights[node_idx]
    #                 ys[node_idx] = child_y / count

    #     return xs, ys
    return


@app.cell
def _():
    # def fast_getInternal(self, secondFilter=None):
    #     if not hasattr(self, "_cached_internals"):
    #         self._cached_internals = [k for k in self.Objects if k.is_node()]
    #     if secondFilter is None:
    #         return self._cached_internals
    #     return [k for k in self._cached_internals if secondFilter(k)]

    import marimo as mo
    import baltic as bt
    # bt.tree.getInternal = fast_getInternal

    import baltic_bokeh as bt_bokeh
    # Monkey-patch
    # bt.tree.drawTree = fast_drawTree

    import pandas as pd
    import numpy as np

    # tree = bt.loadNewick("./examples/vgsc_focal.fasta.treefile")
    # metadata = pd.read_csv("./examples/vgsc_focal.metadata.tsv", sep="\t")

    tree = bt.loadNewick("./examples/example.newick")
    metadata = pd.read_csv("./examples/example_metadata.tsv", sep="\t")#[:1000]

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
        type='r',
        df_metadata=metadata,
        color_column="taxon",
        color_discrete_map=TAXON_COLORS,
        size=20,
        plot_width=600,
        plot_height=600,
        output_backend='webgl',
        marker='hex_dot',
        marker_line_color='white',
        marker_line_width=2,
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
