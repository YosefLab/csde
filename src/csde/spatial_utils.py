from __future__ import annotations

from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _micron_to_global(sdata, xmin: float, ymin: float, xmax: float, ymax: float):
    """Convert bounding-box corners from micron (intrinsic) to global (pixel) space."""
    from spatialdata.transformations import get_transformation

    shapes_key = list(sdata.shapes.keys())[0]
    t = get_transformation(sdata.shapes[shapes_key], to_coordinate_system="global")
    m = t.matrix  # 3×3 affine; for MERSCOPE this is a pure scale+translation

    # Transform all four corners and take the bounding box (handles negative scales).
    xs = [m[0, 0] * x + m[0, 2] for x in (xmin, xmax)]
    ys = [m[1, 1] * y + m[1, 2] for y in (ymin, ymax)]
    return min(xs), min(ys), max(xs), max(ys)


def plot_region(
    sdata,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    gene_colors: dict[str, str],
    image_channel: str,
    coord_system: Literal["global", "micron"] = "global",
    image_cmap: str = "gray",
    outline_color: str = "white",
    point_size: int = 2,
    figsize: tuple = (14, 14),
):
    """
    Plot a spatial region: fluorescence image + cell boundaries + transcript dots.

    Parameters
    ----------
    sdata
        SpatialData object.
    xmin, ymin, xmax, ymax
        Bounding-box corners.  Interpreted according to ``coord_system``.
    gene_colors
        Mapping of gene name → hex/named colour.  Only these genes are shown
        as transcript dots, using their assigned colours.
    image_channel
        Fluorescence channel name to use as background (e.g. ``"DAPI"``,
        ``"Cellbound1"``).
    coord_system
        ``"global"``  – coordinates are in mosaic-pixel space (the SpatialData
        global coordinate system).
        ``"micron"``  – coordinates are in physical microns, i.e. the same units
        as ``center_x`` / ``center_y`` stored in ``adata.obs``.  They are
        converted to pixel space before querying.
    image_cmap
        Colormap for the fluorescence background.
    outline_color
        Colour for cell-boundary outlines.
    point_size
        Size of transcript dots.
    figsize
        Figure size passed to matplotlib.
    """
    import spatialdata_plot  # noqa: F401 — registers the .pl accessor on SpatialData

    shapes_key = list(sdata.shapes.keys())[0]
    image_key = list(sdata.images.keys())[0]
    points_key = list(sdata.points.keys())[0]

    if coord_system == "micron":
        xmin, ymin, xmax, ymax = _micron_to_global(sdata, xmin, ymin, xmax, ymax)
    elif coord_system != "global":
        raise ValueError(f"coord_system must be 'global' or 'micron', got {coord_system!r}")

    cropped = sdata.query.bounding_box(
        min_coordinate=[xmin, ymin],
        max_coordinate=[xmax, ymax],
        axes=("x", "y"),
        target_coordinate_system="global",
    )

    genes = list(gene_colors.keys())
    palette = list(gene_colors.values())

    renderer = (
        cropped.pl
        .render_images(image_key, channel=image_channel, cmap=image_cmap)
        .pl.render_shapes(
            shapes_key,
            fill_alpha=0.0,
            outline_alpha=0.8,
            outline_color=outline_color,
            outline_width=0.5,
        )
    )
    if genes:
        renderer = renderer.pl.render_points(
            points_key,
            color="gene",
            groups=genes,
            palette=palette,
            size=point_size,
            alpha=0.9,
        )
    ax = renderer.pl.show(
        figsize=figsize,
        title=f"Region [{xmin:.0f}-{xmax:.0f}, {ymin:.0f}-{ymax:.0f}]",
        return_ax=True,
    )
    return ax


def compute_importance_weights(
    adata,
    cell_type_key: str,
    cell_type_of_interest: str,
    target_proportion: float,
) -> np.ndarray:
    """
    Return unnormalized per-cell weights so that after sampling the fraction of
    `cell_type_of_interest` cells equals `target_proportion`.

    Cells of interest receive weight w; all others receive weight 1.
    Derived from: n_int * w / (n_int * w + n_other) = target_proportion.
    """
    is_interest = adata.obs[cell_type_key] == cell_type_of_interest
    n_interest = int(is_interest.sum())
    n_other = int((~is_interest).sum())

    if n_interest == 0:
        return np.ones(len(adata))
    if not (0.0 < target_proportion < 1.0):
        raise ValueError("target_proportion must be in (0, 1)")

    w = target_proportion * n_other / (n_interest * (1.0 - target_proportion))
    return np.where(is_interest.values, w, 1.0)


def subsample_cells(
    adata,
    cell_type_key: str,
    cell_type_of_interest: str,
    target_proportion: float,
    n_cells: int,
    seed: int = 0,
):
    """
    Sample `n_cells` from `adata` without replacement using importance weights
    that target `target_proportion` cells of `cell_type_of_interest`.

    The unnormalized sampling weight is stored in ``obs["sampling_weight"]``
    of the returned AnnData (needed downstream for weighted estimation).
    """
    weights = compute_importance_weights(
        adata, cell_type_key, cell_type_of_interest, target_proportion
    )
    norm_weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(adata), size=n_cells, p=norm_weights, replace=False)
    sub = adata[indices].copy()
    sub.obs["sampling_weight"] = weights[indices]
    return sub


def plot_top_genes(
    adata_cell,
    n_genes: int = 15,
    ax=None,
    layer: str | None = None,
    title: str | None = None,
    gene_colors: dict[str, str] | None = None,
):
    """
    Horizontal bar chart of the top `n_genes` expressed genes for a single cell.

    Parameters
    ----------
    adata_cell
        AnnData slice for one cell (shape 1 × n_vars).
    n_genes
        Number of top genes to display.
    ax
        Existing matplotlib axes; created if None.
    layer
        Layer to use for expression values; falls back to X if absent or None.
    title
        Axes title.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    if layer is not None and layer in adata_cell.layers:
        x = adata_cell.layers[layer]
    else:
        x = adata_cell.X

    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x).flatten()

    expr = (
        pd.Series(x, index=adata_cell.var_names)
        .sort_values(ascending=False)
        .head(n_genes)
        .iloc[::-1]  # highest bar at the top
    )

    colors = (
        [gene_colors.get(g, "grey") for g in expr.index]
        if gene_colors
        else ["steelblue"] * len(expr)
    )
    ax.barh(expr.index, expr.values, color=colors)
    ax.set_xlabel("counts")
    ax.tick_params(axis="y", labelsize=8)
    if title is not None:
        ax.set_title(title)
    return ax
