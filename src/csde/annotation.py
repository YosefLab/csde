from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .spatial_utils import plot_region, plot_top_genes, subsample_cells


def prepare_csde_inputs(
    annotation_dir: str | Path,
    sdata=None,
    spatial_group_key: str = "spatial_group",
    spatial_group_target=1,
    spatial_group_reference=0,
    layer: str | None = None,
    n_cells_expressed_threshold: int = 10,
) -> dict:
    """
    Build adata_gt and adata_other for run_csde() from a completed annotation directory.

    Reads config.json (cell_type_key, cell_type_of_interest, sdata path),
    metadata.csv (sampling_weight per annotated cell), and annotations.json
    (is_correct per cell) from annotation_dir.

    Label encoding in .obs["prediction"] / .obs["annotation"]:
        1 — cell_type_of_interest with spatial_group == 1  (target)
        0 — cell_type_of_interest with spatial_group == 0  (reference)
        2 — all other cells

    For GT annotation labels, cells predicted as cell_type_of_interest but
    marked incorrect (is_correct=False) are reassigned to class 2.

    Parameters
    ----------
    annotation_dir
        Directory produced by scripts/export.py and scripts/annotate.py.
    sdata
        Already-loaded SpatialData object. If None, loaded from the path
        stored in config.json.
    spatial_group_key
        obs column encoding the two spatial populations.
    spatial_group_target
        Value in spatial_group_key that identifies the target population (label 1).
    spatial_group_reference
        Value in spatial_group_key that identifies the reference population (label 0).
    layer
        AnnData layer to use for expression counts. Defaults to None (uses .X).
    n_cells_expressed_threshold
        A gene is kept only if it is expressed (count >= 1) in at least this
        many annotated pred-target/reference cells.

    Returns
    -------
    dict with keys:

    adata_gt : AnnData
        Annotated cells. obs columns added: ``prediction`` (int 0/1/2),
        ``annotation`` (int 0/1/2), ``is_correct`` (bool),
        ``sampling_weight`` (float). Genes are filtered.
    adata_other : AnnData
        All unannotated cells. obs column added: ``prediction`` (int 0/1/2).
        Same gene set as adata_gt.
    """
    import numpy as np

    annotation_dir = Path(annotation_dir)

    with open(annotation_dir / "config.json") as f:
        config = json.load(f)
    cell_type_key = config["cell_type_key"]
    cell_type_of_interest = config["cell_type_of_interest"]

    ann_path = annotation_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"No annotations found at {ann_path}. Run scripts/annotate.py first."
        )
    with open(ann_path) as f:
        annotations = json.load(f)  # {cell_id: True/False}

    metadata = pd.read_csv(annotation_dir / "metadata.csv")
    metadata["cell_id"] = metadata["cell_id"].astype(str)
    sampling_weights = metadata.set_index("cell_id")["sampling_weight"]

    if sdata is None:
        import spatialdata as sd
        sdata = sd.read_zarr(config["sdata"])
    adata = sdata["table"].copy()
    adata.obs_names = adata.obs_names.astype(str)
    adata = adata[adata.obs[cell_type_key].notna()].copy()

    # --- Prediction labels (automated, all cells) ---
    is_coi = (adata.obs[cell_type_key] == cell_type_of_interest).values
    spatial_group = adata.obs[spatial_group_key].values

    prediction = np.full(len(adata), 2, dtype=int)
    prediction[is_coi & (spatial_group == spatial_group_target)] = 1    # target
    prediction[is_coi & (spatial_group == spatial_group_reference)] = 0  # reference
    adata.obs["prediction"] = prediction

    # --- Split annotated / unannotated ---
    annotated_ids = set(annotations.keys())
    annotated_mask = adata.obs_names.isin(annotated_ids)

    adata_gt = adata[annotated_mask].copy()

    is_correct_arr = np.array(
        [annotations[cid] for cid in adata_gt.obs_names], dtype=bool
    )
    adata_gt.obs["is_correct"] = is_correct_arr

    # --- Annotation (GT) labels ---
    is_coi_gt = (adata_gt.obs[cell_type_key] == cell_type_of_interest).values
    spatial_group_gt = adata_gt.obs[spatial_group_key].values

    annotation = np.full(len(adata_gt), 2, dtype=int)
    annotation[is_coi_gt & is_correct_arr & (spatial_group_gt == spatial_group_target)] = 1
    annotation[is_coi_gt & is_correct_arr & (spatial_group_gt == spatial_group_reference)] = 0
    adata_gt.obs["annotation"] = annotation

    adata_gt.obs["sampling_weight"] = adata_gt.obs_names.map(sampling_weights).values

    # --- Gene filter: expressed in >= threshold pred-target/ref cells in adata_gt ---
    # pred_mask = adata_gt.obs["prediction"].isin([0, 1])
    pred_mask = adata_gt.obs["annotation"].isin([0, 1])
    _sub = adata_gt[pred_mask]
    x = _sub.layers[layer] if layer is not None else _sub.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = x.astype(float)
    n_expressing = np.array((x >= 1).sum(0)).flatten()
    gene_mask = n_expressing >= n_cells_expressed_threshold

    adata_gt = adata_gt[:, gene_mask].copy()
    adata_other = adata[~annotated_mask][:, gene_mask].copy()

    return {"adata_gt": adata_gt, "adata_other": adata_other}


def load_annotations(annotation_dir: str | Path) -> pd.DataFrame:
    """
    Merge ``metadata.csv`` and ``annotations.json`` into a single DataFrame.

    Returns only annotated cells, with an added boolean ``is_correct`` column.
    Pass the result as ``adata_gt`` to :func:`~csde.run_csde`.
    """
    annotation_dir = Path(annotation_dir)
    metadata = pd.read_csv(annotation_dir / "metadata.csv")
    metadata["cell_id"] = metadata["cell_id"].astype(str)

    ann_path = annotation_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"No annotations found at {ann_path}. Run scripts/annotate.py first."
        )
    with open(ann_path) as f:
        annotations = json.load(f)

    metadata["is_correct"] = metadata["cell_id"].map(annotations)
    return metadata[metadata["is_correct"].notna()].copy()


def export_cell_panels(
    sdata,
    annotation_dir: str | Path,
    cell_type_key: str,
    cell_type_of_interest: str,
    target_proportion: float,
    n_cells: int = 600,
    image_channel: str = "DAPI",
    delta: float = 50.0,
    n_top_genes: int = 15,
    layer: str | None = None,
    gene_colors: dict[str, str] | None = None,
    seed: int = 0,
    dpi: int = 150,
) -> pd.DataFrame:
    """
    Subsample cells from a SpatialData object and export per-cell annotation panels.

    Each panel is a side-by-side figure:
      left  — spatial context: fluorescence image + cell boundaries + top-gene transcripts
      right — horizontal bar chart of the cell's top expressed genes

    A ``metadata.csv`` is written to ``annotation_dir`` with one row per exported cell,
    containing: cell_id, cell_type, image_path, sampling_weight, center_x, center_y.

    Parameters
    ----------
    sdata
        SpatialData object.  Must have a ``"table"`` AnnData whose obs contains
        ``cell_type_key``, ``center_x``, and ``center_y``.
    annotation_dir
        Root output directory.  Images are written to ``annotation_dir/images/``.
    cell_type_key
        obs column that holds cell-type labels.
    cell_type_of_interest
        Cell type to oversample (e.g. ``"macrophages"``).
    target_proportion
        Desired fraction of ``cell_type_of_interest`` cells in the subsample, in (0, 1).
    n_cells
        Total number of cells to sample.
    image_channel
        Fluorescence channel used as image background (e.g. ``"DAPI"``, ``"Cellbound2"``).
    delta
        Half-width of the bounding box around each cell center, in microns.
    n_top_genes
        Number of top-expressed genes shown in the right panel.
    layer
        AnnData layer to use for expression values; falls back to X when absent or None.
    gene_colors
        Mapping of gene name → colour.  Left panel shows only these genes (with their
        colours); right panel uses the colour for keyed genes and grey for the rest.
    seed
        Random seed for reproducibility.
    dpi
        Resolution of saved images.

    Returns
    -------
    pd.DataFrame
        The metadata table also written to ``annotation_dir/metadata.csv``.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **_):
            return it

    annotation_dir = Path(annotation_dir)
    images_dir = annotation_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    adata = sdata["table"]
    adata = adata[adata.obs[cell_type_key].notna()].copy()
    sub = subsample_cells(
        adata,
        cell_type_key=cell_type_key,
        cell_type_of_interest=cell_type_of_interest,
        target_proportion=target_proportion,
        n_cells=n_cells,
        seed=seed,
    )

    metadata_rows = []
    for cell_id in tqdm(sub.obs_names, desc="Exporting cells"):
        obs = sub.obs.loc[cell_id]
        cx, cy = float(obs["center_x"]), float(obs["center_y"])
        cell_type = obs[cell_type_key]

        cell_adata = sub[[cell_id]]

        # --- left panel: spatial context ---
        ax_left = plot_region(
            sdata,
            xmin=cx - delta,
            xmax=cx + delta,
            ymin=cy - delta,
            ymax=cy + delta,
            gene_colors=gene_colors or {},
            image_channel=image_channel,
            coord_system="micron",
            figsize=(7, 7),
        )
        buf = io.BytesIO()
        ax_left.figure.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        plt.close(ax_left.figure)

        img_left = plt.imread(buf)

        # --- combined figure ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].imshow(img_left)
        axes[0].axis("off")
        axes[0].set_title(f"{cell_type}  |  {cell_id}", fontsize=8)

        plot_top_genes(
            cell_adata,
            n_genes=n_top_genes,
            ax=axes[1],
            layer=layer,
            title="Top expressed genes",
            gene_colors=gene_colors,
        )

        fig.tight_layout()
        img_path = images_dir / f"cell_{cell_id}.png"
        fig.savefig(img_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        metadata_rows.append(
            {
                "cell_id": cell_id,
                "cell_type": cell_type,
                "image_path": str(img_path),
                "sampling_weight": float(obs["sampling_weight"]),
                "center_x": cx,
                "center_y": cy,
            }
        )
        pd.DataFrame(metadata_rows).to_csv(annotation_dir / "metadata.csv", index=False)

    return pd.DataFrame(metadata_rows)
