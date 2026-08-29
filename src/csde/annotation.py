from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .spatial_utils import plot_region, plot_top_genes, subsample_cells

# --- annotations.json schema -------------------------------------------------
# {cell_id: {"action": "accept" | "correct" | "reject", "label": str | None}}
#
# ``label`` is set only for "correct"; the manual cell type of an accepted cell
# is resolved from the automated label at read time, so a cell type is never
# recorded in two places. scripts/annotate.py writes this file and mirrors these
# constants; the validation below is what keeps the two in step.
ACCEPT = "accept"
CORRECT = "correct"
REJECT = "reject"
ANNOTATION_ACTIONS = (ACCEPT, CORRECT, REJECT)


def _resolve_group_value(value, observed: set):
    """
    Match a requested spatial-group value against the values in the obs column.

    Exact equality first, then string equality, so that a value coming from the
    command line (always a string) matches an integer-encoded column. Returns the
    value as it appears in the column, or the input unchanged when there is no
    match — the caller reports that.
    """
    if value in observed:
        return value
    by_str = {str(seen): seen for seen in observed}
    return by_str.get(str(value), value)


def read_annotations(annotation_dir: str | Path) -> dict:
    """
    Load ``annotations.json`` and validate it against the schema above.

    Returns
    -------
    dict
        ``{cell_id: {"action": ..., "label": ...}}``, with ``cell_id`` as ``str``.
    """
    annotation_dir = Path(annotation_dir)
    ann_path = annotation_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"No annotations found at {ann_path}. Run scripts/annotate.py first."
        )
    with open(ann_path) as f:
        records = json.load(f)

    for cell_id, record in records.items():
        if not isinstance(record, dict) or "action" not in record:
            raise ValueError(
                f"Malformed annotation for cell '{cell_id}': {record!r}. Expected "
                '{"action": "accept" | "correct" | "reject", "label": str | None}.'
            )
        action, label = record["action"], record.get("label")
        if action not in ANNOTATION_ACTIONS:
            raise ValueError(
                f"Unknown action '{action}' for cell '{cell_id}'. "
                f"Expected one of {ANNOTATION_ACTIONS}."
            )
        if action == CORRECT and label is None:
            raise ValueError(
                f"Cell '{cell_id}' is annotated as '{CORRECT}' but carries no label."
            )
        if action != CORRECT and label is not None:
            raise ValueError(
                f"Cell '{cell_id}' is annotated as '{action}' but carries label "
                f"'{label}'. Only '{CORRECT}' annotations may set a label."
            )
    return {str(cell_id): record for cell_id, record in records.items()}


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
    (accept / correct / reject per cell) from annotation_dir.

    This function owns the translation from annotation scheme to model labels:
    :func:`~csde.run_csde` consumes ``.obs["annotation"]`` as-is.

    Label encoding in .obs["prediction"] / .obs["annotation"]:
        1 — cell_type_of_interest with spatial_group == 1  (target)
        0 — cell_type_of_interest with spatial_group == 0  (reference)
        2 — all other cells

    ``prediction`` uses the automated cell type; ``annotation`` uses the manual
    cell type, which is the automated one for an *accepted* cell, the annotator's
    choice for a *corrected* one, and undefined (hence class 2) for a *rejected*
    one. A corrected cell keeps its automated spatial group: only the cell-type
    component of the label is curated. Correcting a cell *into*
    cell_type_of_interest therefore moves it from class 2 into class 0 or 1,
    which is the case an accept/reject scheme cannot express.

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
        ``annotation`` (int 0/1/2), ``manual_action`` (str),
        ``manual_cell_type`` (str or None), ``sampling_weight`` (float).
        Genes are filtered.
    adata_other : AnnData
        All unannotated cells. obs column added: ``prediction`` (int 0/1/2).
        Same gene set as adata_gt.
    summary : dict
        Per-action counts plus ``n_promoted`` / ``n_demoted``, the number of
        cells the manual curation moved into / out of the compared groups.
    """
    annotation_dir = Path(annotation_dir)

    with open(annotation_dir / "config.json") as f:
        config = json.load(f)
    cell_type_key = config["cell_type_key"]
    cell_type_of_interest = config["cell_type_of_interest"]

    annotations = read_annotations(annotation_dir)

    metadata = pd.read_csv(annotation_dir / "metadata.csv")
    metadata["cell_id"] = metadata["cell_id"].astype(str)
    sampling_weights = metadata.set_index("cell_id")["sampling_weight"]

    if sdata is None:
        import spatialdata as sd
        sdata_path = config.get("sdata")
        if not sdata_path:
            raise ValueError(
                "No sdata path found in config.json. "
                "Pass sdata directly: prepare_csde_inputs(..., sdata=your_sdata_object)."
            )
        if not Path(sdata_path).exists():
            raise FileNotFoundError(
                f"SpatialData zarr not found at '{sdata_path}' (path stored in config.json). "
                "Either restore the zarr to that path, or pass sdata directly: "
                "prepare_csde_inputs(..., sdata=your_sdata_object)."
            )
        sdata = sd.read_zarr(sdata_path)
    adata = sdata["table"].copy()
    adata.obs_names = adata.obs_names.astype(str)
    adata = adata[adata.obs[cell_type_key].notna()].copy()

    if cell_type_of_interest not in set(adata.obs[cell_type_key].unique()):
        raise ValueError(
            f"cell_type_of_interest={cell_type_of_interest!r} was not found in obs "
            f"column '{cell_type_key}', which contains "
            f"{sorted(map(str, adata.obs[cell_type_key].unique()))}."
        )

    # Fail before any label math: a region value that does not occur would send
    # every cell to class 2, and the resulting emptiness would only surface much
    # later as a confusing "population not found" error. Not auto-detected from a
    # two-level column on purpose — which value becomes the target sets the sign
    # of every log-fold change, so it has to be chosen explicitly.
    observed_groups = set(adata.obs[spatial_group_key].unique())
    spatial_group_target = _resolve_group_value(spatial_group_target, observed_groups)
    spatial_group_reference = _resolve_group_value(
        spatial_group_reference, observed_groups
    )
    missing = [
        value
        for value in (spatial_group_target, spatial_group_reference)
        if value not in observed_groups
    ]
    if missing:
        raise ValueError(
            f"spatial_group_target={spatial_group_target!r} and "
            f"spatial_group_reference={spatial_group_reference!r}: "
            f"{missing!r} not found in obs column '{spatial_group_key}', which "
            f"contains {sorted(map(str, observed_groups))}. Pass "
            "--spatial-group-target / --spatial-group-reference to choose the two "
            "regions to compare."
        )

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

    # --- Manual cell type: accepted keeps the automated label, corrected takes
    #     the annotator's, rejected has none ---
    actions = np.array(
        [annotations[cid]["action"] for cid in adata_gt.obs_names], dtype=object
    )
    manual_cell_type = []
    for cell_id, predicted_type in zip(
        adata_gt.obs_names, adata_gt.obs[cell_type_key]
    ):
        record = annotations[cell_id]
        if record["action"] == REJECT:
            manual_cell_type.append(None)
        elif record["action"] == CORRECT:
            manual_cell_type.append(record["label"])
        else:
            manual_cell_type.append(predicted_type)

    # The vocabulary is closed: a corrected label absent from the data would
    # otherwise fall through to class 2 and be indistinguishable from a rejection.
    vocabulary = set(adata.obs[cell_type_key].unique())
    unknown = sorted(
        {
            label
            for label, action in zip(manual_cell_type, actions)
            if action == CORRECT and label not in vocabulary
        }
    )
    if unknown:
        raise ValueError(
            f"Corrected labels {unknown} are not present in "
            f"'{cell_type_key}' of the SpatialData table. Known cell types: "
            f"{sorted(vocabulary)}."
        )

    adata_gt.obs["manual_action"] = actions
    adata_gt.obs["manual_cell_type"] = manual_cell_type

    # --- Annotation (GT) labels ---
    # The spatial group is always the automated one: manual curation revises the
    # cell type, never the region.
    is_coi_gt = np.array(
        [label == cell_type_of_interest for label in manual_cell_type]
    )
    spatial_group_gt = adata_gt.obs[spatial_group_key].values

    annotation = np.full(len(adata_gt), 2, dtype=int)
    annotation[is_coi_gt & (spatial_group_gt == spatial_group_target)] = 1
    annotation[is_coi_gt & (spatial_group_gt == spatial_group_reference)] = 0
    adata_gt.obs["annotation"] = annotation

    adata_gt.obs["sampling_weight"] = adata_gt.obs_names.map(sampling_weights).values

    prediction_gt = adata_gt.obs["prediction"].values
    summary = {
        "n_annotated": int(len(adata_gt)),
        **{
            f"n_{action}": int((actions == action).sum())
            for action in ANNOTATION_ACTIONS
        },
        # Cells the curation moved into / out of the two compared groups.
        "n_promoted": int(((prediction_gt == 2) & (annotation != 2)).sum()),
        "n_demoted": int(((prediction_gt != 2) & (annotation == 2)).sum()),
    }

    # Both label sets must populate both compared groups. Checked here, rather
    # than left to fail at fit time, so the message can name the region and the
    # counts: this is usually an annotation-budget problem, not a wiring one.
    for labels, source in (
        (prediction_gt, "the automated pipeline"),
        (annotation, "manual annotation"),
    ):
        for group, region in (
            (0, spatial_group_reference),
            (1, spatial_group_target),
        ):
            if (labels == group).sum() > 0:
                continue
            n_in_groups = int((labels != 2).sum())
            raise ValueError(
                f"No annotated cell falls in group {group} "
                f"({cell_type_of_interest!r} in region {region!r}) according to "
                f"{source}, so its expression cannot be estimated. Of "
                f"{len(adata_gt)} annotated cells, {n_in_groups} are "
                f"{cell_type_of_interest!r} across both regions. Annotate more "
                "cells, or raise --target-proportion at export time to sample "
                f"more {cell_type_of_interest!r}."
            )

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

    return {"adata_gt": adata_gt, "adata_other": adata_other, "summary": summary}


def load_annotations(annotation_dir: str | Path) -> pd.DataFrame:
    """
    Merge ``metadata.csv`` and ``annotations.json`` into a single DataFrame.

    Returns only annotated cells, with added ``action`` (accept / correct /
    reject) and ``manual_cell_type`` columns. The latter is the automated
    ``cell_type`` for accepted cells, the annotator's choice for corrected ones,
    and None for rejected ones.

    This is a convenience view for inspecting annotations; the model labels are
    built by :func:`prepare_csde_inputs`.
    """
    annotation_dir = Path(annotation_dir)
    metadata = pd.read_csv(annotation_dir / "metadata.csv")
    metadata["cell_id"] = metadata["cell_id"].astype(str)

    annotations = read_annotations(annotation_dir)

    metadata["action"] = metadata["cell_id"].map(
        {cell_id: record["action"] for cell_id, record in annotations.items()}
    )
    metadata = metadata[metadata["action"].notna()].copy()
    metadata["manual_cell_type"] = [
        None
        if action == REJECT
        else (annotations[cell_id]["label"] if action == CORRECT else predicted)
        for cell_id, action, predicted in zip(
            metadata["cell_id"], metadata["action"], metadata["cell_type"]
        )
    ]
    return metadata


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
