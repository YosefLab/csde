"""
Export per-cell annotation panels from a SpatialData zarr.

Example
-------
python scripts/export.py \
    --sdata  /ewsc/pboyeau/data/processed/region_R2_annotated.zarr \
    --out    /ewsc/pboyeau/data/annotations/R2_macrophages \
    --cell-type-key cell_type \
    --cell-type-of-interest macrophages \
    --target-proportion 0.4 \
    --gene-colors scripts/gene_colors_macrophages.json \
    --image-channel Cellbound2 \
    --n-cells 600
"""

import argparse
import json
from pathlib import Path

import spatialdata as sd

from csde import export_cell_panels


def parse_args():
    p = argparse.ArgumentParser(description="Export per-cell annotation panels.")
    p.add_argument("--sdata", required=True, help="Path to annotated SpatialData zarr.")
    p.add_argument("--out", required=True, help="Output annotation directory.")
    p.add_argument("--cell-type-key", default="cell_type")
    p.add_argument("--cell-type-of-interest", required=True)
    p.add_argument("--target-proportion", type=float, required=True,
                   help="Desired fraction of cells of interest in the subsample.")
    p.add_argument("--gene-colors", default=None,
                   help="JSON file mapping gene name → colour.")
    p.add_argument("--image-channel", default="DAPI")
    p.add_argument("--n-cells", type=int, default=600)
    p.add_argument("--annotation-mode",
                   choices=["accept_correct_reject", "accept_reject"],
                   default="accept_correct_reject",
                   help="Actions offered by annotate.py. accept_correct_reject "
                        "additionally lets the annotator relabel a cell.")
    p.add_argument("--delta", type=float, default=50.0,
                   help="Half-width of the spatial crop around each cell (microns).")
    p.add_argument("--n-top-genes", type=int, default=15)
    p.add_argument("--layer", default=None,
                   help="AnnData layer for expression counts (default: X).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()

    gene_colors = None
    if args.gene_colors:
        with open(args.gene_colors) as f:
            gene_colors = json.load(f)

    sdata = sd.read_zarr(args.sdata)

    annotation_dir = Path(args.out)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    cell_types = sdata["table"].obs[args.cell_type_key].dropna().unique()
    config = vars(args) | {"cell_type_vocabulary": sorted(map(str, cell_types))}
    with open(annotation_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    metadata = export_cell_panels(
        sdata=sdata,
        annotation_dir=annotation_dir,
        cell_type_key=args.cell_type_key,
        cell_type_of_interest=args.cell_type_of_interest,
        target_proportion=args.target_proportion,
        n_cells=args.n_cells,
        image_channel=args.image_channel,
        delta=args.delta,
        n_top_genes=args.n_top_genes,
        layer=args.layer,
        gene_colors=gene_colors,
        seed=args.seed,
        dpi=args.dpi,
    )
    print(f"Done. {len(metadata)} cells exported to {args.out}")
    print(metadata["cell_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
