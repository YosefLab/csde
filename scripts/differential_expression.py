"""
Run CSDE differential expression from a completed annotation directory.

Example
-------
python scripts/differential_expression.py --dir /path/to/annotations/R2_macrophages

Reads all export settings from config.json in the annotation directory.
Results are written to <dir>/results.csv by default.
"""

import argparse
import json
from pathlib import Path

from csde import prepare_csde_inputs, run_csde


def parse_args():
    p = argparse.ArgumentParser(description="Run CSDE differential expression.")
    p.add_argument("--dir", required=True,
                   help="Annotation directory (output of export.py + annotate.py).")
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: <dir>/results.csv).")
    p.add_argument("--spatial-group-key", default="spatial_group",
                   help="obs column encoding the two spatial populations.")
    p.add_argument("--spatial-group-target", default="1",
                   help="Value of --spatial-group-key identifying the target region "
                        "(group 1). Positive log-fold changes mean higher expression "
                        "here.")
    p.add_argument("--spatial-group-reference", default="0",
                   help="Value of --spatial-group-key identifying the reference "
                        "region (group 0).")
    p.add_argument("--n-cells-expressed-threshold", type=int, default=10,
                   help="Min cells expressing a gene for it to be tested.")
    p.add_argument("--noise-model", default="poisson", choices=["poisson", "nb"])
    return p.parse_args()


def main():
    args = parse_args()
    annotation_dir = Path(args.dir)
    out = Path(args.out) if args.out else annotation_dir / "results.csv"

    with open(annotation_dir / "config.json") as f:
        config = json.load(f)
    layer = config.get("layer", None)

    inputs = prepare_csde_inputs(
        annotation_dir=annotation_dir,
        spatial_group_key=args.spatial_group_key,
        spatial_group_target=args.spatial_group_target,
        spatial_group_reference=args.spatial_group_reference,
        layer=layer,
        n_cells_expressed_threshold=args.n_cells_expressed_threshold,
    )
    adata_gt = inputs["adata_gt"]
    adata_other = inputs["adata_other"]
    summary = inputs["summary"]

    print("Manual annotations:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    results = run_csde(
        adata_pred=adata_other,
        adata_gt=adata_gt,
        pred_cell_pop_key="prediction",
        gt_cell_pop_key="annotation",
        cell_pop_a=0,
        cell_pop_b=1,
        layer_name=layer,
        importance_weights=1.0 / adata_gt.obs["sampling_weight"].values,
        noise_model=args.noise_model,
    )

    results.to_csv(out)
    print(f"Results written to {out}")
    print(f"{len(results)} genes tested.")
    print(results.sort_values("p_value_adj").head(20).to_string())


if __name__ == "__main__":
    main()
