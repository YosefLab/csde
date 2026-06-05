# CSDE: Corrected Spatial Differential Expression

[![Tests](https://github.com/YosefLab/csde/actions/workflows/test.yml/badge.svg)](https://github.com/YosefLab/csde/actions/workflows/test.yml)

Automated pipelines for spatial transcriptomics produce cell quantifications (cell-by-gene expression matrices and label assignments) that contain systematic errors, e.g., due to mis-segmentation of cell boundaries.
These errors can propagate into downstream analyses of differential expression, leading to false discoveries or missed signals

CSDE corrects for these errors by combining the large automated dataset with a small set of manually validated cells, using prediction-powered inference to recover unbiased estimates with valid confidence intervals.

The current codebase focuses on the comparison of a given cell type across two spatial regions.
It allows users to
1. export per-cell annotation panels for a small subset of cells (e.g. 600)
2. manually validate the segmentation and type assignment for these cells
3. run the CSDE model to get corrected DE estimates for all genes

Refer to the [preprint](https://www.biorxiv.org/content/10.64898/2026.01.15.699786v1) for details on the method.

### Input requirements

The workflow takes a [SpatialData](https://spatialdata.scverse.org/) zarr as input. Its `"table"` AnnData must contain:

- **raw expression counts** in `.X` or a named layer
- **the following `obs` columns:**

| obs column | content |
| --- | --- |
| `cell_type` (configurable) | cell-type label for each cell |
| `spatial_group` (configurable) | binary spatial region label (e.g. `0` = outside tumour, `1` = inside tumour) |
| `center_x`, `center_y` | cell centroid in microns |

The zarr must also expose at least one **fluorescence image channel** (e.g. `"DAPI"`, `"Cellbound2"`) used to render the per-cell annotation panels.

## Installation

```bash
pip install csde
pip install "csde[cuda12]"          # GPU (CUDA 12)
pip install "csde[annotate]"        # annotation UI (Step 2, requires streamlit)
pip install "csde[cuda12,annotate]" # both
```

## Workflow overview

```
SpatialData zarr
      │
      ▼
1. Export annotation panels   ←─ scripts/export.py
   (importance-sampled cells,
    one image per cell)
      │
      ▼
2. Manual validation          ←─ scripts/annotate.py
   (annotator marks each cell
    as correctly / incorrectly labelled)
      │
      ▼
3. Run CSDE                   ←─ scripts/differential_expression.py
   (corrected DE estimates)
```

---

## Step 1 — Export annotation panels (`scripts/export.py`)

Before running the statistical model, a small subset of cells must be manually validated. `csde` provides tooling to generate the per-cell images needed for that step.

```bash
python scripts/export.py \
--sdata  /path/to/region.zarr \
--out    /path/to/annotation_dir \
--cell-type-key cell_type \
--cell-type-of-interest macrophages \
--target-proportion 0.4 \
--gene-colors scripts/gene_colors_file.json \
--image-channel Cellbound2 \
--n-cells 600
```

`--target-proportion` controls the fraction of cells of interest in the subsample. Cells of interest are upweighted accordingly (importance sampling); the unnormalized weight for each sampled cell is stored in `metadata.csv` for downstream use.

The script writes:

```
/path/to/annotation_dir/
├── images/
│   ├── cell_<id>.png   # one panel per cell
│   └── ...
├── config.json         # all export arguments (read by annotate.py)
├── metadata.csv        # cell_id, cell_type, image_path, sampling_weight, center_x, center_y
└── annotations.json    # {cell_id: true/false} — written by annotate.py
```

Each panel contains:
- **Left** — fluorescence image crop + cell boundaries + transcript dots for genes listed in `gene_colors`
- **Right** — top expressed genes (bar chart); genes in `gene_colors` use their assigned colour, others are grey

### Gene color file

A simple JSON mapping gene names to colours:

```json
{
    "CD68":   "#e41a1c",
    "MRC1":   "#377eb8",
    "C1QA":   "#4daf4a",
    "FCGR3A": "#ff7f00"
}
```

<details>
<summary>Python API</summary>

```python
import json
import spatialdata as sd
from csde import export_cell_panels, subsample_cells, plot_top_genes

sdata = sd.read_zarr("/path/to/region.zarr")
gene_colors = json.load(open("gene_colors.json"))

metadata = export_cell_panels(
    sdata=sdata,
    annotation_dir="/path/to/annotation_dir",
    cell_type_key="cell_type",
    cell_type_of_interest="macrophages",
    target_proportion=0.4,
    gene_colors=gene_colors,
    image_channel="Cellbound2",
    n_cells=600,
)
```
</details>

---

## Step 2 — Manual validation (`scripts/annotate.py`)

For each exported image, an annotator decides whether the automated cell-type label is correct. The result is a boolean column `is_correct` added to `metadata.csv`, which becomes `adata_gt` in Step 3.

```bash
streamlit run scripts/annotate.py -- --dir /path/to/annotation_dir
```

VS Code Remote forwards the Streamlit port automatically. Open the URL printed in the terminal, then use:

- **`1`** — label as correct
- **`2`** — label as incorrect

Progress is saved after every keypress to `annotations.json`. Re-running the command resumes from where you left off. You can also start annotating while `export.py` is still running — the UI picks up newly exported cells automatically.

---

## Step 3 — Differential expression (`scripts/differential_expression.py`)

```bash
python scripts/differential_expression.py --dir /path/to/annotation_dir
```

Reads all export settings from `config.json` and writes gene-level results to `<dir>/results.csv`.

| option | default | description |
|---|---|---|
| `--dir` | *(required)* | annotation directory (output of steps 1 & 2) |
| `--out` | `<dir>/results.csv` | output CSV path |
| `--spatial-group-key` | `spatial_group` | obs column encoding the two spatial populations |
| `--n-cells-expressed-threshold` | `10` | min annotated cells expressing a gene for it to be tested |
| `--noise-model` | `poisson` | `poisson` or `nb` (negative binomial) |

### Output columns

| column | description |
|---|---|
| `log_fold_change` | estimated LFC (positive = upregulated in target population) |
| `p_value` | raw two-sided p-value |
| `p_value_adj` | Benjamini-Hochberg adjusted p-value |

<details>
<summary>Python API</summary>

The full CSDE statistical model is callable directly from Python, without going through the CLI scripts.

`prepare_csde_inputs` reads `config.json`, `metadata.csv`, and `annotations.json` from the annotation directory produced by Steps 1 & 2. It returns two AnnData objects restricted to the same gene set:

- `adata_gt` — the manually validated cells, with an `is_correct` boolean column in `.obs` and a `sampling_weight` column reflecting the importance-sampling weight assigned during export
- `adata_other` — all remaining cells (not manually validated); their `obs` must contain a `prediction` column (integer) encoding the spatial population each cell was assigned to by the automated pipeline: `0` = reference region, `1` = target region, `2` = neither

```python
from csde import prepare_csde_inputs, run_csde

inputs = prepare_csde_inputs(
    annotation_dir="/path/to/annotation_dir",  # same dir as Steps 1 & 2
    spatial_group_key="spatial_group",
    n_cells_expressed_threshold=10,
)
adata_gt    = inputs["adata_gt"]    # manually validated cells
adata_other = inputs["adata_other"] # all other cells

results = run_csde(
    adata_pred=adata_other,
    adata_gt=adata_gt,
    pred_cell_pop_key="prediction",  # obs column: 0=reference, 1=target, 2=other
    cell_pop_a=0,                    # reference population
    cell_pop_b=1,                    # target population (LFC = log(target/reference))
    gt_key="is_correct",             # boolean correctness label from Step 2
    layer_name="counts",
    importance_weights=adata_gt.obs["sampling_weight"].values,  # from metadata.csv
)
# DataFrame indexed by gene: log_fold_change, p_value, p_value_adj
print(results.head())
```

</details>
