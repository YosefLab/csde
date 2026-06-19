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

Refer to the [preprint](https://www.biorxiv.org/content/10.64898/2026.01.15.699786v1) for details on the method. Reproducibility code is available [here](https://github.com/PierreBoyeau/csde_experiments).

### Input requirements

The workflow takes a [SpatialData](https://spatialdata.scverse.org/) zarr as input.

Its `"table"` AnnData must contain:

- **raw expression counts** in `.X` or a named layer
- **the following `obs` columns:**

| obs column | content |
| --- | --- |
| `cell_type` (configurable) | cell-type label for each cell |
| `spatial_group` (configurable) | binary spatial region label (e.g. `0` = outside tumour, `1` = inside tumour) |
| `center_x`, `center_y` | cell centroid in microns |

The zarr must also expose the following SpatialData elements, used to render the per-cell annotation panels (Step 1):

| element | requirement |
| --- | --- |
| `images` | at least one image with a named **fluorescence channel** (e.g. `"DAPI"`, `"Cellbound2"`) |
| `shapes` | at least one element holding the **cell-boundary polygons** |
| `points` | at least one element holding **transcript locations**, with a `gene` column |

The cell-boundary `shapes` must carry a transformation to the `global` coordinate system: it converts the micron `center_x`/`center_y` centroids into the image's pixel space. This conversion assumes a pure scale-and-translation transform (as produced for MERSCOPE); transforms with rotation or shear are not handled.

## Installation

```bash
pip install csde
pip install "csde[cuda12]"          # GPU (CUDA 12)
pip install "csde[annotate]"        # annotation UI (Step 2, requires streamlit)
pip install "csde[cuda12,annotate]" # both
```

## Workflow overview

CSDE runs as three scripts executed in sequence, each consuming the previous one's output: `export.py` samples a small set of cells and renders an annotation panel for each, `annotate.py` lets you manually mark those cells as correct or incorrect, and `differential_expression.py` feeds those validated labels into the CSDE model to produce corrected DE estimates. All three share a single annotation directory.

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
--n-cells 600 \
--layer counts
```

`--target-proportion` controls the fraction of cells of interest in the subsample. Cells of interest are upweighted accordingly (importance sampling); the unnormalized weight for each sampled cell is stored in `metadata.csv` for downstream use.

`--layer` selects which expression matrix to read: the named `.layers` entry holding the raw counts (e.g. `counts`), or `.X` when omitted. The value is saved to `config.json` and reused throughout the workflow — the same layer feeds the top-gene panels here in Step 1 and the CSDE model in Step 3, so set it once at export time. **It must point at raw counts**, since the noise model (Poisson / negative binomial) assumes integer counts; pointing it at normalised or log-transformed values will produce invalid results.

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

---

## Step 2 — Manual validation (`scripts/annotate.py`)

For each exported image, an annotator decides whether the cell is **correct** — meaning it is both properly **segmented** and properly **labelled**. A cell should be rejected (marked incorrect) when either check fails:

- **Segmentation** — the cell boundary (left panel) is not consistent with the nuclei / membrane staining, e.g. it merges two cells or clips part of one.
- **Cell-type label** — the top expressed genes (right panel) include genes unlikely to be expressed by the cell type of interest, suggesting the automated label is wrong.

The result is a boolean column `is_correct` added to `metadata.csv`, which becomes `adata_gt` in Step 3.

```bash
streamlit run scripts/annotate.py -- --dir /path/to/annotation_dir
```

The `--` is required: it tells Streamlit to pass everything after it to the script rather than interpreting it as Streamlit's own options.

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
