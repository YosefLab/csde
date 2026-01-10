# CSDE: Corrected Spatial Differential Expression

`csde` (Corrected Spatial Differential Expression) is a Python package designed to **identify differentially expressed (DE) genes between spatially-resolved cell populations** (e.g., T-cells inside vs. outside a tumor).

Standard analysis relies on cell population assignments (e.g., "infiltrating" vs. "non-infiltrating") obtained automatically from clustering/ML that are often prone to errors. `csde` corrects for these inaccuracies by leveraging a small subset of validated "ground-truth" data, providing rigorous statistical guarantees for spatially-resolved DE analyses.

Refer to the preprint and the [project repository](https://github.com/YosefLab/csde) for more details.

## Installation

```bash
pip install csde
```

By default, this installs JAX with CPU support. To enable GPU support (CUDA), install with the appropriate extra (e.g., for CUDA 12):
```bash
pip install "csde[cuda12]"
```

## Data Requirements

`csde` requires two `AnnData` objects containing gene expression counts. Typically, these are obtained by splitting your full dataset into two groups:

### 1. `adata_pred`: The dataset to analyze
This object contains the bulk of your cells (e.g., the majority of the tissue) where only standard (predicted) cell population assignments are available.

**Requirements:**
*   A column in `.obs` (e.g., `"cell_population"`) containing cell population labels (e.g., "T cell (infiltrating)" vs. "T cell (non-infiltrating)"). These labels can be derived from heuristics (e.g., distance to tumor) and/or computational classifiers.

### 2. `adata_gt`: The correction set
This object contains a small subset of randomly sampled cells whose cell population assignments have been **validated** to serve as a ground truth. This set allows `csde` to estimate the error rate of the standard predictions.

**Requirements:**
*   **Prediction column:** The same column name as in `adata_pred` (e.g., `"cell_population"`), containing the automated labels.
*   **Validation column:** A **boolean** column in `.obs` (e.g., `"is_correct"`) indicating if the automated label matches the validation ground truth (see [How to construct `adata_gt`?](#how-to-construct-adata_gt)).

## Usage

```python
from csde import run_csde

results = run_csde(
    # `AnnData` datasets to analyze
    adata_pred=adata_pred,
    adata_gt=adata_gt,
    
    # Column containing the predicted labels (in BOTH datasets)
    pred_cell_pop_key="cell_population",
    
    # The two populations to compare
    cell_pop_a="T-cell (infiltrating)",       # Reference group
    cell_pop_b="T-cell (non-infiltrating)",   # Target group
    
    # Boolean column in adata_gt verifying the prediction
    gt_key="is_correct",
    
    # Optional: Use a specific layer for counts (default uses .X)
    layer_name="counts"
)

# Returns a DataFrame with log_fold_change, p_value, and adjusted p_value
print(results.head())
```

### Output Columns
The returned DataFrame is indexed by gene name and contains:
*   `log_fold_change`: The estimated log-fold change of expression (Target vs. Reference). Positive values indicate upregulation in `cell_pop_b`.
*   `p_value`: The raw p-value from the hypothesis test (two-sided).
*   `p_value_adj`: The p-value adjusted for multiple testing (Benjamini-Hochberg FDR).

## How to construct `adata_gt`?

Constructing `adata_gt` requires validating the cell population labels for a small subset of cells (e.g., random sample). This involves:
1.  **Sampling**: Select a small random subset of cells from your dataset.
2.  **Data Access**: Extract the relevant data for these cells: their gene expression profile, their spatial coordinates, and importantly, a **high-resolution image crop** of the cell (with segmentation boundaries if available) to assess morphology.
3.  **Validation**: Visually inspect these data points to determine the true cell identity.
4.  **Annotation**: Create the `is_correct` boolean column based on your assessment.

These steps can be performed manually or using dedicated tools.
Our [experimental repository](https://github.com/YosefLab/csde/blob/main/csde_experiments)
provides an example of how these steps were performed for MERFISH data.

To streamline this process, for MERFISH or other spatial transcriptomics data, we recommend using **[SpatialData](https://spatialdata.scverse.org/)** to access the data and perform the manual validation.

