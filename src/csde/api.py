from typing import Optional

import anndata
import numpy as np
import pandas as pd

from csde.model import InterceptRegression


def _map_cell_types(
    obs: pd.DataFrame,
    cell_type_col: str,
    cell_pop_a: str,
    cell_pop_b: str,
) -> np.ndarray:
    """
    Map cell types to a simplified 3-class representation.
    0: cell_pop_a (Reference)
    1: cell_pop_b (Target)
    2: Other
    """
    labels = np.full(len(obs), 2, dtype=int)

    # Check if cell types exist
    if cell_pop_a not in obs[cell_type_col].values:
        raise ValueError(
            f"Cell population '{cell_pop_a}' not found in column '{cell_type_col}'"
        )
    if cell_pop_b not in obs[cell_type_col].values:
        raise ValueError(
            f"Cell population '{cell_pop_b}' not found in column '{cell_type_col}'"
        )

    labels[obs[cell_type_col] == cell_pop_a] = 0
    labels[obs[cell_type_col] == cell_pop_b] = 1

    return labels


def run_csde(
    adata_pred: anndata.AnnData,
    adata_gt: anndata.AnnData,
    pred_cell_pop_key: str,
    cell_pop_a: str,
    cell_pop_b: str,
    gt_key: str,
    layer_name: Optional[str] = None,
    importance_weights: Optional[np.ndarray] = None,
    **model_kwargs,
) -> pd.DataFrame:
    """
    Perform differential expression analysis between two cell populations using CSDE.

    This function corrects for unreliable prediction-based cell population assignments
    using a small subset of ground-truth assignments.

    Args:
        adata_pred: AnnData object containing cells with prediction-based assignments only.
        adata_gt: AnnData object containing cells with ground-truth assignments.
        pred_cell_pop_key: Column in .obs containing the prediction-based cell population labels.
        cell_pop_a: Name of the first cell population (reference group).
        cell_pop_b: Name of the second cell population (target group).
        gt_key: Boolean column in adata_gt.obs indicating if the prediction is correct.
        layer_name: Layer in adata.layers to use for expression counts. If None, uses .X.
        importance_weights: Optional 1-D array of importance weights for the ground-truth
            observations. Will be normalized to sum to n_obs internally.
        **model_kwargs: Additional arguments passed to InterceptRegression (e.g., family, optimizer).

    Returns:
        DataFrame indexed by gene names with columns:
        - log_fold_change: Log-fold change of expression (cell_pop_b vs cell_pop_a).
        - p_value: P-value for the differential expression hypothesis.
        - p_value_adj: Multiplicity-adjusted p-value.
        - beta: The estimated coefficient.
    """

    # create simplified 3-class representation for predictions  (pop_a, pop_b, other)
    y_pred_unl = _map_cell_types(
        adata_pred.obs, pred_cell_pop_key, cell_pop_a, cell_pop_b
    )
    y_pred_gt_set = _map_cell_types(
        adata_gt.obs, pred_cell_pop_key, cell_pop_a, cell_pop_b
    )

    # logic to construct gt labels based on boolean column gt_key
    # - if predicted as a and correct (gt_key=true) -> gt is a (0)
    # - if predicted as b and correct (gt_key=true) -> gt is b (1)
    # - else -> gt is other (2)
    y_gt = np.full(len(adata_gt), 2, dtype=int)
    is_correct = adata_gt.obs[gt_key].values.astype(bool)
    is_pred_a = (adata_gt.obs[pred_cell_pop_key] == cell_pop_a).values
    is_pred_b = (adata_gt.obs[pred_cell_pop_key] == cell_pop_b).values
    y_gt[is_pred_a & is_correct] = 0
    y_gt[is_pred_b & is_correct] = 1

    def get_X(adata):
        if layer_name:
            X = adata.layers[layer_name]
        else:
            X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        return X.astype(float)

    X_gt = get_X(adata_gt)
    X_unl = get_X(adata_pred)
    inputs_gt = (X_gt, y_gt)
    inputs_hat = (X_gt, y_pred_gt_set)
    inputs_unl = (X_unl, y_pred_unl)

    # inference
    model = InterceptRegression(
        inputs_gt=inputs_gt,
        inputs_hat=inputs_hat,
        inputs_unl=inputs_unl,
        importance_weights=importance_weights,
        **model_kwargs,
    )
    model.fit(lambd_=None)
    model.get_asymptotic_distribution()

    # statistical test for DE
    res = model.test_differential_expression(
        idx_a=1, feature_names=list(adata_gt.var_names)
    )
    res = res.rename(
        columns={"beta": "log_fold_change", "pval": "p_value", "padj": "p_value_adj"}
    )
    if "feature_name" in res.columns:
        res = res.set_index("feature_name")
    return res[["log_fold_change", "p_value", "p_value_adj"]]
