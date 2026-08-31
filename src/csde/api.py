from typing import Optional

import anndata
import numpy as np
import pandas as pd

from csde.model_poisson import PoissonIntercept
from csde.model_nb import NBIntercept


def _map_cell_types(
    obs: pd.DataFrame,
    cell_type_col: str,
    cell_pop_a: str,
    cell_pop_b: str,
    context: str = "dataset",
    hint: str = "",
) -> np.ndarray:
    """
    Map cell types to a simplified 3-class representation.
    0: cell_pop_a (Reference)
    1: cell_pop_b (Target)
    2: Other

    ``context`` and ``hint`` only shape the error raised when a population is
    empty; the mapping itself is identical for the automated and manual labels.
    """
    labels = np.full(len(obs), 2, dtype=int)

    for cell_pop in (cell_pop_a, cell_pop_b):
        if cell_pop not in obs[cell_type_col].values:
            raise ValueError(
                f"No cells assigned to population '{cell_pop}' in column "
                f"'{cell_type_col}' of the {context}.{hint}"
            )

    labels[obs[cell_type_col] == cell_pop_a] = 0
    labels[obs[cell_type_col] == cell_pop_b] = 1

    return labels


def run_csde(
    adata_pred: anndata.AnnData,
    adata_gt: anndata.AnnData,
    pred_cell_pop_key: str,
    gt_cell_pop_key: str,
    cell_pop_a: str,
    cell_pop_b: str,
    layer_name: Optional[str] = None,
    importance_weights: Optional[np.ndarray] = None,
    noise_model: str = "poisson",
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
            Read from both ``adata_pred`` and ``adata_gt``.
        gt_cell_pop_key: Column in adata_gt.obs containing the manually curated cell
            population labels. Built upstream by
            :func:`~csde.prepare_csde_inputs`, which resolves the annotation scheme
            (accept/reject or accept/correct/reject) into labels; this function only
            consumes them.
        cell_pop_a: Name of the first cell population (reference group).
        cell_pop_b: Name of the second cell population (target group).
        layer_name: Layer in adata.layers to use for expression counts. If None, uses .X.
        importance_weights: Optional 1-D array of importance weights for the ground-truth
            observations. Will be normalized to sum to n_obs internally.
        noise_model: Noise model to use. Either "poisson" or "nb".
        **model_kwargs: Additional arguments passed to PoissonIntercept (e.g., optimizer).

    Returns:
        DataFrame indexed by gene names with columns:
        - log_fold_change: Estimated log-fold change of expression
          (positive = upregulated in cell_pop_b relative to cell_pop_a).
        - p_value: Raw two-sided p-value for the differential expression hypothesis.
        - p_value_adj: Benjamini-Hochberg multiplicity-adjusted p-value.
    """

    # create simplified 3-class representation  (pop_a, pop_b, other).
    # The automated and manual labels are mapped identically; the annotation
    # scheme that produced the manual labels is resolved upstream.
    y_pred_unl = _map_cell_types(
        adata_pred.obs,
        pred_cell_pop_key,
        cell_pop_a,
        cell_pop_b,
        context="unlabeled set",
    )
    y_pred_gt_set = _map_cell_types(
        adata_gt.obs,
        pred_cell_pop_key,
        cell_pop_a,
        cell_pop_b,
        context="automated labels of the manually annotated set",
    )
    y_gt = _map_cell_types(
        adata_gt.obs,
        gt_cell_pop_key,
        cell_pop_a,
        cell_pop_b,
        context="manually annotated set",
        hint=(
            " No annotated cell was curated into this group, so its expression "
            "cannot be estimated. Annotate more cells, or increase the "
            "importance-sampling weight of the cell type of interest."
        ),
    )

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
    if noise_model == "poisson":
        model = PoissonIntercept(
        inputs_gt=inputs_gt,
        inputs_hat=inputs_hat,
        inputs_unl=inputs_unl,
        importance_weights=importance_weights,
        **model_kwargs,
    )
    elif noise_model == "nb":
        model = NBIntercept(
            inputs_gt=inputs_gt,
            inputs_hat=inputs_hat,
            inputs_unl=inputs_unl,
            importance_weights=importance_weights,
            **model_kwargs,
        )
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")
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
