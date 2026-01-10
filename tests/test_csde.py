import unittest
import numpy as np
import pandas as pd
import anndata
from csde import run_csde


class TestCSDE(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        n_genes = 10

        # adata_pred
        n_pred = 100
        X_pred = np.random.poisson(lam=2.0, size=(n_pred, n_genes)).astype(float)
        obs_pred = pd.DataFrame(
            {"cell_type": np.random.choice(["TypeA", "TypeB", "TypeC"], size=n_pred)}
        )
        self.adata_pred = anndata.AnnData(X=X_pred, obs=obs_pred)
        self.adata_pred.var_names = [f"Gene_{i}" for i in range(n_genes)]

        # adata_gt
        n_gt = 50
        X_gt = np.random.poisson(lam=2.0, size=(n_gt, n_genes)).astype(float)
        obs_gt = pd.DataFrame(
            {
                "cell_type": np.random.choice(["TypeA", "TypeB", "TypeC"], size=n_gt),
                "is_correct": np.random.choice([True, False], size=n_gt),
            }
        )
        self.adata_gt = anndata.AnnData(X=X_gt, obs=obs_gt)
        self.adata_gt.var_names = [f"Gene_{i}" for i in range(n_genes)]

        # Make sure cell types exist
        # Force at least one of each to avoid mapping errors in small random samples
        self.adata_pred.obs.iloc[0, 0] = "TypeA"
        self.adata_pred.obs.iloc[1, 0] = "TypeB"
        self.adata_gt.obs.iloc[0, 0] = "TypeA"
        self.adata_gt.obs.iloc[1, 0] = "TypeB"

    def test_run_csde(self):
        res = run_csde(
            adata_pred=self.adata_pred,
            adata_gt=self.adata_gt,
            pred_cell_pop_key="cell_type",
            cell_pop_a="TypeA",
            cell_pop_b="TypeB",
            gt_key="is_correct",
            optimizer="gd",
            optimizer_kwargs={"n_iter": 10},  # Fast run
        )

        # Check output
        self.assertIsInstance(res, pd.DataFrame)
        self.assertEqual(len(res), 10)  # 10 genes
        self.assertListEqual(
            list(res.columns), ["log_fold_change", "p_value", "p_value_adj"]
        )
        self.assertTrue(not res.isnull().values.any())


if __name__ == "__main__":
    unittest.main()
