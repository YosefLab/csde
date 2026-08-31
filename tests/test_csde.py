import json
import tempfile
import unittest
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from csde import prepare_csde_inputs, run_csde


class TestCSDE(unittest.TestCase):
    def setUp(self):
        n_genes = 10

        n_pred = 100
        X_pred = np.random.poisson(lam=2.0, size=(n_pred, n_genes)).astype(float)
        obs_pred = pd.DataFrame(
            {"cell_type": np.random.choice(["TypeA", "TypeB", "TypeC"], size=n_pred)}
        )
        self.adata_pred = anndata.AnnData(X=X_pred, obs=obs_pred)
        self.adata_pred.var_names = [f"Gene_{i}" for i in range(n_genes)]

        n_gt = 50
        X_gt = np.random.poisson(lam=2.0, size=(n_gt, n_genes)).astype(float)
        is_correct = np.random.choice([True, False], size=n_gt)
        obs_gt = pd.DataFrame(
            {
                "cell_type": np.random.choice(["TypeA", "TypeB", "TypeC"], size=n_gt),
                "is_correct": is_correct,
            }
        )
        self.adata_gt = anndata.AnnData(X=X_gt, obs=obs_gt)
        self.adata_gt.var_names = [f"Gene_{i}" for i in range(n_genes)]

        self.adata_pred.obs.iloc[0, 0] = "TypeA"
        self.adata_pred.obs.iloc[1, 0] = "TypeB"
        self.adata_gt.obs.iloc[0, 0] = "TypeA"
        self.adata_gt.obs.iloc[1, 0] = "TypeB"
        self.adata_gt.obs.iloc[0, 1] = True
        self.adata_gt.obs.iloc[1, 1] = True

        # Manual labels: run_csde consumes a label column, not a boolean.
        self.adata_gt.obs["manual_cell_type"] = np.where(
            self.adata_gt.obs["is_correct"].values,
            self.adata_gt.obs["cell_type"].values,
            "Rejected",
        )

    def test_run_csde(self):
        res = run_csde(
            adata_pred=self.adata_pred,
            adata_gt=self.adata_gt,
            pred_cell_pop_key="cell_type",
            gt_cell_pop_key="manual_cell_type",
            cell_pop_a="TypeA",
            cell_pop_b="TypeB",
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

    def test_run_csde_with_importance_weights(self):
        n_gt = len(self.adata_gt)
        rng = np.random.default_rng(0)
        importance_weights = rng.uniform(0.5, 2.0, size=n_gt)

        for noise_model in ("poisson", "nb"):
            res = run_csde(
                adata_pred=self.adata_pred,
                adata_gt=self.adata_gt,
                pred_cell_pop_key="cell_type",
                gt_cell_pop_key="manual_cell_type",
                cell_pop_a="TypeA",
                cell_pop_b="TypeB",
                optimizer="gd",
                optimizer_kwargs={"n_iter": 10},
                importance_weights=importance_weights,
                noise_model=noise_model,
            )

            self.assertIsInstance(res, pd.DataFrame)
            self.assertEqual(len(res), 10)
            self.assertListEqual(
                list(res.columns), ["log_fold_change", "p_value", "p_value_adj"]
            )
            self.assertTrue(not res.isnull().values.any())

    def test_missing_gt_population_raises(self):
        # No annotated cell curated into TypeB -> the group cannot be estimated.
        adata_gt = self.adata_gt.copy()
        adata_gt.obs["manual_cell_type"] = np.where(
            adata_gt.obs["manual_cell_type"] == "TypeB",
            "Rejected",
            adata_gt.obs["manual_cell_type"],
        )
        with self.assertRaises(ValueError) as ctx:
            run_csde(
                adata_pred=self.adata_pred,
                adata_gt=adata_gt,
                pred_cell_pop_key="cell_type",
                gt_cell_pop_key="manual_cell_type",
                cell_pop_a="TypeA",
                cell_pop_b="TypeB",
                optimizer="gd",
                optimizer_kwargs={"n_iter": 10},
            )
        self.assertIn("manually annotated set", str(ctx.exception))

    def test_importance_weights_wrong_shape(self):
        from csde.model_poisson import PoissonIntercept as InterceptRegression

        x_gt, y_gt = self.adata_gt.X.astype(float), np.zeros(
            len(self.adata_gt), dtype=int
        )
        x_hat = x_gt.copy()
        x_unl = self.adata_pred.X.astype(float)
        y_hat = np.zeros(len(self.adata_gt), dtype=int)
        y_unl = np.zeros(len(self.adata_pred), dtype=int)

        bad_weights = np.ones(len(self.adata_gt) + 5)
        with self.assertRaises(ValueError):
            InterceptRegression(
                inputs_gt=(x_gt, y_gt),
                inputs_hat=(x_hat, y_hat),
                inputs_unl=(x_unl, y_unl),
                importance_weights=bad_weights,
            )


COI = "macrophage"

# cell_id -> (automated cell_type, spatial_group)
CELLS = {
    "c0": (COI, 0),           # accept  -> 0
    "c1": (COI, 1),           # accept  -> 1
    "c2": (COI, 1),           # reject  -> 2
    "c3": (COI, 1),           # correct away from COI -> 2
    "c4": ("fibroblast", 1),  # correct into COI -> 1   (promotion)
    "c5": ("fibroblast", 0),  # correct into COI -> 0   (promotion)
    "c6": ("fibroblast", 0),  # accept  -> 2
    "c7": (COI, 1),           # unannotated
    "c8": ("fibroblast", 0),  # unannotated
}
ANNOTATIONS = {
    "c0": {"action": "accept", "label": None},
    "c1": {"action": "accept", "label": None},
    "c2": {"action": "reject", "label": None},
    "c3": {"action": "correct", "label": "fibroblast"},
    "c4": {"action": "correct", "label": COI},
    "c5": {"action": "correct", "label": COI},
    "c6": {"action": "accept", "label": None},
}


def _build_sdata():
    cell_ids = list(CELLS)
    obs = pd.DataFrame(
        {
            "cell_type": [CELLS[c][0] for c in cell_ids],
            "spatial_group": [CELLS[c][1] for c in cell_ids],
        },
        index=cell_ids,
    )
    adata = anndata.AnnData(X=np.ones((len(cell_ids), 4)), obs=obs)
    adata.var_names = [f"Gene_{i}" for i in range(4)]
    # prepare_csde_inputs only ever does sdata["table"].
    return {"table": adata}


def _write_annotation_dir(tmpdir: Path, annotations: dict) -> Path:
    with open(tmpdir / "config.json", "w") as f:
        json.dump(
            {"cell_type_key": "cell_type", "cell_type_of_interest": COI},
            f,
        )
    pd.DataFrame(
        {
            "cell_id": list(annotations),
            "cell_type": [CELLS[c][0] for c in annotations],
            "sampling_weight": [1.0] * len(annotations),
        }
    ).to_csv(tmpdir / "metadata.csv", index=False)
    with open(tmpdir / "annotations.json", "w") as f:
        json.dump(annotations, f)
    return tmpdir


class TestPrepareCsdeInputs(unittest.TestCase):
    """Label construction: the path where a bug yields wrong DE rather than a crash."""

    def _run(self, annotations=None):
        with tempfile.TemporaryDirectory() as tmp:
            annotation_dir = _write_annotation_dir(
                Path(tmp), ANNOTATIONS if annotations is None else annotations
            )
            return prepare_csde_inputs(
                annotation_dir=annotation_dir,
                sdata=_build_sdata(),
                n_cells_expressed_threshold=1,
            )

    def test_label_construction(self):
        obs = self._run()["adata_gt"].obs
        expected = {"c0": 0, "c1": 1, "c2": 2, "c3": 2, "c4": 1, "c5": 0, "c6": 2}
        self.assertEqual(obs["annotation"].to_dict(), expected)

    def test_promotion_changes_the_label(self):
        # The case an accept/reject scheme cannot express: automated says 2,
        # manual curation moves the cell into a compared group.
        obs = self._run()["adata_gt"].obs
        for cell_id, expected_group in (("c4", 1), ("c5", 0)):
            self.assertEqual(obs.loc[cell_id, "prediction"], 2)
            self.assertEqual(obs.loc[cell_id, "annotation"], expected_group)
            self.assertEqual(obs.loc[cell_id, "manual_cell_type"], COI)

    def test_manual_cell_type_resolution(self):
        obs = self._run()["adata_gt"].obs
        self.assertEqual(obs.loc["c0", "manual_cell_type"], COI)  # accepted
        self.assertEqual(obs.loc["c3", "manual_cell_type"], "fibroblast")  # corrected
        self.assertTrue(pd.isna(obs.loc["c2", "manual_cell_type"]))  # rejected

    def test_summary_counts(self):
        summary = self._run()["summary"]
        self.assertEqual(summary["n_annotated"], 7)
        self.assertEqual(summary["n_accept"], 3)
        self.assertEqual(summary["n_correct"], 3)
        self.assertEqual(summary["n_reject"], 1)
        self.assertEqual(summary["n_promoted"], 2)  # c4, c5
        self.assertEqual(summary["n_demoted"], 2)  # c2, c3

    def test_unannotated_cells_go_to_adata_other(self):
        inputs = self._run()
        self.assertEqual(set(inputs["adata_other"].obs_names), {"c7", "c8"})
        self.assertEqual(inputs["adata_other"].obs["prediction"].to_dict(), {"c7": 1, "c8": 2})

    def test_unknown_corrected_label_raises(self):
        annotations = dict(ANNOTATIONS)
        annotations["c4"] = {"action": "correct", "label": "not_a_cell_type"}
        with self.assertRaises(ValueError) as ctx:
            self._run(annotations)
        self.assertIn("not_a_cell_type", str(ctx.exception))

    def test_malformed_annotations_raise(self):
        cases = [
            {"c0": True},  # the old boolean format is no longer accepted
            {"c0": {"action": "maybe", "label": None}},
            {"c0": {"action": "correct", "label": None}},
            {"c0": {"action": "accept", "label": "fibroblast"}},
        ]
        for annotations in cases:
            with self.subTest(annotations=annotations):
                with self.assertRaises(ValueError):
                    self._run(annotations)

    def test_accept_reject_only_matches_previous_behaviour(self):
        # Strict extension: with no corrections, labels equal the old
        # (is_coi & is_correct & spatial_group) formula.
        annotations = {
            cell_id: {
                "action": "accept" if cell_id in ("c0", "c1", "c6") else "reject",
                "label": None,
            }
            for cell_id in ANNOTATIONS
        }
        obs = self._run(annotations)["adata_gt"].obs
        expected = {}
        for cell_id in annotations:
            cell_type, spatial_group = CELLS[cell_id]
            is_correct = annotations[cell_id]["action"] == "accept"
            expected[cell_id] = (
                spatial_group if (cell_type == COI and is_correct) else 2
            )
        self.assertEqual(obs["annotation"].to_dict(), expected)
        self.assertEqual(self._run(annotations)["summary"]["n_promoted"], 0)


if __name__ == "__main__":
    unittest.main()
