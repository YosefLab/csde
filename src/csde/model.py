from typing import Any, Dict, List, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro.distributions import Normal, Poisson
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from csde.optimization import _zstat_generic2, optimize_ppi, optimize_ppi_gd

jax.config.update("jax_enable_x64", True)


class PPIAbstractClass:
    """
    Abstract base class for Prediction-Powered Inference (PPI) models.
    """

    def __init__(
        self,
        inputs_gt: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
        inputs_hat: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
        inputs_unl: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
        lambd_mode: str = "overall",
    ):
        """
        Initialize the PPI model.

        Args:
            inputs_gt: Ground truth data (features, labels).
            inputs_hat: Predicted data for the labeled set (features, predicted labels).
            inputs_unl: Unlabeled data (features, predicted labels).
            lambd_mode: Mode for lambda parameter ('overall' or 'element').
        """
        self.inputs_gt = inputs_gt
        self.inputs_hat = inputs_hat
        self.inputs_unl = inputs_unl

        inputs_are_tuples = isinstance(inputs_gt, tuple)
        if inputs_are_tuples:
            self.n = inputs_gt[0].shape[0]
            self.N = inputs_unl[0].shape[0]
        else:
            self.n = self.inputs_gt.shape[0]
            self.N = self.inputs_unl.shape[0]
        self.r = float(self.n) / self.N
        self.theta = None
        self.sigma = None
        self.hessian = None
        self.v = None
        self.lambd_mode = lambd_mode
        self.lambd_ = None

    def get_asymptotic_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the asymptotic distribution of the estimator.

        Returns:
            Tuple containing the point estimate (theta) and the covariance matrix (sigma).
        """
        self.sigma = self.compute_sigma(self.lambd_)
        return self.theta, self.sigma

    def compute_sigma(self, lambd: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute the covariance matrix of the estimator.
        """
        grad_f_unl = self.grad_fn(self.inputs_unl)
        grad_f_hat = self.grad_fn(self.inputs_hat)
        grad_f_all = np.vstack([grad_f_hat, grad_f_unl])
        grad_f_gt = self.grad_fn(self.inputs_gt)

        grad_f_ = grad_f_all - grad_f_all.mean(axis=0)

        # Handle lambda broadcasting if necessary
        if self.lambd_mode == "element" and isinstance(lambd, np.ndarray):
            # This part was implementation specific in subclass, but generalized here based on pattern
            # Assuming lambd matches gradient dimensions or is handled in subclass override
            pass

        # Base implementation for scalar lambda, override in subclass if needed
        vf = (lambd**2) * (grad_f_.T @ grad_f_) / (self.n + self.N)
        rect_ = grad_f_gt - lambd * grad_f_hat
        rect_ = rect_ - rect_.mean(axis=0)
        vdelta = (rect_.T @ rect_) / self.n
        v = vdelta + (self.r * vf)

        hess = self.hessian_fn(self.inputs_gt)
        self.hessian = hess
        self.v = v
        return self._compute_sigma(hess, v, self.n)

    @staticmethod
    def _compute_sigma(hess: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
        """
        Compute the asymptotic covariance matrix of the parameter estimates.
        """
        inv_hess = np.linalg.pinv(hess)
        sigma_ = inv_hess @ v @ inv_hess
        sigma_ = sigma_ / n
        return sigma_

    def get_lambda(
        self,
        lambd_0: float = 0.5,
        idx_to_optimize: Optional[Union[int, List[int]]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Estimate the optimal lambda parameter.
        """
        print("get point estimate ...")
        self.theta = self.get_pointestimate(lambd_=lambd_0)
        print("done")

        hess = self.hessian_fn(self.inputs_gt)

        inv_hess = np.linalg.pinv(hess)
        grad_f_unl = self.grad_fn(self.inputs_unl)
        grad_f_hat = self.grad_fn(self.inputs_hat)
        grad_f_all = np.vstack([grad_f_hat, grad_f_unl])
        grad_f_gt = self.grad_fn(self.inputs_gt)

        grad_f_hat_ = grad_f_hat - grad_f_hat.mean(0)
        grad_f_gt_ = grad_f_gt - grad_f_gt.mean(0)
        cov1 = (grad_f_hat_.T @ grad_f_gt_) / self.n
        cov2 = (grad_f_gt_.T @ grad_f_hat_) / self.n

        grad_f_ = grad_f_all - grad_f_all.mean(axis=0)
        vf = (grad_f_.T @ grad_f_) / (self.n + self.N)
        num = inv_hess @ (cov1 + cov2) @ inv_hess
        denom = 2 * (1.0 + self.r) * (inv_hess @ vf @ inv_hess)
        if self.lambd_mode == "element":
            lambd_star = num / denom
            return np.diag(lambd_star)
        elif idx_to_optimize is not None:
            print("optimize lambda for a single theta comp.")
            if isinstance(idx_to_optimize, int):
                return (
                    num[idx_to_optimize, idx_to_optimize]
                    / denom[idx_to_optimize, idx_to_optimize]
                )
            else:
                return np.trace(num[idx_to_optimize, :][:, idx_to_optimize]) / np.trace(
                    denom[idx_to_optimize, :][:, idx_to_optimize]
                )
        else:
            return np.trace(num) / np.trace(denom)

    def get_pointestimate(self, lambd_: float) -> np.ndarray:
        raise NotImplementedError

    def grad_fn(self, inputs: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def hessian_fn(self, inputs: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class RegressionInterceptModel(nn.Module):
    """
    Flax module for the intercept regression model.
    """

    n_classes: int
    n_features: int
    mu_prior_std: Union[float, jnp.ndarray]
    n_obs_real: int
    family: str

    def setup(self):
        self.mu0 = self.param("mu0", nn.initializers.normal(), (self.n_features))
        self.mu = self.param(
            "mu", nn.initializers.normal(), (self.n_classes - 1, self.n_features)
        )

    def __call__(self, x, y):
        y_ = y.astype(jnp.int32)
        mu_placeholder = jnp.zeros_like(self.mu0)
        mu = jnp.concatenate([mu_placeholder[None], self.mu], axis=0)
        y_oh = jnp.eye(self.n_classes)[y_]
        mus_ = y_oh @ mu + self.mu0

        if self.family == "poisson":
            rates = jnp.exp(mus_)
            log_px_c_unsummed = Poisson(rate=rates).log_prob(x)
            log_px_c = log_px_c_unsummed.sum(axis=-1)
        elif self.family == "gaussian":
            log_px_c_unsummed = Normal(loc=mus_, scale=1.0).log_prob(x)
            log_px_c = log_px_c_unsummed.sum(axis=-1)
        else:
            raise ValueError(f"Unknown family: {self.family}")

        loss = -log_px_c
        return {
            "loss": loss,
            "loss_unsummed": -log_px_c_unsummed,
        }


class InterceptRegression(PPIAbstractClass):
    """
    Intercept Regression model for spatial differential expression analysis.
    """

    def __init__(
        self,
        mu_prior_std: Optional[Union[float, jnp.ndarray]] = None,
        optimizer: str = "gd",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        family: str = "poisson",
        jit: bool = True,
        **kwargs,
    ):
        """
        Initialize the InterceptRegression model.

        Args:
            mu_prior_std: Prior standard deviation for mu.
            optimizer: Optimization method ('gd' or 'lbfgs').
            optimizer_kwargs: Keyword arguments for the optimizer.
            family: Distribution family ('poisson' or 'gaussian').
            jit: Whether to JIT compile the optimization.
            **kwargs: Arguments passed to PPIAbstractClass (inputs_gt, inputs_hat, inputs_unl).
        """
        super().__init__(**kwargs)

        x_gt, y_gt = self.inputs_gt
        x_hat, y_hat = self.inputs_hat
        x_unl, y_unl = self.inputs_unl
        all_y = np.hstack([y_gt, y_hat, y_unl])
        unique_y = np.unique(all_y)
        self.n_classes = unique_y.shape[0]
        necessary_range = np.arange(self.n_classes)
        assert np.isin(necessary_range, unique_y).all()

        self.inputs_gt = (x_gt, y_gt)
        self.inputs_hat = (x_hat, y_hat)
        self.inputs_unl = (x_unl, y_unl)
        n_obs_real = x_gt.shape[0]

        self.n_features = x_gt.shape[1]
        self.n_params = (self.n_classes - 1) * self.n_features + self.n_features
        self.model = RegressionInterceptModel(
            n_classes=self.n_classes,
            n_features=self.n_features,
            mu_prior_std=mu_prior_std,
            n_obs_real=n_obs_real,
            family=family,
        )
        self.model_params = None

        self.zero_init()
        self.lambd_ = None
        self.log = None
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self.jit = jax.jit if jit else lambda x: x

    def fit(
        self, lambd_: Optional[Union[float, np.ndarray]] = None, refit: bool = False
    ):
        """
        Fit the model parameters.

        Args:
            lambd_: Lambda parameter. If None, it is estimated.
            refit: Whether to re-initialize parameters before fitting.
        """
        if lambd_ is None:
            lambd_ = self.get_lambda()
        print(f"lambda: {lambd_}")
        self.lambd_ = lambd_
        if refit:
            self.zero_init()
        self.theta = self.get_pointestimate(lambd_=lambd_)

    def get_lambda(
        self,
        lambd_0: float = 0.5,
        idx_to_optimize: Optional[Union[int, List[int]]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Estimate the optimal lambda parameter.
        Overriding parent method to handle element-wise lambda specific logic.
        """
        # Call parent to get num and denom matrices/values if needed, but the parent implementation
        # might need access to _construct_contrast for element-wise which is specific to this class.
        # So copying logic from original implementation to be safe and consistent.

        print("get point estimate ...")
        self.theta = self.get_pointestimate(lambd_=lambd_0)
        print("done")

        hess = self.hessian_fn(self.inputs_gt)

        inv_hess = np.linalg.pinv(hess)
        grad_f_unl = self.grad_fn(self.inputs_unl)
        grad_f_hat = self.grad_fn(self.inputs_hat)
        grad_f_all = np.vstack([grad_f_hat, grad_f_unl])
        grad_f_gt = self.grad_fn(self.inputs_gt)

        grad_f_hat_ = grad_f_hat - grad_f_hat.mean(0)
        grad_f_gt_ = grad_f_gt - grad_f_gt.mean(0)
        cov1 = (grad_f_hat_.T @ grad_f_gt_) / self.n
        cov2 = (grad_f_gt_.T @ grad_f_hat_) / self.n

        grad_f_ = grad_f_all - grad_f_all.mean(axis=0)
        vf = (grad_f_.T @ grad_f_) / (self.n + self.N)
        num = inv_hess @ (cov1 + cov2) @ inv_hess
        denom = 2 * (1.0 + self.r) * (inv_hess @ vf @ inv_hess)

        if self.lambd_mode == "element":
            lambd_design = [
                np.where(self._construct_contrast(feature_id, 1))[0][0]
                for feature_id in range(self.n_features)
            ]
            lambd_star = num / denom
            lambd_star = np.diag(lambd_star)
            lambd_star = lambd_star[lambd_design]
            return lambd_star
        elif idx_to_optimize is not None:
            print("optimize lambda for a single theta comp.")
            if isinstance(idx_to_optimize, int):
                return (
                    num[idx_to_optimize, idx_to_optimize]
                    / denom[idx_to_optimize, idx_to_optimize]
                )
            else:
                return np.trace(num[idx_to_optimize, :][:, idx_to_optimize]) / np.trace(
                    denom[idx_to_optimize, :][:, idx_to_optimize]
                )
        else:
            return np.trace(num) / np.trace(denom)

    def get_pointestimate(self, lambd_: Union[float, np.ndarray]) -> np.ndarray:
        x_gt, y_gt = self.inputs_gt
        x_hat, y_hat = self.inputs_hat
        x_unl, y_unl = self.inputs_unl

        model_params0 = self.model_params if self.model_params is not None else None
        if self.optimizer == "lbfgs":
            model_params = optimize_ppi(
                self.model,
                lambd_=lambd_,
                x_gt=x_gt,
                y_gt=y_gt,
                x_hat=x_hat,
                y_hat=y_hat,
                x_unl=x_unl,
                y_unl=y_unl,
                model_params0=model_params0,
                **self.optimizer_kwargs,
            )
        elif self.optimizer == "gd":
            model_params = optimize_ppi_gd(
                self.model,
                lambd_=lambd_,
                x_gt=x_gt,
                y_gt=y_gt,
                x_hat=x_hat,
                y_hat=y_hat,
                x_unl=x_unl,
                y_unl=y_unl,
                model_params0=model_params0,
                **self.optimizer_kwargs,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        self.model_params = model_params

        mu = np.array(model_params["params"]["mu"].reshape(-1))
        mu0 = np.array(model_params["params"]["mu0"])
        return np.hstack([mu, mu0])

    def compute_sigma(self, lambd: Union[float, np.ndarray]) -> np.ndarray:
        # Override to handle element-wise lambda and specific broadcasting
        grad_f_unl = self.grad_fn(self.inputs_unl)
        grad_f_hat = self.grad_fn(self.inputs_hat)
        grad_f_all = np.vstack([grad_f_hat, grad_f_unl])
        grad_f_gt = self.grad_fn(self.inputs_gt)

        grad_f_ = grad_f_all - grad_f_all.mean(axis=0)
        if self.lambd_mode == "element":
            mask = self.idx_to_feat()
            lambd_good = lambd[mask]
        else:
            lambd_good = lambd
        grad_f_ = lambd_good * grad_f_
        vf = (grad_f_.T @ grad_f_) / (self.n + self.N)
        rect_ = grad_f_gt - lambd_good * grad_f_hat
        rect_ = rect_ - rect_.mean(axis=0)
        vdelta = (rect_.T @ rect_) / self.n
        v = vdelta + (self.r * vf)

        hess = self.hessian_fn(self.inputs_gt)
        self.hessian = hess
        self.v = v
        return self._compute_sigma(hess, v, self.n)

    def grad_fn(
        self, inputs: Tuple[np.ndarray, np.ndarray], batch_size: int = 128
    ) -> np.ndarray:
        x, y = inputs
        n_obs = x.shape[0]

        def likelihood(model_params, x, y):
            return self.model.apply(model_params, x, y)["loss"]

        all_grads = np.zeros((n_obs, self.n_params))
        for i in tqdm(range(0, n_obs, batch_size), desc="Gradient computation"):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            n_obs_batch = x_batch.shape[0]
            score = self.jit(jax.jacfwd(likelihood))
            grads = score(self.model_params, x_batch, y_batch)
            grad_mu = np.array(grads["params"]["mu"].reshape(n_obs_batch, -1))
            grad_mu0 = np.array(grads["params"]["mu0"].reshape(n_obs_batch, -1))
            all_grads[i:i+batch_size] = np.hstack([grad_mu, grad_mu0])
        return np.array(all_grads)

    def _construct_contrast(self, feature_id: int, idx_a: int) -> np.ndarray:
        mu_contrast = np.zeros((self.n_classes - 1, self.n_features))
        mu_contrast[idx_a - 1, feature_id] = 1.0

        mu0_contrast = np.zeros(self.n_features)
        contrast = np.hstack([mu_contrast.flatten(), mu0_contrast])
        return contrast.astype(int)

    def idx_to_feat(self) -> np.ndarray:
        mu_identifier = np.ones((self.n_classes - 1, self.n_features))
        mu_identifier = mu_identifier * np.arange(self.n_features)

        mu0_identifier = np.arange(self.n_features)
        identifier = np.hstack([mu_identifier.flatten(), mu0_identifier])
        return identifier.astype(int)

    def construct_contrast(self, idx_a: int) -> np.ndarray:
        _contrast = [
            self._construct_contrast(feature_id, idx_a)
            for feature_id in range(self.n_features)
        ]
        _contrast = np.vstack(_contrast)
        return _contrast

    def get_beta(self, idx_a: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if idx_a == 0:
            raise ValueError("`class_a` cannot be the reference class.")

        contrast = self.construct_contrast(idx_a)
        beta = contrast @ self.theta
        cov = contrast @ self.sigma @ contrast.T
        return beta, cov, contrast

    def _get_param_id(
        self, feature_id: int = None, class_id: int = None, param_type: str = None
    ) -> int:
        n_params_mu = self.n_features * (self.n_classes - 1)
        if param_type == "mu":
            return ((class_id - 1) * self.n_features) + feature_id
        elif param_type == "mu0":
            return n_params_mu + feature_id
        return 0

    def _get_param_mask(self, feature_id: int) -> np.ndarray:
        mu_indices = [
            self._get_param_id(
                feature_id=feature_id, class_id=class_id, param_type="mu"
            )
            for class_id in range(1, self.n_classes)
        ]
        mu0_indices = [self._get_param_id(feature_id=feature_id, param_type="mu0")]
        indices_to_keep = np.hstack([mu_indices, mu0_indices])
        return indices_to_keep

    def test_differential_expression(
        self,
        idx_a: int,
        feature_names: Optional[List[str]] = None,
        cond_thresh: float = np.inf,
    ) -> pd.DataFrame:
        """
        Perform differential expression testing.

        Args:
            idx_a: The index of the target class (1 or 2). Note: 0 is the reference class.
            feature_names: List of feature names.
            cond_thresh: Condition number threshold for Hessian.

        Returns:
            DataFrame containing the results (p-values, log-fold changes, etc.).
        """
        idx_a_ = idx_a - 1
        results = []
        for feature_id in range(self.n_features):
            mask_ = self._get_param_mask(feature_id)
            v_ = self.v[mask_][:, mask_]
            hess_ = self.hessian[mask_][:, mask_]

            beta = self.theta[mask_]
            cov = self._compute_sigma(hess_, v_, self.n)
            cond = np.linalg.cond(hess_)

            is_cond_below_thresh = cond >= cond_thresh
            if is_cond_below_thresh:
                print(f"Hessian is below threshold for feature {feature_id}")
                pval = 1.0
            else:
                _, pval = _zstat_generic2(
                    beta[idx_a_], np.sqrt(cov[idx_a_, idx_a_]), alternative="two-sided"
                )
            results.append(
                {
                    "pval": pval,
                    "hess": hess_[idx_a_, idx_a_],
                    "beta": beta[idx_a_],
                    "cov": cov[idx_a_, idx_a_],
                    "hess_cond": cond,
                }
            )
        res = pd.DataFrame(results)
        res["pval"].iloc[np.isnan(res["pval"])] = 1.0
        res["padj"] = multipletests(res["pval"], method="fdr_bh")[1]
        res["is_significant_005"] = res["padj"] < 0.05
        if feature_names is not None:
            res["feature_name"] = feature_names
        return res

    def zero_init(self):
        mu = np.zeros((self.n_classes - 1, self.n_features))
        mu0 = np.zeros(self.n_features)
        params = {
            "params": {
                "mu": jnp.array(mu, dtype=jnp.float64),
                "mu0": jnp.array(mu0, dtype=jnp.float64),
            }
        }
        self.model_params = params

    def hessian_fn(
        self, inputs: Tuple[np.ndarray, np.ndarray], device=None
    ) -> np.ndarray:
        x, y = inputs

        if device is None:
            device = jax.devices("cpu")[0]
        model_params_ = jax.device_put(self.model_params, device)
        n_obs = x.shape[0]
        obs_ids = np.arange(n_obs)
        model_ = self.model

        def likelihood(model_params, x, y):
            return model_.apply(model_params, x, y)["loss"]

        hess_fn = jax.hessian(likelihood)

        def process_hess(x, y):
            hess_ = hess_fn(model_params_, x, y)
            mu_mu = (
                hess_["params"]["mu"]["params"]["mu"]
                .mean(0)
                .reshape(
                    (self.n_classes - 1) * self.n_features,
                    (self.n_classes - 1) * self.n_features,
                )
            )
            mu_mu0 = (
                hess_["params"]["mu"]["params"]["mu0"]
                .mean(0)
                .reshape((self.n_classes - 1) * self.n_features, self.n_features)
            )
            mu0_mu0 = (
                hess_["params"]["mu0"]["params"]["mu0"]
                .mean(0)
                .reshape(self.n_features, self.n_features)
            )
            blk = jnp.block(
                [
                    [mu_mu, mu_mu0],
                    [mu_mu0.T, mu0_mu0],
                ]
            )
            return blk

        hessian = np.zeros((self.n_params, self.n_params), dtype=np.float64)
        for obs_id in tqdm(obs_ids, desc="Hessian computation"):
            x_ = jnp.array(x[[obs_id]], dtype=jnp.float64)
            y_ = jnp.array(y[[obs_id]], dtype=jnp.int32)
            x_obs = jax.device_put(x_, device)
            y_obs = jax.device_put(y_, device)
            hess_ = process_hess(x_obs, y_obs)
            hessian += hess_ / float(n_obs)
        return hessian
