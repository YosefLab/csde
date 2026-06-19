import time
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.stats as stats
from tqdm import trange


def _zstat_generic2(value: float, std: float, alternative: str) -> Tuple[float, float]:
    """
    Compute z-statistic and p-value.

    Args:
        value: The estimated value (beta).
        std: The standard deviation of the estimate.
        alternative: The alternative hypothesis ("two-sided", "larger", "smaller").

    Returns:
        Tuple containing the z-statistic and the p-value.
    """
    zstat = value / std
    if alternative in ["two-sided", "2-sided", "2s"]:
        pvalue = stats.norm.sf(np.abs(zstat)) * 2
    elif alternative in ["larger", "l"]:
        pvalue = stats.norm.sf(zstat)
    elif alternative in ["smaller", "s"]:
        pvalue = stats.norm.cdf(zstat)
    else:
        raise ValueError("invalid alternative")
    return zstat, pvalue


def optimize_ppi(
    model: Any,
    x_gt: jnp.ndarray,
    y_gt: jnp.ndarray,
    x_hat: jnp.ndarray,
    y_hat: jnp.ndarray,
    x_unl: jnp.ndarray,
    y_unl: jnp.ndarray,
    model_params0: Optional[Any] = None,
    lambd_: float = 1.0,
    tol: float = 1e-3,
    jit: bool = True,
    **lbfgs_kwargs,
) -> Any:
    """
    Optimize the PPI objective using L-BFGS.
    """
    jitter = jax.jit if jit else lambda x: x
    x_gt = jax.device_put(x_gt)
    y_gt = jax.device_put(y_gt)
    x_hat = jax.device_put(x_hat)
    y_hat = jax.device_put(y_hat)
    x_unl = jax.device_put(x_unl)
    y_unl = jax.device_put(y_unl)

    x0 = jnp.ones((32, x_gt.shape[1]))
    y0 = jnp.ones(32, dtype=jnp.int32)
    if model_params0 is not None:
        theta = model_params0
        print("using existing model params as initial point")
    else:
        theta = model.init(jax.random.PRNGKey(0), x0, y0)
    opt = optax.lbfgs(**lbfgs_kwargs)
    opt_state = opt.init(theta)

    def loss_fn(zetas):
        loss_gt = model.apply(zetas, x_gt, y_gt)["loss"].mean()
        loss_hat = model.apply(zetas, x_hat, y_hat)["loss"].mean()
        loss_unl = model.apply(zetas, x_unl, y_unl)["loss"].mean()
        loss = (lambd_ * loss_unl) - (lambd_ * loss_hat) + loss_gt
        return loss

    value_and_grad_fn = jitter(optax.value_and_grad_from_state(loss_fn))
    previous_loss = 1e6
    print("lambda:", lambd_)
    for _ in range(100):
        start = time.time()
        loss, grad = value_and_grad_fn(theta, state=opt_state)
        updates, opt_state = opt.update(
            grad, opt_state, theta, value=loss, grad=grad, value_fn=loss_fn
        )
        theta = optax.apply_updates(theta, updates)
        stopping_criterion = np.abs(loss - previous_loss)
        if stopping_criterion < tol:
            break
        previous_loss = loss
        end = time.time() - start
        print(f"loss: {loss}; stopping criterion: {stopping_criterion}; time: {end}")
    return theta


def optimize_ppi_gd(
    model: Any,
    x_gt: jnp.ndarray,
    y_gt: jnp.ndarray,
    x_hat: jnp.ndarray,
    y_hat: jnp.ndarray,
    x_unl: jnp.ndarray,
    y_unl: jnp.ndarray,
    w: Optional[jnp.ndarray] = None,
    model_params0: Optional[Any] = None,
    lambd_: float = 1.0,
    tol: float = 1e-3,
    n_iter: int = 10000,
    optimizer: str = "adam",
    learning_rate: float = 0.01,
    mode: str = "overall",
    jit: bool = True,
    **kwargs,
) -> Any:
    """
    Optimize the PPI objective using Gradient Descent (Adam or SGD).
    """
    jitter = jax.jit if jit else lambda x: x
    tol_ = np.inf if tol is None else tol
    x_gt = jax.device_put(jnp.array(x_gt, dtype=jnp.float64))
    y_gt = jax.device_put(jnp.array(y_gt, dtype=jnp.int32))
    x_hat = jax.device_put(jnp.array(x_hat, dtype=jnp.float64))
    y_hat = jax.device_put(jnp.array(y_hat, dtype=jnp.int32))
    x_unl = jax.device_put(jnp.array(x_unl, dtype=jnp.float64))
    y_unl = jax.device_put(jnp.array(y_unl, dtype=jnp.int32))
    if w is not None:
        w = jax.device_put(jnp.array(w, dtype=jnp.float64))

    x0 = jnp.ones((32, x_gt.shape[1]), dtype=jnp.float64)
    y0 = jnp.ones(32, dtype=jnp.int32)
    if model_params0 is not None:
        theta = model_params0
        print("using existing model params as initial point")
    else:
        theta = model.init(jax.random.PRNGKey(0), x0, y0)
    if optimizer == "adam":
        opt = optax.adam(learning_rate=learning_rate, **kwargs)
    elif optimizer == "gd":
        opt = optax.sgd(learning_rate=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    opt_state = opt.init(theta)

    lambd_ = jax.device_put(lambd_)

    def step_fn(theta_, opt_state_, x_gt_, y_gt_, x_hat_, y_hat_, x_unl_, y_unl_):
        def loss_fn(zetas):
            loss_gt = model.apply(zetas, x_gt_, y_gt_, w=w)["loss_unsummed"].mean(0)
            loss_hat = model.apply(zetas, x_hat_, y_hat_, w=w)["loss_unsummed"].mean(0)
            loss_unl = model.apply(zetas, x_unl_, y_unl_)["loss_unsummed"].mean(0)
            loss = (lambd_ * loss_unl) - (lambd_ * loss_hat) + loss_gt
            loss = loss.sum(-1)

            # loss_gt = model.apply(zetas, x_gt_, y_gt_, w=w)["loss"].mean()
            # loss_hat = model.apply(zetas, x_hat_, y_hat_, w=w)["loss"].mean()
            # loss_unl = model.apply(zetas, x_unl_, y_unl_)["loss"].mean()
            # loss = (lambd_ * loss_unl) - (lambd_ * loss_hat) + loss_gt
            return loss

        loss, grad = jax.value_and_grad(loss_fn)(theta_)
        updates, opt_state_ = opt.update(grad, opt_state_, theta_)
        theta_ = optax.apply_updates(theta_, updates)
        return theta_, opt_state_, loss

    compiled_step = jitter(step_fn)

    previous_loss = 1e6
    print("lambda:", lambd_)
    print("tol:", tol_)
    pbar = trange(n_iter)
    for _ in pbar:
        theta, opt_state, loss = compiled_step(
            theta, opt_state, x_gt, y_gt, x_hat, y_hat, x_unl, y_unl
        )
        stopping_criterion = np.abs(loss - previous_loss)

        if np.allclose(loss, previous_loss, atol=tol_, rtol=0):
            print("stopping criterion met: ", stopping_criterion)
            break
        previous_loss = loss
        pbar.set_postfix(loss=loss, stopping_criterion=stopping_criterion)
    return theta
