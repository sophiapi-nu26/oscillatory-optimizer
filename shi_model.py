"""Shi repelling-with-information model construction and diagnostics.

W = I - alpha L_+ + beta L_- - D
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, eigvalsh


def compute_degrees(A_plus: np.ndarray, A_minus: np.ndarray):
    return A_plus.sum(axis=1), A_minus.sum(axis=1)


def compute_laplacians(A_plus: np.ndarray, A_minus: np.ndarray):
    d_plus, d_minus = compute_degrees(A_plus, A_minus)
    L_plus = np.diag(d_plus) - A_plus
    L_minus = np.diag(d_minus) - A_minus
    return L_plus, L_minus


def build_information_matrix(c: np.ndarray) -> np.ndarray:
    return np.diag(np.asarray(c, dtype=float))


def build_shi_repelling_matrix(
    A_plus: np.ndarray,
    A_minus: np.ndarray,
    alpha: float,
    beta: float,
    c: np.ndarray,
) -> np.ndarray:
    n = A_plus.shape[0]
    L_plus, L_minus = compute_laplacians(A_plus, A_minus)
    D = build_information_matrix(c)
    return np.eye(n) - alpha * L_plus + beta * L_minus - D


def compute_threshold(
    alpha: float,
    L_plus: np.ndarray,
    L_minus: np.ndarray,
    D: np.ndarray,
    eps: float = 1e-10,
) -> float | None:
    """beta_thresh(c) = inf_{x: x^T L_- x > 0} (x^T A x) / (x^T L_- x),
    where A = alpha L_+ + D.

    Computed as the smallest generalized eigenvalue of (A, L_-) restricted to
    range(L_-). Returns +inf if L_- = 0; None on numerical failure.
    """
    A = alpha * L_plus + D
    if np.allclose(L_minus, 0):
        return float("inf")
    try:
        # Restrict to range(L_-): take eigenvectors of L_- with positive eigenvalues.
        w, V = eigh(L_minus)
        keep = w > eps
        if not np.any(keep):
            return float("inf")
        Vk = V[:, keep]
        A_r = Vk.T @ A @ Vk
        L_r = Vk.T @ L_minus @ Vk
        gen_eigs = eigvalsh(A_r, L_r)
        return float(gen_eigs.min())
    except Exception:
        return None


def compute_stability_diagnostics(
    W: np.ndarray,
    omega: float,
    A_plus: np.ndarray | None = None,
    alpha: float | None = None,
    c: np.ndarray | None = None,
) -> dict:
    n = W.shape[0]
    eigs = np.linalg.eigvals(W)
    rho = float(np.max(np.abs(eigs)))
    is_symmetric = np.allclose(W, W.T)
    if is_symmetric:
        real_eigs = eigvalsh(W)
        lambda_min = float(real_eigs.min())
        lambda_max = float(real_eigs.max())
    else:
        lambda_min = float(np.min(eigs.real))
        lambda_max = float(np.max(eigs.real))

    M = np.exp(1j * omega) * np.eye(n) - W
    cond = float(np.linalg.cond(M))

    diag = {
        "rho": rho,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "is_stable": rho < 1.0,
        "condition_number": cond,
        "is_symmetric": is_symmetric,
    }
    if A_plus is not None and alpha is not None and c is not None:
        d_plus = A_plus.sum(axis=1)
        baseline = 1.0 - alpha * d_plus - np.asarray(c)
        diag["baseline_self_weight"] = baseline
        diag["baseline_self_weight_min"] = float(baseline.min())
        diag["baseline_self_weight_ok"] = bool((baseline >= 0).all())
    return diag
