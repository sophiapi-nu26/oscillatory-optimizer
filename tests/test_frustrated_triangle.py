"""Frustrated-triangle three-phase optimum.

Implements the spec at numerics/frustrated_triangle_unit_test_spec.md.

The test confirms that on the minimal frustrated network (K_3 with all three
edges negative), the optimizer recovers a genuinely non-binary phase optimum:
three magnitudes at the constraint and three phases separated by 2 pi / 3.

It is designed to catch optimizers that collapse to binary 0 / pi lag patterns.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import optimize_signal
from response import build_quadratic_form, frequency_response
from shi_model import build_shi_repelling_matrix


def _setup():
    A_plus = np.zeros((3, 3))
    A_minus = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    alpha = 0.1
    beta = 1.0 / 9.0
    c = np.full(3, 0.5)
    return A_plus, A_minus, alpha, beta, c


def test_frustrated_triangle_W_matches_spec():
    A_plus, A_minus, alpha, beta, c = _setup()
    W = build_shi_repelling_matrix(A_plus, A_minus, alpha, beta, c)
    expected = np.array([
        [13/18, -1/9, -1/9],
        [-1/9, 13/18, -1/9],
        [-1/9, -1/9, 13/18],
    ])
    assert np.allclose(W, expected, atol=1e-12)


def test_frustrated_triangle_W_eigenvalues():
    A_plus, A_minus, alpha, beta, c = _setup()
    W = build_shi_repelling_matrix(A_plus, A_minus, alpha, beta, c)
    eigs = sorted(np.linalg.eigvalsh(W).tolist())
    # Expected: 1/2, 5/6, 5/6
    assert np.isclose(eigs[0], 0.5, atol=1e-12)
    assert np.isclose(eigs[1], 5/6, atol=1e-12)
    assert np.isclose(eigs[2], 5/6, atol=1e-12)


def test_frustrated_triangle_is_stable():
    A_plus, A_minus, alpha, beta, c = _setup()
    W = build_shi_repelling_matrix(A_plus, A_minus, alpha, beta, c)
    rho = float(np.max(np.abs(np.linalg.eigvals(W))))
    assert rho < 1.0
    assert np.isclose(rho, 5/6, atol=1e-12)


def test_frustrated_triangle_variance_linf_three_phase_optimum():
    """The headline test: linf-constrained variance objective recovers
    three magnitudes near 1 and three phases separated by 120 degrees."""
    A_plus, A_minus, alpha, beta, c = _setup()
    n = 3
    W = build_shi_repelling_matrix(A_plus, A_minus, alpha, beta, c)
    D = np.diag(c)
    omega = np.pi / 2
    B = frequency_response(W, D, omega)
    Q = build_quadratic_form(B, "variance", n)

    res = optimize_signal(
        Q,
        norm_type="linf",
        R=1.0,
        num_restarts=100,
        max_iter=1000,
        seed=42,
    )
    a = res.a_star
    r = res.r_star

    # Check 1: magnitudes near 1
    assert r.min() >= 1.0 - 1e-3, f"min r = {r.min():.6f} < 1 - 1e-3"

    # Check 2: zero complex sum (the key test that distinguishes from binary)
    sum_abs = float(np.abs(a.sum()))
    assert sum_abs <= 1e-3, f"|sum a| = {sum_abs:.6f} should be 0 (was 1 for binary)"

    # Check 3: pairwise 120-degree separation
    for i in range(3):
        for j in range(i + 1, 3):
            cos_diff = float(np.cos(np.angle(a[i]) - np.angle(a[j])))
            assert abs(cos_diff + 0.5) <= 1e-3, (
                f"cos(phi_{i} - phi_{j}) = {cos_diff:.6f}, expected -0.5"
            )

    # Check 4: objective near 9/122
    expected_J = 9.0 / 122.0
    assert abs(res.J_star - expected_J) <= 1e-4, (
        f"J* = {res.J_star:.8f}, expected {expected_J:.8f}"
    )

    # Check 5: strictly above the best binary-lag value 4/61
    binary_J = 4.0 / 61.0
    assert res.J_star >= binary_J + 1e-3, (
        f"J* = {res.J_star:.8f} should beat binary {binary_J:.8f} by >= 1e-3"
    )


def test_frustrated_triangle_robust_across_seeds():
    """Sanity-check that the optimizer recovers the three-phase optimum
    across a range of random seeds, not just one lucky one."""
    A_plus, A_minus, alpha, beta, c = _setup()
    n = 3
    W = build_shi_repelling_matrix(A_plus, A_minus, alpha, beta, c)
    D = np.diag(c)
    B = frequency_response(W, D, np.pi / 2)
    Q = build_quadratic_form(B, "variance", n)

    expected_J = 9.0 / 122.0
    for seed in [0, 1, 7, 13, 42, 99, 2024]:
        res = optimize_signal(
            Q, norm_type="linf", R=1.0, num_restarts=100, max_iter=500, seed=seed
        )
        assert abs(res.J_star - expected_J) <= 1e-4, (
            f"seed={seed}: J* = {res.J_star:.8f}, expected {expected_J:.8f}"
        )
        assert abs(res.a_star.sum()) <= 1e-3, (
            f"seed={seed}: |sum a| = {abs(res.a_star.sum()):.6f}"
        )
