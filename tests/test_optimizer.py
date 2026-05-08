"""Tests for optimizer.py: closed-form vs brute force on small n."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import optimize_l1, optimize_l2, optimize_linf


def _random_psd(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return A.conj().T @ A


def test_l2_matches_top_eigenvalue():
    n = 6
    Q = _random_psd(n, seed=0)
    R = 1.5
    res = optimize_l2(Q, R=R)
    expected = R**2 * np.linalg.eigvalsh(0.5 * (Q + Q.conj().T)).max()
    assert np.isclose(res.J_star, expected, rtol=1e-9)
    # |a|_2 == R
    assert np.isclose(np.linalg.norm(res.a_star), R, atol=1e-9)


def test_linf_brute_force_n3():
    """For n=3, brute-force over a phase grid (with |a_i|=R) and check optimizer
    returns the correct sup."""
    n = 3
    Q = _random_psd(n, seed=1)
    R = 1.0
    grid = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    best = -np.inf
    for p1 in grid:
        for p2 in grid:
            for p3 in grid:
                a = R * np.array([np.exp(1j * p1), np.exp(1j * p2), np.exp(1j * p3)])
                v = float(np.real(np.vdot(a, Q @ a)))
                if v > best:
                    best = v
    res = optimize_linf(Q, R=R, num_restarts=30, max_iter=300, seed=42)
    # Allow small tolerance because brute grid is coarse
    assert res.J_star >= best - 1e-2
    # Check |a_i| = R within tolerance
    assert np.allclose(np.abs(res.a_star), R, atol=1e-6)


def test_l1_concentrates_at_vertex_n4():
    """For PSD Q, the L1-ball maximum of a*Qa is attained at a vertex
    R*e_k*e^{i theta} for some k. So J* = R^2 * max_k Q_kk."""
    n = 4
    Q = _random_psd(n, seed=2)
    R = 1.3
    res = optimize_l1(Q, R=R, num_restarts=30, max_iter=200, seed=7)
    expected = R**2 * np.real(np.diag(Q)).max()
    assert np.isclose(res.J_star, expected, rtol=1e-6, atol=1e-8)
    # sum |a_i| <= R
    assert np.sum(np.abs(res.a_star)) <= R + 1e-6
