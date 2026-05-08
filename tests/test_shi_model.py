"""Tests for shi_model.py."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shi_model import (
    build_shi_repelling_matrix,
    compute_laplacians,
    compute_threshold,
)


def test_W_row_sums():
    n = 5
    A_plus = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=float)
    A_minus = np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
    ], dtype=float)
    alpha = 0.1
    beta = 0.05
    c = np.array([0.05, 0.1, 0.05, 0.1, 0.05])
    W = build_shi_repelling_matrix(A_plus, A_minus, alpha, beta, c)
    # Row sums of W = 1 - c_i  (because L_+ 1 = 0 and L_- 1 = 0)
    row_sums = W.sum(axis=1)
    expected = 1.0 - c
    assert np.allclose(row_sums, expected)


def test_W_diagonal():
    n = 4
    A_plus = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=float)
    A_minus = np.zeros((4, 4))
    alpha = 0.2
    beta = 0.1
    c = np.array([0.05, 0.05, 0.05, 0.05])
    W = build_shi_repelling_matrix(A_plus, A_minus, alpha, beta, c)
    d_plus = A_plus.sum(axis=1)
    d_minus = A_minus.sum(axis=1)
    expected_diag = 1 - alpha * d_plus + beta * d_minus - c
    assert np.allclose(np.diag(W), expected_diag)


def test_threshold_no_negative_edges_is_inf():
    n = 4
    A_plus = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=float)
    A_minus = np.zeros((4, 4))
    L_plus, L_minus = compute_laplacians(A_plus, A_minus)
    D = np.diag([0.1, 0.1, 0.1, 0.1])
    thresh = compute_threshold(0.2, L_plus, L_minus, D)
    assert thresh == float("inf")
