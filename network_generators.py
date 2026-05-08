"""Signed network templates.

Each generator returns (A_plus, A_minus, communities, positions), where:
  A_plus, A_minus : (n, n) symmetric, zero diagonal, nonnegative
  communities     : (n,) int array of community labels, or None
  positions       : (n, 2) float array of layout coords, or None
"""

from __future__ import annotations

import numpy as np


def _symmetrize_zero_diag(A: np.ndarray) -> np.ndarray:
    A = np.triu(A, 1)
    A = A + A.T
    return A


def generate_cycle_signed(n: int, sign_pattern: str = "all_pos"):
    A_plus = np.zeros((n, n))
    A_minus = np.zeros((n, n))
    edges = [(i, (i + 1) % n) for i in range(n)]
    if sign_pattern == "all_pos":
        signs = [+1] * n
    elif sign_pattern == "alternating":
        # If n is odd, alternation is imperfect; we fall back to (n-1) alternations
        # plus a single sign for the closing edge.
        signs = [(+1) if k % 2 == 0 else -1 for k in range(n)]
    elif sign_pattern == "one_neg":
        signs = [+1] * n
        signs[0] = -1
    else:
        raise ValueError(f"unknown sign_pattern: {sign_pattern}")

    for (i, j), s in zip(edges, signs):
        if s > 0:
            A_plus[i, j] = A_plus[j, i] = 1
        else:
            A_minus[i, j] = A_minus[j, i] = 1

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    communities = None
    return A_plus, A_minus, communities, positions


def _block_positions(community_sizes, radius=2.5, jitter=0.6, seed=0):
    rng = np.random.default_rng(seed)
    K = len(community_sizes)
    n = sum(community_sizes)
    centers = np.array(
        [[radius * np.cos(2 * np.pi * k / K), radius * np.sin(2 * np.pi * k / K)] for k in range(K)]
    )
    pos = np.zeros((n, 2))
    idx = 0
    for k, sz in enumerate(community_sizes):
        for _ in range(sz):
            pos[idx] = centers[k] + jitter * rng.standard_normal(2)
            idx += 1
    return pos


def _block_signed_graph(community_sizes, p_in, p_out, seed=None):
    rng = np.random.default_rng(seed)
    n = sum(community_sizes)
    labels = np.concatenate([np.full(sz, k) for k, sz in enumerate(community_sizes)])
    A_plus = np.zeros((n, n))
    A_minus = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            same = labels[i] == labels[j]
            p = p_in if same else p_out
            if rng.random() < p:
                if same:
                    A_plus[i, j] = A_plus[j, i] = 1
                else:
                    A_minus[i, j] = A_minus[j, i] = 1
    return A_plus, A_minus, labels


def generate_two_block_balanced(n1: int, n2: int, p_in: float = 1.0, p_out: float = 1.0, seed=None):
    A_plus, A_minus, labels = _block_signed_graph([n1, n2], p_in, p_out, seed)
    positions = _block_positions([n1, n2], seed=seed if seed is not None else 0)
    return A_plus, A_minus, labels, positions


def generate_three_block_signed(
    n1: int, n2: int, n3: int, p_in: float = 1.0, p_out: float = 1.0, seed=None
):
    A_plus, A_minus, labels = _block_signed_graph([n1, n2, n3], p_in, p_out, seed)
    positions = _block_positions([n1, n2, n3], seed=seed if seed is not None else 0)
    return A_plus, A_minus, labels, positions


def generate_signed_star(
    n_leaves: int,
    leaf_signs: str = "positive",
    seed=None,
):
    """Hub at index 0, leaves at indices 1..n_leaves."""
    rng = np.random.default_rng(seed)
    n = n_leaves + 1
    A_plus = np.zeros((n, n))
    A_minus = np.zeros((n, n))
    for j in range(1, n):
        if leaf_signs == "positive":
            s = +1
        elif leaf_signs == "negative":
            s = -1
        elif leaf_signs == "mixed":
            s = +1 if rng.random() < 0.5 else -1
        else:
            raise ValueError(f"unknown leaf_signs: {leaf_signs}")
        if s > 0:
            A_plus[0, j] = A_plus[j, 0] = 1
        else:
            A_minus[0, j] = A_minus[j, 0] = 1

    positions = np.zeros((n, 2))
    angles = np.linspace(0, 2 * np.pi, n_leaves, endpoint=False)
    positions[1:] = np.column_stack([np.cos(angles), np.sin(angles)]) * 2.0
    return A_plus, A_minus, None, positions


def generate_signed_er(n: int, p_edge: float, q_neg: float, seed=None):
    rng = np.random.default_rng(seed)
    A_plus = np.zeros((n, n))
    A_minus = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge:
                if rng.random() < q_neg:
                    A_minus[i, j] = A_minus[j, i] = 1
                else:
                    A_plus[i, j] = A_plus[j, i] = 1
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 2.0
    positions = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    # Light spring relaxation if networkx is available, else cycle layout.
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if A_plus[i, j] or A_minus[i, j]:
                    G.add_edge(i, j)
        if G.number_of_edges() > 0:
            pos = nx.spring_layout(G, seed=seed if seed is not None else 0, k=1.5)
            positions = np.array([pos[i] for i in range(n)])
    except Exception:
        pass
    return A_plus, A_minus, None, positions


def generate_signed_lollipop(
    n_clique: int,
    n_path: int,
    sign_clique: str = "positive",
    sign_path: str = "positive",
    sign_join: str = "positive",
    seed=None,
):
    """K_{n_clique} attached to a path of n_path nodes via node n_clique-1."""
    n = n_clique + n_path
    A_plus = np.zeros((n, n))
    A_minus = np.zeros((n, n))

    def add(i, j, sign):
        target = A_plus if sign == "positive" else A_minus
        target[i, j] = target[j, i] = 1

    for i in range(n_clique):
        for j in range(i + 1, n_clique):
            add(i, j, sign_clique)
    if n_path >= 1:
        add(n_clique - 1, n_clique, sign_join)
        for k in range(n_path - 1):
            add(n_clique + k, n_clique + k + 1, sign_path)

    positions = np.zeros((n, 2))
    for i in range(n_clique):
        ang = 2 * np.pi * i / max(n_clique, 1)
        positions[i] = [np.cos(ang), np.sin(ang)]
    for k in range(n_path):
        positions[n_clique + k] = [2.0 + 0.7 * (k + 1), 0.0]
    return A_plus, A_minus, None, positions


def generate_signed_wheel(
    n_outer: int,
    sign_hub: str = "positive",
    sign_rim: str = "positive",
    seed=None,
):
    """Wheel: hub at index 0, cycle on outer nodes 1..n_outer."""
    n = n_outer + 1
    A_plus = np.zeros((n, n))
    A_minus = np.zeros((n, n))

    def add(i, j, sign):
        target = A_plus if sign == "positive" else A_minus
        target[i, j] = target[j, i] = 1

    for k in range(n_outer):
        i = k + 1
        add(0, i, sign_hub)
        j = ((k + 1) % n_outer) + 1
        add(i, j, sign_rim)

    positions = np.zeros((n, 2))
    angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    positions[1:] = np.column_stack([np.cos(angles), np.sin(angles)]) * 1.6
    return A_plus, A_minus, None, positions


def generate_signed_lattice(
    rows: int,
    cols: int,
    sign: str = "positive",
    sign_pattern: str = "uniform",
    seed=None,
):
    """rows x cols 4-neighbor lattice.

    sign_pattern:
      uniform     — all edges have the same sign as `sign`.
      checkerboard— horizontal edges positive, vertical edges negative.
    """
    n = rows * cols
    A_plus = np.zeros((n, n))
    A_minus = np.zeros((n, n))

    def add(i, j, s):
        target = A_plus if s == "positive" else A_minus
        target[i, j] = target[j, i] = 1

    def idx(r, c):
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            if c < cols - 1:
                if sign_pattern == "uniform":
                    add(idx(r, c), idx(r, c + 1), sign)
                else:
                    add(idx(r, c), idx(r, c + 1), "positive")
            if r < rows - 1:
                if sign_pattern == "uniform":
                    add(idx(r, c), idx(r + 1, c), sign)
                else:
                    add(idx(r, c), idx(r + 1, c), "negative")

    positions = np.zeros((n, 2))
    for r in range(rows):
        for c in range(cols):
            positions[idx(r, c)] = [c - (cols - 1) / 2.0, -(r - (rows - 1) / 2.0)]
    return A_plus, A_minus, None, positions


GENERATORS = {
    "cycle": generate_cycle_signed,
    "two_block": generate_two_block_balanced,
    "three_block": generate_three_block_signed,
    "star": generate_signed_star,
    "erdos_renyi": generate_signed_er,
    "lollipop": generate_signed_lollipop,
    "wheel": generate_signed_wheel,
    "lattice": generate_signed_lattice,
}
