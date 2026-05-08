"""Preset disagreement direction vectors for the directional gain objective.

Each preset returns a real-valued direction v in R^n. v is *not* normalized
here — `build_quadratic_form` and `functional_over_time` normalize internally.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


# All preset names exposed in the UI. The display order in the dropdown is
# determined by the per-network-type filter in app.py.
ALL_PRESETS = [
    "two_block",
    "three_block_first_vs_last",
    "three_block_outer_vs_inner",
    "alternating",
    "hub_vs_leaves",
    "checkerboard",
    "first_half_vs_second_half",
    "top_L_minus_eigvec",
    "consensus",
    "custom",
]


PRESET_DESCRIPTIONS = {
    "two_block":
        "Block-disagreement vector: +1 on every node of the first community, "
        "-1 on every node of the second. Captures the canonical structurally "
        "balanced bipartition. Falls back to first-half vs second-half when "
        "communities aren't labeled.",
    "three_block_first_vs_last":
        "First-vs-last block: +1 on community 1, 0 on community 2, -1 on "
        "community 3. Probes amplification along the first-vs-last axis while "
        "ignoring the middle block. Matches the spec example "
        "[1,…,1, 0,…,0, -1,…,-1].",
    "three_block_outer_vs_inner":
        "Outer-vs-inner: +1 on the two outer communities, -2 on the middle. "
        "Orthogonal to the consensus direction, so it cleanly isolates the "
        "outer-vs-middle disagreement mode.",
    "alternating":
        "Alternating sign by node index: [+1, -1, +1, -1, …]. Natural for "
        "cycles (highest-frequency Laplacian mode); for an odd-n cycle there "
        "is one wrap-around mismatch.",
    "hub_vs_leaves":
        "Hub-vs-leaves: +(n-1) on node 0 (the hub), -1 on every leaf. "
        "Orthogonal to the consensus direction. Designed for star and wheel "
        "networks.",
    "checkerboard":
        "Lattice checkerboard: v_{r,c} = (-1)^(row+col). The natural "
        "two-coloring on a grid; aligns with the bipartite mode of a 4-neighbor lattice.",
    "first_half_vs_second_half":
        "Network-agnostic block split: +1 on the first floor(n/2) nodes, -1 "
        "on the rest. Useful when no community labels are available.",
    "top_L_minus_eigvec":
        "Top eigenvector of L_- — the direction in opinion space that the "
        "antagonism graph most strongly resists. Often the most physically "
        "meaningful 'disagreement' direction for arbitrary signed graphs.",
    "consensus":
        "All-ones vector. Useful as a baseline; corresponds to the consensus "
        "direction in opinion space.",
    "custom":
        "User-specified vector. Enter a comma-separated list of n real numbers "
        "(brackets optional). The vector is normalized internally before use.",
}


def parse_custom_vector(text: str, n: int):
    """Parse text like '[1, -1, 0, 1]' or '1,-1,0,1' into a length-n float array.

    Returns (vec, error_msg). On success vec is np.ndarray of length n and
    error_msg is None. On failure vec is None and error_msg explains why.
    """
    if text is None or not str(text).strip():
        return None, "Enter a vector like  1, -1, 0, 1  (length must match n)."
    cleaned = str(text).strip()
    # Strip surrounding brackets / parens / whitespace
    for ch in "[](){}":
        cleaned = cleaned.replace(ch, " ")
    # Allow whitespace as separator too
    parts = [p.strip() for p in cleaned.replace(";", ",").replace("\t", ",").split(",")]
    parts = [p for p in parts if p]
    if len(parts) == 1 and " " in parts[0]:
        parts = parts[0].split()
    try:
        vals = [float(p) for p in parts]
    except ValueError as e:
        return None, f"Could not parse '{cleaned.strip()}' as numbers ({e})."
    if len(vals) != n:
        return None, f"Got {len(vals)} entries; need exactly n = {n}."
    arr = np.array(vals, dtype=float)
    if np.linalg.norm(arr) < 1e-12:
        return None, "Zero vector — direction is undefined."
    return arr, None


def make_direction_vector(
    preset: str,
    n: int,
    communities=None,
    A_minus=None,
    grid_shape=None,
) -> np.ndarray:
    """Construct a direction vector for the given preset.

    Falls back gracefully when a preset's required structure isn't present
    (e.g. asking for two_block on a graph with no community labels).
    """
    if preset == "consensus":
        return np.ones(n)

    if preset == "alternating":
        return np.array([(-1) ** i for i in range(n)], dtype=float)

    if preset == "first_half_vs_second_half":
        v = np.ones(n)
        v[n // 2:] = -1.0
        return v

    if preset == "two_block":
        if communities is not None and len(np.unique(communities)) >= 2:
            uniq = np.unique(communities)
            return np.where(communities == uniq[0], 1.0, -1.0).astype(float)
        v = np.ones(n)
        v[n // 2:] = -1.0
        return v

    if preset == "three_block_first_vs_last":
        v = np.zeros(n)
        if communities is not None and len(np.unique(communities)) >= 3:
            uniq = np.unique(communities)
            v[communities == uniq[0]] = 1.0
            v[communities == uniq[-1]] = -1.0
            return v
        third = max(n // 3, 1)
        v[:third] = 1.0
        v[-third:] = -1.0
        return v

    if preset == "three_block_outer_vs_inner":
        v = np.zeros(n)
        if communities is not None and len(np.unique(communities)) >= 3:
            uniq = np.unique(communities)
            v[communities == uniq[0]] = 1.0
            v[communities == uniq[-1]] = 1.0
            v[communities == uniq[1]] = -2.0
            return v
        third = max(n // 3, 1)
        v[:third] = 1.0
        v[-third:] = 1.0
        v[third:n - third] = -2.0
        return v

    if preset == "hub_vs_leaves":
        v = np.full(n, -1.0)
        v[0] = float(n - 1)
        return v

    if preset == "checkerboard":
        if grid_shape is not None:
            rows, cols = grid_shape
            v = np.zeros(n)
            for r in range(rows):
                for c in range(cols):
                    v[r * cols + c] = (-1.0) ** (r + c)
            return v
        # fallback: alternating
        return np.array([(-1) ** i for i in range(n)], dtype=float)

    if preset == "top_L_minus_eigvec":
        if A_minus is not None and np.asarray(A_minus).sum() > 0:
            d = np.asarray(A_minus).sum(axis=1)
            L_minus = np.diag(d) - np.asarray(A_minus)
            _, V = eigh(L_minus)
            v = V[:, -1]
            if v.sum() < 0:
                v = -v
            return v.astype(float)
        return np.ones(n)

    raise ValueError(f"unknown preset: {preset}")


# Default preset per network type and the filtered list of relevant presets
DEFAULT_PRESET = {
    "cycle": "alternating",
    "two_block": "two_block",
    "three_block": "three_block_first_vs_last",
    "star": "hub_vs_leaves",
    "erdos_renyi": "top_L_minus_eigvec",
    "lollipop": "first_half_vs_second_half",
    "wheel": "alternating",
    "lattice": "checkerboard",
}


def relevant_presets(network_type: str) -> list[str]:
    """Curate which presets appear in the dropdown for a given network type.
    'custom' is always appended so the user can enter a free-form vector."""
    if network_type == "two_block":
        out = ["two_block", "consensus", "first_half_vs_second_half", "top_L_minus_eigvec"]
    elif network_type == "three_block":
        out = [
            "three_block_first_vs_last",
            "three_block_outer_vs_inner",
            "consensus",
            "top_L_minus_eigvec",
        ]
    elif network_type == "star":
        out = ["hub_vs_leaves", "consensus", "top_L_minus_eigvec"]
    elif network_type == "lattice":
        out = ["checkerboard", "consensus", "first_half_vs_second_half", "top_L_minus_eigvec"]
    elif network_type == "wheel":
        out = ["alternating", "hub_vs_leaves", "consensus", "top_L_minus_eigvec"]
    elif network_type == "lollipop":
        out = ["first_half_vs_second_half", "consensus", "alternating", "top_L_minus_eigvec"]
    elif network_type == "cycle":
        out = ["alternating", "consensus", "first_half_vs_second_half", "top_L_minus_eigvec"]
    else:
        out = ["consensus", "alternating", "first_half_vs_second_half", "top_L_minus_eigvec"]
    out.append("custom")
    return out
