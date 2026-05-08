"""Tests for plotting helpers (mainly the phase clustering)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plots import _cluster_phases_circular


def test_cluster_distinct_phases():
    phi = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    clusters = _cluster_phases_circular(phi, tol_rad=np.deg2rad(1.0))
    assert len(clusters) == 4
    counts = sorted(len(m) for _, m in clusters)
    assert counts == [1, 1, 1, 1]


def test_cluster_coincident_phases():
    phi = np.array([0.0, 0.0001, np.pi, np.pi + 0.0002, np.pi + 0.0004])
    clusters = _cluster_phases_circular(phi, tol_rad=np.deg2rad(1.0))
    counts = sorted(len(m) for _, m in clusters)
    assert counts == [2, 3]


def test_wrap_around_merge():
    # Phases at 359.5 deg and 0.5 deg should merge with 1 deg tolerance.
    phi = np.array([np.deg2rad(359.5), np.deg2rad(0.5)])
    clusters = _cluster_phases_circular(phi, tol_rad=np.deg2rad(2.0))
    assert len(clusters) == 1
    rep_deg = np.degrees(clusters[0][0])
    # Representative should be near 0 / 360
    assert rep_deg < 1.0 or rep_deg > 359.0


def test_phase_plot_excludes_zero_magnitude_nodes():
    """Nodes with r ~ 0 have undefined phase (np.angle(0) returns 0).
    plot_phase_histogram should exclude them so a vertex-l1 optimum
    (r = [1, 0, 0]) does not display two spurious vectors at 0 deg."""
    from plots import plot_phase_histogram
    phi = np.array([0.5, 0.0, 0.0])
    r = np.array([1.0, 0.0, 0.0])
    fig = plot_phase_histogram(phi, r_star=r, r_threshold=1e-6)
    # Each non-empty cluster contributes 2 traces (shaft + tip). With only
    # one active node, expect exactly 2 traces.
    assert len(fig.data) == 2
    # And the title should mention exclusion
    assert "excluded" in fig.layout.title.text


def test_representative_matches_input_when_unique():
    """The drawn vector angle must match the hover phase angle on the node."""
    rng = np.random.default_rng(0)
    phi = rng.uniform(0, 2 * np.pi, size=8)
    clusters = _cluster_phases_circular(phi, tol_rad=np.deg2rad(0.5))
    # With well-separated random phases, every cluster has exactly 1 member,
    # and the representative equals that member.
    assert len(clusters) == 8
    for rep, members in clusters:
        assert len(members) == 1
        assert np.isclose(rep, phi[members[0]] % (2 * np.pi), atol=1e-12)
