"""Tests for direction-vector handling, especially parse_custom_vector."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from directions import (
    PRESET_DESCRIPTIONS,
    make_direction_vector,
    parse_custom_vector,
    relevant_presets,
)


def test_parse_brackets_and_commas():
    v, err = parse_custom_vector("[1, -1, 0, 2.5]", 4)
    assert err is None
    assert np.allclose(v, [1.0, -1.0, 0.0, 2.5])


def test_parse_no_brackets():
    v, err = parse_custom_vector("1,1,-1", 3)
    assert err is None
    assert np.allclose(v, [1.0, 1.0, -1.0])


def test_parse_whitespace_separated():
    v, err = parse_custom_vector("1 2 -3 4.5", 4)
    assert err is None
    assert np.allclose(v, [1.0, 2.0, -3.0, 4.5])


def test_parse_semicolon_separated():
    v, err = parse_custom_vector("(1; 2; 3)", 3)
    assert err is None
    assert np.allclose(v, [1.0, 2.0, 3.0])


def test_parse_wrong_length_returns_error():
    v, err = parse_custom_vector("1, 2, 3", 4)
    assert v is None
    assert err is not None
    assert "4" in err


def test_parse_zero_vector_rejected():
    v, err = parse_custom_vector("0, 0, 0", 3)
    assert v is None
    assert "Zero" in err or "undefined" in err


def test_parse_empty_returns_error():
    v, err = parse_custom_vector("", 3)
    assert v is None
    assert err is not None


def test_parse_invalid_string_returns_error():
    v, err = parse_custom_vector("a, b, c", 3)
    assert v is None
    assert err is not None


def test_custom_in_relevant_presets_for_every_network_type():
    for nt in [
        "cycle", "two_block", "three_block", "star", "erdos_renyi",
        "lollipop", "wheel", "lattice", "unknown_type",
    ]:
        presets = relevant_presets(nt)
        assert "custom" in presets, f"'custom' missing for {nt}"


def test_every_preset_has_a_description():
    for p in relevant_presets("two_block") + relevant_presets("three_block"):
        assert p in PRESET_DESCRIPTIONS
        assert len(PRESET_DESCRIPTIONS[p]) > 10  # something substantial


def test_make_direction_vector_handles_all_named_presets():
    """All presets except 'custom' should be constructable from name alone."""
    n = 9
    A_minus = np.array(
        [[0, 1, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=float,
    )
    communities = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    grid_shape = (3, 3)
    for p, desc in PRESET_DESCRIPTIONS.items():
        if p == "custom":
            continue
        v = make_direction_vector(
            p, n,
            communities=communities,
            A_minus=A_minus,
            grid_shape=grid_shape,
        )
        assert v.shape == (n,)
        assert np.linalg.norm(v) > 0, f"preset {p} returned zero vector"
