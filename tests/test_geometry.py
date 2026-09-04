from __future__ import annotations

import numpy as np
import pytest

from vrtda.beartype_guard import beartype_module
from vrtda.geometry import min_enclosing_ball_radius, min_enclosing_ball


def test_single_point() -> None:
    assert min_enclosing_ball_radius(np.array([[1.0, 2.0]])) == 0.0


def test_two_points() -> None:
    r = min_enclosing_ball_radius(np.array([[0.0, 0.0], [6.0, 0.0]]))
    assert r == pytest.approx(3.0)


def test_three_acute() -> None:
    # equilateral triangle side 2 -> circumradius 2/sqrt(3)
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, np.sqrt(3.0)]])
    r = min_enclosing_ball_radius(pts)
    assert r == pytest.approx(2.0 / np.sqrt(3.0), rel=1e-6)


def test_three_obtuse() -> None:
    # obtuse: MEB is half the longest edge
    pts = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 0.5]])
    r = min_enclosing_ball_radius(pts)
    # longest edge ~ distance (0,0)-(4,0)=4 -> but (4,0)-(1,0.5)=sqrt(9+0.25)=3.04; (0,0)-(4,0)=4
    assert r == pytest.approx(2.0, rel=1e-6)


def test_four_in_3d() -> None:
    # regular tetrahedron edge 2 -> circumradius sqrt(6)/4 * 2 = sqrt(6)/2
    pts = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=float)
    r = min_enclosing_ball_radius(pts)
    edge = np.linalg.norm(pts[0] - pts[1])
    assert r == pytest.approx(edge * np.sqrt(6) / 4.0, rel=1e-6)


def test_meb_contains_all() -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(8, 3))
    c, r = min_enclosing_ball(pts)
    assert np.all(np.linalg.norm(pts - c, axis=1) <= r + 1e-9)


def test_meb_radius_matches_center() -> None:
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(6, 4))
    c, r = min_enclosing_ball(pts)
    assert r == pytest.approx(min_enclosing_ball_radius(pts), rel=1e-9)


beartype_module(__name__)
