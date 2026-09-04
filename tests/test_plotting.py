import importlib.util

import numpy as np
import pytest

from vrtda import plotting
from vrtda.errors import DataError

HAS_MPL = importlib.util.find_spec("matplotlib") is not None


@pytest.mark.skipif(HAS_MPL, reason="matplotlib present; testing the happy path needs display-less backend")
def test_plotting_without_matplotlib_raises():
    with pytest.raises(DataError):
        plotting.plot_point_cloud_2d(np.ones((3, 2)), "/tmp/_vrtda_test.png")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plotting_smoke(tmp_path):
    pts = np.random.default_rng(0).normal(size=(20, 2))
    plotting.plot_point_cloud_2d(pts, tmp_path / "pc.png")
    assert (tmp_path / "pc.png").exists()
    plotting.plot_betti_function([0, 1, 2], np.array([[3, 0], [1, 1], [1, 1]]), tmp_path / "betti.png")
    assert (tmp_path / "betti.png").exists()


def test_plot_point_cloud_rejects_1d():
    with pytest.raises((DataError, Exception)):
        plotting.plot_point_cloud_2d(np.ones((5, 1)), "/tmp/_vrtda_test.png")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plotting_new_methods_smoke(tmp_path):
    from vrtda.mapper import MapperGraph, MapperNode
    from vrtda.dynamics import Convergence

    # persistence diagram / landscape / image
    plotting.plot_persistence_diagram(np.array([[0.1, 0.4], [0.2, 0.9]]), tmp_path / "diag.png")
    xg = np.linspace(0, 2, 50)
    plotting.plot_persistence_landscape(xg, np.stack([np.abs(xg - 1), np.zeros_like(xg)]), tmp_path / "land.png")
    plotting.plot_persistence_image(np.zeros((32, 32)), tmp_path / "img.png")

    # depth heatmap + betti over depth
    H = np.zeros((4, 3)); H[2, 1] = 3.0
    plotting.plot_depth_heatmap(H, np.linspace(0.5, 2.0, 4), [0, 8, 16], tmp_path / "heat.png")
    profile = {0: {"total_persistence": 10.0, "beta_peak": 2}, 8: {"total_persistence": 20.0, "beta_peak": 5}}
    plotting.plot_betti_over_depth(profile, tmp_path / "bedepth.png")

    # mapper graph
    nodes = [MapperNode((0, 1), 4, 1, 1, [0, 1, 2, 3]), MapperNode((1, 2), 4, 1, 1, [4, 5, 6, 7])]
    g = MapperGraph(nodes, [(0, 1, 2)])
    plotting.plot_mapper(g, tmp_path / "mapper.png")

    # attractor overlay
    pts = np.random.default_rng(0).normal(size=(12, 2))
    plotting.plot_attractor_overlay(pts, [[0, 1, 2, 3]], tmp_path / "overlay.png")

    # convergence
    L = list(range(0, 65))
    conv = Convergence(L, np.linspace(2, 200, 65), np.linspace(0.5, 110, 65),
                       np.linspace(1, 137, 65), np.linspace(1, 200, 65))
    plotting.plot_convergence(conv, tmp_path / "conv.png")

    for name in ["diag", "land", "img", "heat", "bedepth", "mapper", "overlay", "conv"]:
        assert (tmp_path / f"{name}.png").exists()
