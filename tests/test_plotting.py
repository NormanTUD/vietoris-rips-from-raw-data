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
