import numpy as np
import pytest

from vrtda import dynamics


def test_convergence_loads():
    conv = dynamics.convergence()
    assert len(conv.layers) == 65
    assert conv.layers[0] == 0 and conv.layers[-1] == 64
    assert np.all(conv.mean_pairwise > 0)
    assert np.all(conv.centroid_norm >= 0)


def test_convergence_summary_keys():
    s = dynamics.convergence_summary(dynamics.convergence())
    for k in ["n_layers", "peak_spread_layer", "final_centroid_norm",
              "final_mean_dist_to_centroid", "converged_layer", "spread_shrink"]:
        assert k in s
    assert s["n_layers"] == 65
    assert 0 <= s["converged_layer"] <= 64
    assert -1e-9 <= s["spread_shrink"] <= 1.0 + 1e-9


def test_per_language_final_token_distance_shape_and_convergence():
    layers, mat, prompts = dynamics.per_language_final_token_distance(layers=[0, 16, 64])
    assert mat.shape == (12, 3)
    assert len(prompts) == 12
    # the answer tokens converge at the final layer: mean distance to the centroid
    # is much smaller at layer 64 than at the mid-layer
    assert mat[:, 2].mean() < mat[:, 1].mean()
    assert mat[:, 2].mean() < 1000.0  # tightly clustered at the output


def test_flow_svd_rank1():
    comp, var, ls = dynamics.flow_svd(layers=list(range(0, 65, 8)))
    assert comp.shape[0] == 5 and comp.shape[1] == 5120
    assert len(var) == 5
    assert var.sum() > 0.99  # top-5 capture nearly all variance (rank-1 trajectory)
    assert var[0] > 0.8  # the attractor centroid moves along ~one direction


def test_attention_over_depth():
    layers, curve, peak = dynamics.attention_over_depth(metric="to_self")
    assert len(layers) == 64
    assert len(curve) == 64
    assert 0 <= peak < 64
    assert np.all(np.isfinite(curve))
