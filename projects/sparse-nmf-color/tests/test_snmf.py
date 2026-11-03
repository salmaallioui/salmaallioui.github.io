import numpy as np
import pytest

from snmf import metrics as M
from snmf.data import from_channels, make_images, to_channels
from snmf.models import fit, greedy, init_factors, penalty_grad


@pytest.fixture(scope="module")
def X():
    return to_channels(make_images(n_images=8, size=16, seed=1))


def test_channel_roundtrip():
    imgs = make_images(n_images=3, size=8)
    assert np.allclose(from_channels(to_channels(imgs), 8, 8), imgs)


def test_jzr_counts_only_joint_zeros():
    H = np.ones((3, 1, 4))
    H[:, 0, 0] = 0          # cero conjunto
    H[0, 0, 1] = 0          # cero parcial: no cuenta
    assert M.jzr(H) == 25.0
    assert M.channel_zero_rate(H) == pytest.approx(100 * 4 / 12)


def test_joint_penalty_gradient_sums_to_lambda_per_group():
    H = np.random.default_rng(0).random((3, 2, 5))
    g = penalty_grad(H, "joint_linf", 0.3)
    assert np.allclose(g.sum(0), 0.3)


def test_mu_decreases_error(X):
    F0 = init_factors(X, 4, 0)
    e0 = M.rel_frobenius(X, F0.reconstruct())
    e1 = M.rel_frobenius(X, fit(X, F0, iters=200).reconstruct())
    assert e1 < e0


def test_joint_penalty_gives_more_joint_zeros_than_l1(X):
    F0 = init_factors(X, 4, 0)
    l1 = fit(X, F0, "l1", 0.05, 400)
    jl = greedy(X, fit(X, F0, "joint_linf", 0.02, 400), alpha=0.05)
    share = lambda F: M.jzr(F.H) / max(M.channel_zero_rate(F.H), 1e-9)
    assert share(jl) > share(l1)


def test_ssim_identity():
    img = make_images(n_images=1, size=16)[0]
    assert M.ssim(img, img) == pytest.approx(1.0)
