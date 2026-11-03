import numpy as np
import pytest

from abtest import stats as st
from abtest.simulate import simulate


def test_sample_size_matches_textbook_formula():
    # 8 % -> 9 %, alpha 5 %, potencia 80 %: ~12.2k por grupo
    assert st.sample_size_two_proportions(0.08, 0.09) == 12208


def test_power_at_required_n_is_80pct():
    n = st.sample_size_two_proportions(0.08, 0.09)
    assert st.power_two_proportions(0.08, 0.09, n) == pytest.approx(0.80, abs=0.005)


def test_ztest_symmetric_and_ci_contains_diff():
    r = st.two_proportion_ztest(100, 1000, 130, 1000)
    r2 = st.two_proportion_ztest(130, 1000, 100, 1000)
    assert r.p_value == pytest.approx(r2.p_value)
    assert r.ci_low < r.diff < r.ci_high


def test_srm_flags_skewed_split():
    assert st.srm_test(5000, 5000) > 0.9
    assert st.srm_test(5000, 5400) < 0.001


def test_holm_known_values():
    assert st.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_bayes_probability_is_half_with_identical_data():
    assert st.beta_binomial(80, 1000, 80, 1000)["prob_b_better"] == pytest.approx(0.5, abs=0.01)


def test_peeking_inflates_false_positives():
    r = st.peeking_false_positive_rate(n_per_arm=4000, looks=10, n_sims=1500)
    assert r["fixed_horizon"] == pytest.approx(0.05, abs=0.02)
    assert r["with_peeking"] > 0.12


def test_bootstrap_ci_covers_true_difference():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 2, 2000), rng.normal(10.5, 2, 2000)
    lo, hi, _ = st.bootstrap_diff_ci(a, b, n_boot=2000)
    assert lo < 0.5 < hi


def test_simulation_is_reproducible():
    assert simulate(1000, seed=1).equals(simulate(1000, seed=1))
