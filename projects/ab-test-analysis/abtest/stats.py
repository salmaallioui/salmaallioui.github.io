"""Herramientas estadísticas para tests A/B (solo NumPy + SciPy)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- diseño
def sample_size_two_proportions(p1: float, p2: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """n por grupo para detectar p1 -> p2 con un z-test bilateral de dos proporciones."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return int(np.ceil(num ** 2 / (p2 - p1) ** 2))


def power_two_proportions(p1: float, p2: float, n: int, alpha: float = 0.05) -> float:
    za = stats.norm.ppf(1 - alpha / 2)
    pbar = (p1 + p2) / 2
    se0 = np.sqrt(2 * pbar * (1 - pbar) / n)
    se1 = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    d = abs(p2 - p1)
    return float(stats.norm.cdf((d - za * se0) / se1) + stats.norm.cdf((-d - za * se0) / se1))


# --------------------------------------------------------------------------- validez
def srm_test(n_a: int, n_b: int, expected_share_a: float = 0.5) -> float:
    """Sample Ratio Mismatch: p-valor chi-cuadrado de la asignación observada."""
    total = n_a + n_b
    exp = [total * expected_share_a, total * (1 - expected_share_a)]
    return float(stats.chisquare([n_a, n_b], exp).pvalue)


# --------------------------------------------------------------------------- frecuentista
@dataclass
class PropResult:
    p_a: float
    p_b: float
    diff: float
    ci_low: float
    ci_high: float
    rel_lift: float
    z: float
    p_value: float


def two_proportion_ztest(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05) -> PropResult:
    pa, pb = x_a / n_a, x_b / n_b
    pp = (x_a + x_b) / (n_a + n_b)
    se_pooled = np.sqrt(pp * (1 - pp) * (1 / n_a + 1 / n_b))
    z = (pb - pa) / se_pooled
    p = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)  # IC de Wald no agrupado
    zc = stats.norm.ppf(1 - alpha / 2)
    return PropResult(pa, pb, pb - pa, pb - pa - zc * se, pb - pa + zc * se, pb / pa - 1, float(z), float(p))


def welch_ttest(a: np.ndarray, b: np.ndarray, alpha: float = 0.05):
    t = stats.ttest_ind(b, a, equal_var=False)
    ci = t.confidence_interval(1 - alpha)
    return {"diff": float(b.mean() - a.mean()), "ci_low": float(ci.low), "ci_high": float(ci.high),
            "t": float(t.statistic), "p_value": float(t.pvalue)}


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    """IC bootstrap percentil de la diferencia de medias (no asume normalidad)."""
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    diffs = b[ib].mean(1) - a[ia].mean(1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), diffs


def holm(pvalues: list[float]) -> list[float]:
    """Corrección de Holm-Bonferroni (controla FWER). Devuelve p-valores ajustados."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


# --------------------------------------------------------------------------- bayesiano
def beta_binomial(x_a: int, n_a: int, x_b: int, n_b: int, prior=(1, 1), draws: int = 200_000, seed: int = 0):
    """Posterior Beta para cada brazo. Devuelve P(B > A), pérdida esperada e IC creíble del lift."""
    rng = np.random.default_rng(seed)
    a = rng.beta(prior[0] + x_a, prior[1] + n_a - x_a, draws)
    b = rng.beta(prior[0] + x_b, prior[1] + n_b - x_b, draws)
    lift = b / a - 1
    return {
        "prob_b_better": float((b > a).mean()),
        "expected_loss_choose_b": float(np.maximum(a - b, 0).mean()),
        "expected_loss_choose_a": float(np.maximum(b - a, 0).mean()),
        "lift_ci": [float(x) for x in np.quantile(lift, [0.025, 0.975])],
        "samples_a": a, "samples_b": b,
    }


# --------------------------------------------------------------------------- peeking
def peeking_false_positive_rate(p: float = 0.08, n_per_arm: int = 12_000, looks: int = 20,
                                n_sims: int = 2000, alpha: float = 0.05, seed: int = 1):
    """Simula tests A/A (sin efecto real) y mide la tasa de falsos positivos si
    se para en cuanto p < alpha en cualquiera de `looks` miradas intermedias."""
    rng = np.random.default_rng(seed)
    checkpoints = np.linspace(n_per_arm / looks, n_per_arm, looks).astype(int)
    ca = rng.binomial(1, p, (n_sims, n_per_arm)).cumsum(1)[:, checkpoints - 1]
    cb = rng.binomial(1, p, (n_sims, n_per_arm)).cumsum(1)[:, checkpoints - 1]
    n = checkpoints[None, :]
    pa, pb = ca / n, cb / n
    pp = (ca + cb) / (2 * n)
    z = (pb - pa) / np.sqrt(pp * (1 - pp) * 2 / n)
    sig = 2 * stats.norm.sf(np.abs(z)) < alpha
    return {"fixed_horizon": float(sig[:, -1].mean()), "with_peeking": float(sig.any(1).mean()),
            "cumulative": sig.cumsum(1).astype(bool).mean(0).tolist(), "checkpoints": checkpoints.tolist()}
