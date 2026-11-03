"""NMF por canal con penalizaciones de dispersión y búsqueda del soporte.

Modelo (un par de factores por canal c ∈ {R, G, B}):

    X_c ≈ W_c H_c,   X_c ∈ R^{d×n}_+,  W_c ∈ R^{d×k}_+,  H_c ∈ R^{k×n}_+

    min  Σ_c ½‖X_c − W_c H_c‖²_F  +  λ · Ω(H)

  * NMF estándar:      Ω = 0
  * ℓ1 separable:      Ω = Σ_c Σ_ij H_c[i,j]               (ceros canal a canal)
  * ℓ∞ conjunta:       Ω = Σ_ij max(H_R[i,j], H_G[i,j], H_B[i,j])

La ℓ∞ conjunta trata el trío (H_R, H_G, H_B) de cada posición como un grupo:
solo reduce la penalización si baja el canal MÁXIMO, así que empuja a que los
tres canales se anulen a la vez. Esto se mide con la JZR (Joint Zero Rate).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import jzr, rel_frobenius

EPS = 1e-10


@dataclass
class Factors:
    W: np.ndarray            # (3, d, k)
    H: np.ndarray            # (3, k, n)
    mask: np.ndarray         # (k, n) bool: grupos (átomo, píxel) activos
    history: list = field(default_factory=list)

    def reconstruct(self) -> np.ndarray:
        return np.einsum("cdk,ckn->cdn", self.W, self.H)

    def copy(self) -> "Factors":
        return Factors(self.W.copy(), self.H.copy(), self.mask.copy(), list(self.history))


def init_factors(X: np.ndarray, k: int, seed: int = 0) -> Factors:
    rng = np.random.default_rng(seed)
    C, d, n = X.shape
    scale = np.sqrt(X.mean() / k)
    W = rng.uniform(0.1, 1.0, (C, d, k)) * scale
    H = rng.uniform(0.1, 1.0, (C, k, n)) * scale
    return Factors(W, H, np.ones((k, n), bool))


def _normalize(F: Factors) -> None:
    """Columnas de W_c con norma 1 (y H_c reescalada): elimina la ambigüedad de
    escala, sin la cual la penalización sobre H se 'esquivaría' agrandando W."""
    norms = np.linalg.norm(F.W, axis=1, keepdims=True) + EPS      # (3, 1, k)
    F.W /= norms
    F.H *= norms.transpose(0, 2, 1)


def penalty_grad(H: np.ndarray, kind: str, lam: float) -> np.ndarray:
    if kind == "none" or lam == 0:
        return np.zeros_like(H)
    if kind == "l1":
        return np.full_like(H, lam)
    if kind == "joint_linf":
        # Subgradiente de max_c H_c repartido proporcionalmente entre canales
        # (suma λ en cada grupo). La versión "solo el canal dominante" oscila
        # cuando dos canales están casi empatados; el reparto proporcional es
        # estable y sigue empujando al grupo entero hacia cero.
        return lam * H / (H.sum(axis=0, keepdims=True) + EPS)
    raise ValueError(kind)


def fit(X: np.ndarray, F: Factors, kind: str = "none", lam: float = 0.0, iters: int = 300,
        track_every: int = 0) -> Factors:
    """Actualizaciones multiplicativas (Lee & Seung) con penalización sobre H.
    Las entradas a cero permanecen a cero, así que `mask` define el soporte."""
    F = F.copy()
    m = F.mask[None].astype(float)
    F.H *= m
    for t in range(iters):
        WT = F.W.transpose(0, 2, 1)
        num = WT @ X
        den = WT @ F.W @ F.H + penalty_grad(F.H, kind, lam) + EPS
        F.H *= num / den
        F.H *= m
        num = X @ F.H.transpose(0, 2, 1)
        den = F.W @ F.H @ F.H.transpose(0, 2, 1) + EPS
        F.W *= num / den
        _normalize(F)
        if track_every and t % track_every == 0:
            F.history.append(rel_frobenius(X, F.reconstruct()))
    F.H[F.H < 1e-9] = 0.0
    return F


def objective(X: np.ndarray, F: Factors, alpha: float) -> float:
    """Criterio de compromiso usado por Greedy/VNS: error (%) + α·(% de grupos activos)."""
    return rel_frobenius(X, F.reconstruct()) + alpha * (100 - jzr(F.H))


# --------------------------------------------------------------------------- Greedy
def greedy(X: np.ndarray, F0: Factors, alpha: float = 0.1, step: float = 0.03,
           refit_iters: int = 40, max_rounds: int = 200) -> Factors:
    """Conjunto activo voraz: en cada ronda anula los grupos (átomo, píxel) cuyo
    máximo entre canales es menor (es decir, max(H_R, H_G, H_B) < τ con τ
    creciente), los TRES canales a la vez, y reajusta sobre el soporte restante.
    Se detiene cuando anular más empeora el criterio."""
    best = F0.copy()
    best.mask &= best.H.max(axis=0) > 0
    f_best = objective(X, best, alpha)
    for _ in range(max_rounds):
        score = np.where(best.mask, best.H.max(axis=0), np.inf)
        n_active = int(best.mask.sum())
        n_drop = max(1, int(step * n_active))
        if n_active <= n_drop:
            break
        drop = np.argsort(score, axis=None)[:n_drop]
        cand = best.copy()
        cand.mask.flat[drop] = False
        cand = fit(X, cand, iters=refit_iters)
        f = objective(X, cand, alpha)
        if f < f_best:
            best, f_best = cand, f
        else:
            break
    best.history.append(("greedy", f_best))
    return best


# --------------------------------------------------------------------------- VNS
def vns(X: np.ndarray, F0: Factors, alpha: float = 0.05, lam: float = 0.02, k_max: int = 4,
        base_step: float = 0.15, max_iter: int = 30, local_iters: int = 200, seed: int = 0) -> Factors:
    """Variable Neighbourhood Search sobre (W, H, soporte).

    La NMF es no convexa: Greedy converge al óptimo local más cercano a su punto
    de partida. VNS lo sacude con vecindarios de tamaño creciente:

      N_κ:  W_c ← W_c · exp(κ·s·ξ), ξ ~ N(0, 1)   y se reabre el soporte
            con un pequeño valor de fondo en H (los átomos pueden "moverse").

    Búsqueda local = ajuste con la penalización ℓ∞ conjunta + Greedy.
    Si el criterio mejora se acepta y κ vuelve a 1; si no, κ ← κ + 1 (cíclico)."""
    rng = np.random.default_rng(seed)
    best = F0.copy()
    f_best = objective(X, best, alpha)
    trace, kappa = [f_best], 1
    for _ in range(max_iter):
        s = base_step * kappa
        cand = best.copy()
        cand.W *= np.exp(s * rng.standard_normal(cand.W.shape))
        cand.mask[:] = True
        cand.H = cand.H + 0.05 * s * cand.H.max() * rng.random(cand.H.shape)
        cand = fit(X, cand, "joint_linf", lam, local_iters)
        cand = greedy(X, cand, alpha=alpha)
        f = objective(X, cand, alpha)
        if f < f_best - 1e-6:
            best, f_best, kappa = cand, f, 1
        else:
            kappa = kappa + 1 if kappa < k_max else 1
        trace.append(f_best)
    best.history.append(("vns_trace", trace))
    return best
