"""Métricas de reconstrucción y dispersión."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

EPS = 1e-12


def rel_frobenius(X: np.ndarray, Xh: np.ndarray) -> float:
    """ε_F = ||X − X̂||_F / ||X||_F (en %)."""
    return float(100 * np.linalg.norm(X - Xh) / (np.linalg.norm(X) + EPS))


def psnr(X: np.ndarray, Xh: np.ndarray, peak: float = 1.0) -> float:
    mse = np.mean((X - Xh) ** 2)
    return float(10 * np.log10(peak ** 2 / (mse + EPS)))


def ssim(img: np.ndarray, ref: np.ndarray, sigma: float = 1.5) -> float:
    """SSIM (Wang et al., 2004) con ventana gaussiana, promediado por canal.
    img, ref: (p, q, 3) en [0, 1]."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for c in range(img.shape[-1]):
        x, y = img[..., c], ref[..., c]
        mx, my = gaussian_filter(x, sigma), gaussian_filter(y, sigma)
        sxx = gaussian_filter(x * x, sigma) - mx ** 2
        syy = gaussian_filter(y * y, sigma) - my ** 2
        sxy = gaussian_filter(x * y, sigma) - mx * my
        m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
        vals.append(m.mean())
    return float(np.mean(vals))


def mean_ssim(imgs: np.ndarray, recs: np.ndarray) -> float:
    return float(np.mean([ssim(r, i) for i, r in zip(imgs, recs)]))


def jzr(H: np.ndarray, tol: float = 1e-8) -> float:
    """Joint Zero Rate: % de posiciones (átomo i, píxel j) donde los TRES
    canales son cero a la vez. H: (3, k, n)."""
    return float(100 * np.mean(H.max(axis=0) <= tol))


def channel_zero_rate(H: np.ndarray, tol: float = 1e-8) -> float:
    """% de entradas nulas contando cada canal por separado."""
    return float(100 * np.mean(H <= tol))


def hoyer(H: np.ndarray) -> float:
    """Dispersión de Hoyer media por átomo y canal (0 = denso, 1 = un solo no nulo)."""
    v = H.reshape(-1, H.shape[-1])
    n = v.shape[1]
    l1 = np.abs(v).sum(1)
    l2 = np.sqrt((v ** 2).sum(1)) + EPS
    return float(np.mean((np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1)))
