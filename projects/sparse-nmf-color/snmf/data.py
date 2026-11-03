"""Dataset sintético de imágenes en color.

Cada imagen combina "zonas" de color (cuadrantes y bandas) con transiciones
gaussianas suaves. Las zonas son los átomos espaciales verdaderos: localizados
y compartidos por los tres canales RGB, que es justo la estructura que la
NMF dispersa conjunta debe recuperar.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

PALETTE = np.array([
    [0.85, 0.20, 0.25],  # rojo
    [0.20, 0.55, 0.85],  # azul
    [0.95, 0.75, 0.20],  # amarillo
    [0.30, 0.70, 0.40],  # verde
    [0.60, 0.35, 0.75],  # violeta
    [0.95, 0.55, 0.25],  # naranja
])


def zone_masks(size: int = 32, sigma: float = 1.5) -> np.ndarray:
    """6 zonas espaciales suavizadas: 4 cuadrantes + banda central + disco."""
    h = size // 2
    yy, xx = np.mgrid[:size, :size]
    raw = np.zeros((6, size, size))
    raw[0, :h, :h] = 1
    raw[1, :h, h:] = 1
    raw[2, h:, :h] = 1
    raw[3, h:, h:] = 1
    raw[4, size // 2 - 3: size // 2 + 3, :] = 1
    raw[5][(yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size / 5) ** 2] = 1
    return np.stack([gaussian_filter(m, sigma) for m in raw])


def make_images(n_images: int = 12, size: int = 32, noise: float = 0.01, seed: int = 0) -> np.ndarray:
    """Devuelve un array (n_images, size, size, 3) con valores en [0, 1]."""
    rng = np.random.default_rng(seed)
    zones = zone_masks(size)
    imgs = np.zeros((n_images, size, size, 3))
    for t in range(n_images):
        active = rng.choice(6, size=rng.integers(2, 5), replace=False)
        for z in active:
            color = PALETTE[rng.integers(len(PALETTE))] * rng.uniform(0.35, 0.55)
            imgs[t] += zones[z][..., None] * color
    imgs += noise * rng.standard_normal(imgs.shape)
    return np.clip(imgs, 0, 1)


def to_channels(imgs: np.ndarray) -> np.ndarray:
    """(d, p, q, 3) -> (3, d, n) con n = p*q: una matriz X_c por canal."""
    d, p, q, _ = imgs.shape
    return imgs.reshape(d, p * q, 3).transpose(2, 0, 1).copy()


def from_channels(X: np.ndarray, p: int, q: int) -> np.ndarray:
    _, d, _ = X.shape
    return np.clip(X.transpose(1, 2, 0).reshape(d, p, q, 3), 0, 1)
