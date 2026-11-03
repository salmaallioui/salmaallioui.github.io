"""Experimento completo: python -m snmf"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import metrics as M
from .data import from_channels, make_images, to_channels
from .models import fit, greedy, init_factors, vns
from .report import render

NAVY, GREY, LIGHT, RED, AMBER = "#1f3a5f", "#6b7280", "#9aa5b8", "#c2410c", "#d4a017"
P = Q = 32
K = 6


def run(seed: int = 0) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    imgs = make_images(seed=seed)
    X = to_channels(imgs)
    F0 = init_factors(X, K, seed)
    models = {}
    models["NMF estándar"] = fit(X, F0, "none", 0, 600)
    models["ℓ1 separable"] = fit(X, F0, "l1", 0.05, 600)
    joint = fit(X, F0, "joint_linf", 0.02, 600)
    models["ℓ∞ conjunta"] = joint
    models["ℓ∞ + Greedy"] = greedy(X, joint, alpha=0.05)
    models["ℓ∞ + VNS"] = vns(X, models["ℓ∞ + Greedy"], alpha=0.05, max_iter=30, seed=seed)

    rows = {}
    for name, F in models.items():
        Xh = F.reconstruct()
        zr = M.channel_zero_rate(F.H)
        rows[name] = {
            "eF": M.rel_frobenius(X, Xh),
            "psnr": M.psnr(X, np.clip(Xh, 0, 1)),
            "ssim": M.mean_ssim(imgs, from_channels(Xh, P, Q)),
            "jzr": M.jzr(F.H),
            "channel_zero_rate": zr,
            "joint_share": 100 * M.jzr(F.H) / zr if zr else 0.0,
            "hoyer": M.hoyer(F.H),
        }
    return rows, models, imgs, X


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d1d5db")
    ax.tick_params(colors=GREY, labelsize=8)
    ax.grid(axis="y", color="#eef0f3", lw=0.8)
    ax.set_axisbelow(True)


def support_map(H: np.ndarray) -> np.ndarray:
    """Imagen RGB del soporte de cada átomo:
    blanco = 3 canales a cero (cero conjunto), color = canales activos,
    gris = cero parcial (algún canal a cero pero no todos)."""
    k, n = H.shape[1:]
    on = H > 1e-8                                  # (3, k, n)
    n_on = on.sum(0)
    out = np.ones((k, n, 3))
    out[n_on == 3] = np.array([0.12, 0.23, 0.37])  # activo en los 3
    out[(n_on > 0) & (n_on < 3)] = np.array([0.83, 0.63, 0.10])  # cero parcial
    return out.reshape(k, P, Q, 3)


def figures(rows: dict, models: dict, imgs: np.ndarray, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    names = list(rows)

    # 1. Reconstrucciones
    show = [0, 3, 7]
    pick = ["NMF estándar", "ℓ1 separable", "ℓ∞ + VNS"]
    fig, axes = plt.subplots(len(show), 1 + len(pick), figsize=(7.2, 5.6))
    for r, t in enumerate(show):
        axes[r, 0].imshow(imgs[t])
        for c, name in enumerate(pick, 1):
            axes[r, c].imshow(from_channels(models[name].reconstruct(), P, Q)[t])
        for c, title in enumerate(["Original"] + pick):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(title, fontsize=9, color="#1f2a44")
    fig.tight_layout()
    fig.savefig(out / "reconstructions.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 2. Soporte de los átomos: ℓ1 separable vs ℓ∞ + VNS
    fig, axes = plt.subplots(2, K, figsize=(9, 3.4))
    for r, name in enumerate(["ℓ1 separable", "ℓ∞ + VNS"]):
        smap = support_map(models[name].H)
        for i in range(K):
            axes[r, i].imshow(smap[i])
            axes[r, i].set_xticks([]); axes[r, i].set_yticks([])
        axes[r, 0].set_ylabel(name, fontsize=9, color="#1f2a44")
    fig.suptitle("Soporte de los 6 átomos · azul: 3 canales activos · amarillo: cero parcial · blanco: cero conjunto",
                 fontsize=8.5, color=GREY)
    fig.tight_layout()
    fig.savefig(out / "supports.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 3. JZR vs ceros por canal
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, [rows[n]["channel_zero_rate"] for n in names], w, color=LIGHT, label="% ceros contando canal a canal")
    ax.bar(x + w / 2, [rows[n]["jzr"] for n in names], w, color=NAVY, label="JZR: % ceros conjuntos (3 canales)")
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("%", color=GREY, fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.savefig(out / "jzr.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 4. Convergencia de VNS
    trace = [h for h in models["ℓ∞ + VNS"].history if isinstance(h, tuple) and h[0] == "vns_trace"][-1][1]
    fig, ax = plt.subplots(figsize=(6, 2.4))
    ax.plot(range(len(trace)), trace, color=NAVY, marker="o", ms=2.5, lw=1.4)
    ax.set_xlabel("iteración VNS", color=GREY, fontsize=8)
    ax.set_ylabel("criterio", color=GREY, fontsize=8)
    _style(ax)
    fig.savefig(out / "vns_trace.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(prog="snmf")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    rows, models, imgs, _ = run(a.seed)
    figures(rows, models, imgs, out / "figures")
    (out / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.html").write_text(render(rows), encoding="utf-8")
    for n, r in rows.items():
        print(f"{n:14} εF={r['eF']:5.2f}%  PSNR={r['psnr']:5.1f}  SSIM={r['ssim']:.3f}  "
              f"JZR={r['jzr']:5.1f}%  ceros conjuntos/ceros={r['joint_share']:5.1f}%")


if __name__ == "__main__":
    main()
