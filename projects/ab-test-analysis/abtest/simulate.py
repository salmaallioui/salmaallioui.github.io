"""Simula un experimento A/B de rediseño del checkout de una tienda online.

Verdad oculta (lo que el análisis debería recuperar):
  * Conversión control 8,0 %; variante +1 p.p. en móvil y +0,4 p.p. en escritorio.
  * El ticket medio de quien compra no cambia (lognormal, cola larga).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRUE_CONV = {("A", "móvil"): 0.075, ("B", "móvil"): 0.087,
             ("A", "escritorio"): 0.088, ("B", "escritorio"): 0.092}


def simulate(n_users: int = 25_000, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    variant = rng.choice(["A", "B"], n_users)
    device = rng.choice(["móvil", "escritorio"], n_users, p=[0.62, 0.38])
    p = np.array([TRUE_CONV[(v, d)] for v, d in zip(variant, device)])
    converted = rng.random(n_users) < p
    revenue = np.where(converted, np.round(rng.lognormal(3.9, 0.7, n_users), 2), 0.0)
    return pd.DataFrame({"user_id": np.arange(1, n_users + 1), "variant": variant, "device": device,
                         "converted": converted.astype(int), "revenue_eur": revenue})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/experiment.csv")
    ap.add_argument("--n", type=int, default=25_000)
    a = ap.parse_args()
    df = simulate(a.n)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df):,} usuarios -> {a.out}")


if __name__ == "__main__":
    main()
