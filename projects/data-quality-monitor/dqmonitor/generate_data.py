"""Genera una base de datos SQLite sintética de pedidos con anomalías inyectadas.

Las anomalías permiten comprobar que el monitor las detecta:
  * caída de volumen  (día 95: solo ~35 % de los pedidos)
  * pico de volumen   (día 70: carga duplicada de un lote, ~2x)
  * pico de nulos     (días 100-102: ~30 % de `customer_id` nulos)
  * duplicados        (día 70: el lote duplicado reutiliza `order_id`)
  * valores fuera de rango (día 108: importes negativos)
  * datos no frescos  (los últimos 2 días no se cargan)
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

START = date(2026, 3, 1)
N_DAYS = 120
COUNTRIES = ["ES", "PT", "FR", "IT", "MA"]
CHANNELS = ["web", "app", "tienda"]


def build_orders(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weekday_factor = np.array([1.0, 0.95, 0.97, 1.02, 1.15, 1.35, 0.8])  # lun..dom
    frames, next_id = [], 1
    for d in range(N_DAYS - 2):  # los 2 últimos días "no llegan" -> frescura
        day = START + timedelta(days=d)
        trend = 1 + 0.002 * d
        lam = 400 * trend * weekday_factor[day.weekday()]
        n = rng.poisson(lam)
        if d == 95:
            n = int(n * 0.35)
        df = pd.DataFrame({
            "order_id": np.arange(next_id, next_id + n),
            "order_date": day.isoformat(),
            "customer_id": rng.integers(1, 20_000, n).astype("float"),
            "country": rng.choice(COUNTRIES, n, p=[0.55, 0.15, 0.12, 0.1, 0.08]),
            "channel": rng.choice(CHANNELS, n, p=[0.5, 0.35, 0.15]),
            "amount_eur": np.round(rng.lognormal(3.6, 0.6, n), 2),
        })
        next_id += n
        if 100 <= d <= 102:
            df.loc[rng.random(n) < 0.30, "customer_id"] = np.nan
        if d == 108:
            idx = rng.choice(n, 25, replace=False)
            df.loc[df.index[idx], "amount_eur"] = -df.loc[df.index[idx], "amount_eur"]
        if d == 70:
            df = pd.concat([df, df.sample(frac=0.95, random_state=seed)])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["customer_id"] = out["customer_id"].astype("Int64")
    return out


def write_db(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as con:
        con.execute("""
            CREATE TABLE orders (
                order_id    INTEGER NOT NULL,
                order_date  TEXT    NOT NULL,
                customer_id INTEGER,
                country     TEXT,
                channel     TEXT,
                amount_eur  REAL
            )""")
        df.to_sql("orders", con, if_exists="append", index=False)
        con.execute("CREATE INDEX ix_orders_date ON orders(order_date)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/orders.db")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    df = build_orders(a.seed)
    write_db(df, Path(a.db))
    print(f"{len(df):,} filas escritas en {a.db}")


if __name__ == "__main__":
    main()
