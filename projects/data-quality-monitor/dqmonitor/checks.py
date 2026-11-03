"""Checks de calidad de datos sobre el perfil diario que devuelve SQL."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


@dataclass
class Config:
    volume_window_weeks: int = 6      # semanas previas (mismo día de la semana) para la línea base
    volume_z_threshold: float = 3.5   # umbral de z-score robusto (mediana / MAD)
    volume_min_rel_change: float = 0.25  # además, el cambio relativo debe superar ±25 %
    null_rate_max: float = 0.02       # % máximo de customer_id nulos
    max_lag_days: int = 1             # días de retraso tolerados
    min_history: int = 3              # observaciones mínimas para evaluar volumen


@dataclass
class Alert:
    day: str
    check: str
    severity: str  # "alta" | "media"
    value: float
    expected: str
    message: str


def _sql(name: str) -> str:
    return (SQL_DIR / name).read_text(encoding="utf-8")


def daily_profile(con: sqlite3.Connection, start: str, end: str) -> pd.DataFrame:
    df = pd.read_sql_query(_sql("daily_profile.sql"), con, params={"start": start, "end": end})
    df["day"] = pd.to_datetime(df["day"])
    return df


def robust_volume_scores(profile: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Para cada día compara el volumen con los mismos días de la semana anteriores.

    z = 0.6745 * (x - mediana) / MAD   (Iglewicz & Hoaglin). Es robusto a que la
    propia historia contenga anomalías, a diferencia de media / desviación típica.
    """
    out = profile[["day", "row_count"]].copy()
    out["baseline"] = np.nan
    out["z"] = np.nan
    s = out.set_index("day")["row_count"]
    for i, d in enumerate(out["day"]):
        prev = [d - pd.Timedelta(weeks=w) for w in range(1, cfg.volume_window_weeks + 1)]
        hist = s.reindex(prev).dropna()
        if len(hist) < cfg.min_history:
            continue
        med = hist.median()
        mad = (hist - med).abs().median()
        mad = max(mad, 0.6745 * np.sqrt(med), 1.0)  # suelo ~ ruido de Poisson: evita MAD≈0
        out.loc[i, "baseline"] = med
        out.loc[i, "z"] = 0.6745 * (out.loc[i, "row_count"] - med) / mad
    return out


def run_checks(con: sqlite3.Connection, as_of: date, start: date, cfg: Config | None = None):
    cfg = cfg or Config()
    profile = daily_profile(con, start.isoformat(), as_of.isoformat())
    vol = robust_volume_scores(profile, cfg)
    profile = profile.merge(vol[["day", "baseline", "z"]], on="day")
    alerts: list[Alert] = []

    for r in profile.itertuples():
        d = r.day.date().isoformat()
        rel = r.row_count / r.baseline - 1 if not np.isnan(r.baseline) else 0.0
        if not np.isnan(r.z) and abs(r.z) > cfg.volume_z_threshold and abs(rel) >= cfg.volume_min_rel_change:
            kind = "caída" if r.z < 0 else "pico"
            pct = 100 * rel
            alerts.append(Alert(d, "volumen", "alta" if abs(r.z) > 2 * cfg.volume_z_threshold else "media",
                                r.row_count, f"≈{r.baseline:,.0f}",
                                f"{kind.capitalize()} de volumen: {r.row_count:,} filas ({pct:+.0f} % vs. línea base, z={r.z:.1f})"))
        if r.duplicate_ids > 0:
            alerts.append(Alert(d, "duplicados", "alta", r.duplicate_ids, "0",
                                f"{r.duplicate_ids:,} order_id duplicados"))
        if r.null_rate_customer_id > cfg.null_rate_max:
            alerts.append(Alert(d, "nulos", "media", round(r.null_rate_customer_id, 4), f"≤ {cfg.null_rate_max:.0%}",
                                f"{r.null_rate_customer_id:.1%} de customer_id nulos"))
        if r.out_of_range_amount > 0:
            alerts.append(Alert(d, "rango", "media", r.out_of_range_amount, "0",
                                f"{r.out_of_range_amount} importes fuera de [0, 5000] €"))

    last = con.execute(_sql("freshness.sql")).fetchone()[0]
    lag = (as_of - date.fromisoformat(last)).days if last else None
    if lag is None or lag > cfg.max_lag_days:
        alerts.append(Alert(as_of.isoformat(), "frescura", "alta", lag if lag is not None else -1,
                            f"≤ {cfg.max_lag_days} día(s)",
                            f"Último dato cargado: {last} ({lag} días de retraso)"))

    alerts.sort(key=lambda a: (a.day, a.check))
    return profile, alerts


def alerts_frame(alerts: list[Alert]) -> pd.DataFrame:
    return pd.DataFrame([asdict(a) for a in alerts],
                        columns=["day", "check", "severity", "value", "expected", "message"])
