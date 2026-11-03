"""CLI: python -m dqmonitor --db data/orders.db --as-of 2026-06-28"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from .checks import Config, alerts_frame, run_checks
from .report import render


def main() -> None:
    p = argparse.ArgumentParser(prog="dqmonitor", description="Monitor de calidad de datos")
    p.add_argument("--db", default="data/orders.db")
    p.add_argument("--as-of", default="2026-06-28", help="fecha de corte (YYYY-MM-DD)")
    p.add_argument("--days", type=int, default=120, help="días hacia atrás a analizar")
    p.add_argument("--out", default="reports")
    a = p.parse_args()

    as_of = date.fromisoformat(a.as_of)
    cfg = Config()
    with sqlite3.connect(a.db) as con:
        profile, alerts = run_checks(con, as_of, as_of - timedelta(days=a.days - 1), cfg)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    alerts_frame(alerts).to_csv(out / "alerts.csv", index=False)
    (out / "report.html").write_text(render(profile, alerts, as_of, cfg.null_rate_max), encoding="utf-8")
    print(f"{len(alerts)} alertas -> {out / 'report.html'}")
    for al in alerts:
        print(f"  [{al.severity:5}] {al.day} {al.check:10} {al.message}")
    raise SystemExit(1 if any(al.severity == "alta" for al in alerts) else 0)


if __name__ == "__main__":
    main()
