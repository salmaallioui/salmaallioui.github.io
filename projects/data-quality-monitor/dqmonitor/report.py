"""Informe HTML autocontenido con gráficos y tabla de alertas."""
from __future__ import annotations

import base64
import html
import io
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .checks import Alert

INK, MUTED, ACCENT, RED, AMBER = "#1f2a44", "#6b7280", "#1f3a5f", "#c2410c", "#b45309"


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color="#eef0f3", lw=0.8)
    ax.set_axisbelow(True)


def volume_chart(profile: pd.DataFrame, alerts: list[Alert]) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(profile["day"], profile["row_count"], color=ACCENT, lw=1.4, label="Filas por día")
    ax.plot(profile["day"], profile["baseline"], color=MUTED, lw=1, ls="--", label="Línea base (mediana mismo día semana)")
    vol_days = pd.to_datetime([a.day for a in alerts if a.check == "volumen"])
    hit = profile[profile["day"].isin(vol_days)]
    ax.scatter(hit["day"], hit["row_count"], color=RED, zorder=3, s=28, label="Alerta de volumen")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylabel("filas", color=MUTED, fontsize=8)
    return _png(fig)


def null_chart(profile: pd.DataFrame, threshold: float) -> str:
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.bar(profile["day"], profile["null_rate_customer_id"] * 100, color=ACCENT, width=0.8)
    ax.axhline(threshold * 100, color=RED, lw=1, ls="--")
    ax.text(profile["day"].iloc[0], threshold * 100 + 0.6, f"umbral {threshold:.0%}", color=RED, fontsize=8)
    _style(ax)
    ax.set_ylabel("% nulos customer_id", color=MUTED, fontsize=8)
    return _png(fig)


CSS = """
body{font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;color:#1f2a44;max-width:960px;margin:40px auto;padding:0 20px;line-height:1.5}
h1{font-size:26px;margin:0}h2{font-size:17px;margin-top:34px;border-bottom:1px solid #e5e7eb;padding-bottom:6px}
.sub{color:#6b7280;margin-top:4px}.kpis{display:flex;gap:12px;margin:22px 0;flex-wrap:wrap}
.kpi{flex:1;min-width:140px;border:1px solid #e5e7eb;border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px}.kpi span{color:#6b7280;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid #eef0f3}
th{color:#6b7280;font-weight:600}.sev{padding:2px 8px;border-radius:99px;font-size:12px;font-weight:600}
.alta{background:#fde8e1;color:#c2410c}.media{background:#fdf3d8;color:#92400e}img{width:100%}
footer{color:#9ca3af;font-size:12px;margin:40px 0}
"""


def render(profile: pd.DataFrame, alerts: list[Alert], as_of: date, null_threshold: float) -> str:
    by_check = pd.Series([a.check for a in alerts]).value_counts()
    rows = "".join(
        f"<tr><td>{a.day}</td><td>{a.check}</td><td><span class='sev {a.severity}'>{a.severity}</span></td>"
        f"<td>{html.escape(a.message)}</td><td>{html.escape(a.expected)}</td></tr>" for a in alerts)
    kpis = [("Días analizados", len(profile)), ("Filas totales", f"{profile['row_count'].sum():,}"),
            ("Alertas", len(alerts)), ("Alertas altas", sum(a.severity == "alta" for a in alerts))]
    kpi_html = "".join(f"<div class='kpi'><b>{v}</b><span>{k}</span></div>" for k, v in kpis)
    checks_html = ", ".join(f"{k}: {v}" for k, v in by_check.items()) or "ninguna"
    return f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Informe de calidad de datos · orders</title>
<style>{CSS}</style></head><body>
<h1>Informe de calidad de datos — tabla <code>orders</code></h1>
<div class="sub">Fecha de corte: {as_of.isoformat()} · Alertas por tipo: {checks_html}</div>
<div class="kpis">{kpi_html}</div>
<h2>Volumen diario</h2>
<p class="sub">Cada día se compara con la mediana de los mismos días de la semana de las 6 semanas anteriores (z-score robusto con MAD).</p>
<img alt="Volumen diario" src="data:image/png;base64,{volume_chart(profile, alerts)}">
<h2>Completitud de <code>customer_id</code></h2>
<img alt="Nulos" src="data:image/png;base64,{null_chart(profile, null_threshold)}">
<h2>Alertas</h2>
<table><thead><tr><th>Día</th><th>Check</th><th>Severidad</th><th>Detalle</th><th>Esperado</th></tr></thead>
<tbody>{rows}</tbody></table>
<footer>Generado por dqmonitor · datos sintéticos · Salma Allioui</footer>
</body></html>"""
