"""Ejecuta el análisis completo: python -m abtest --data data/experiment.csv"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import stats as st
from .report import render

A_COL, B_COL, MUTED, RED = "#9aa5b8", "#1f3a5f", "#6b7280", "#c2410c"
BASELINE, MDE = 0.08, 0.01


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d1d5db")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color="#eef0f3", lw=0.8)
    ax.set_axisbelow(True)


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def analyse(df: pd.DataFrame, fig_dir: Path) -> dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    g = df.groupby("variant").agg(n=("converted", "size"), x=("converted", "sum"))
    nA, nB, xA, xB = int(g.n.A), int(g.n.B), int(g.x.A), int(g.x.B)
    res: dict = {}

    # 1. Diseño y validez
    res["design"] = {"baseline": BASELINE, "mde_pp": MDE, "alpha": 0.05, "power": 0.8,
                     "n_required_per_arm": st.sample_size_two_proportions(BASELINE, BASELINE + MDE),
                     "n_a": nA, "n_b": nB, "srm_p_value": st.srm_test(nA, nB)}
    ns = np.arange(1000, 30001, 500)
    fig, ax = plt.subplots(figsize=(6, 3))
    for mde, c in [(0.005, A_COL), (0.01, B_COL), (0.015, MUTED)]:
        ax.plot(ns, [st.power_two_proportions(BASELINE, BASELINE + mde, n) for n in ns], color=c, lw=1.6,
                label=f"+{mde * 100:.1f} p.p.")
    ax.axhline(0.8, color=RED, ls="--", lw=1)
    ax.axvline(min(nA, nB), color=MUTED, ls=":", lw=1)
    ax.set_xlabel("usuarios por grupo", color=MUTED, fontsize=8)
    ax.set_ylabel("potencia", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=8, title="efecto", title_fontsize=8)
    _style(ax)
    _save(fig, fig_dir / "power_curve.png")

    # 2. Conversión (frecuentista)
    conv = st.two_proportion_ztest(xA, nA, xB, nB)
    res["conversion"] = {k: float(v) for k, v in conv.__dict__.items()}

    # 3. Ingreso por usuario: Welch + bootstrap
    ra = df.loc[df.variant == "A", "revenue_eur"].to_numpy()
    rb = df.loc[df.variant == "B", "revenue_eur"].to_numpy()
    w = st.welch_ttest(ra, rb)
    lo, hi, diffs = st.bootstrap_diff_ci(ra, rb, n_boot=4000)
    res["revenue"] = {**w, "mean_a": float(ra.mean()), "mean_b": float(rb.mean()),
                      "bootstrap_ci": [lo, hi]}
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.hist(diffs, bins=60, color=B_COL, alpha=0.85)
    ax.axvline(0, color=RED, lw=1)
    ax.axvspan(lo, hi, color=A_COL, alpha=0.25)
    ax.set_xlabel("diferencia de ingreso medio por usuario (B − A, €)", color=MUTED, fontsize=8)
    _style(ax)
    _save(fig, fig_dir / "revenue_bootstrap.png")

    # 4. Bayesiano
    bb = st.beta_binomial(xA, nA, xB, nB)
    res["bayes"] = {k: v for k, v in bb.items() if not k.startswith("samples")}
    fig, ax = plt.subplots(figsize=(6, 2.8))
    bins = np.linspace(0.072, 0.1, 120)
    ax.hist(bb["samples_a"] * 1, bins=bins, color=A_COL, alpha=0.8, density=True, label="A (control)")
    ax.hist(bb["samples_b"], bins=bins, color=B_COL, alpha=0.8, density=True, label="B (nuevo checkout)")
    ax.set_xlabel("tasa de conversión (posterior)", color=MUTED, fontsize=8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=1))
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    _save(fig, fig_dir / "posterior.png")

    # 5. Segmentos con corrección de Holm
    seg = []
    for dev, sub in df.groupby("device"):
        gg = sub.groupby("variant").converted.agg(["sum", "size"])
        r = st.two_proportion_ztest(int(gg["sum"].A), int(gg["size"].A), int(gg["sum"].B), int(gg["size"].B))
        seg.append({"segment": dev, "n": int(len(sub)), "p_a": r.p_a, "p_b": r.p_b, "diff": r.diff,
                    "ci_low": r.ci_low, "ci_high": r.ci_high, "p_value": r.p_value})
    for s, adj in zip(seg, st.holm([s["p_value"] for s in seg])):
        s["p_holm"] = adj
    res["segments"] = seg
    fig, ax = plt.subplots(figsize=(6, 1.9))
    rows = [("Global", conv.diff, conv.ci_low, conv.ci_high)] + [(s["segment"].capitalize(), s["diff"], s["ci_low"], s["ci_high"]) for s in seg]
    for i, (lab, d, l, h) in enumerate(rows[::-1]):
        ax.plot([l * 100, h * 100], [i, i], color=B_COL, lw=2)
        ax.plot(d * 100, i, "o", color=B_COL)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=8)
    ax.axvline(0, color=RED, lw=1)
    ax.set_xlabel("diferencia de conversión B − A (p.p., IC 95 %)", color=MUTED, fontsize=8)
    _style(ax)
    ax.grid(False)
    _save(fig, fig_dir / "segments.png")

    # 6. Peeking
    pk = st.peeking_false_positive_rate(BASELINE, n_per_arm=12_000, looks=20, n_sims=2000)
    res["peeking"] = {k: pk[k] for k in ("fixed_horizon", "with_peeking")}
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.plot(range(1, 21), np.array(pk["cumulative"]) * 100, color=B_COL, marker="o", ms=3, lw=1.5)
    ax.axhline(5, color=RED, ls="--", lw=1)
    ax.text(1, 6, "α nominal = 5 %", color=RED, fontsize=8)
    ax.set_xlabel("número de miradas intermedias", color=MUTED, fontsize=8)
    ax.set_ylabel("% falsos positivos (A/A)", color=MUTED, fontsize=8)
    _style(ax)
    _save(fig, fig_dir / "peeking.png")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(prog="abtest")
    ap.add_argument("--data", default="data/experiment.csv")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    out = Path(a.out)
    res = analyse(pd.read_csv(a.data), out / "figures")
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.html").write_text(render(res), encoding="utf-8")
    c = res["conversion"]
    print(f"Conversión A {c['p_a']:.2%} vs B {c['p_b']:.2%}  (p={c['p_value']:.3f})  "
          f"P(B>A)={res['bayes']['prob_b_better']:.1%}  -> {out / 'report.html'}")


if __name__ == "__main__":
    main()
