"""Informe HTML del experimento (las figuras se enlazan desde reports/figures)."""
from __future__ import annotations

CSS = """
body{font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;color:#1f2a44;max-width:860px;margin:40px auto;padding:0 20px;line-height:1.6}
h1{font-size:26px;margin:0}h2{font-size:17px;margin-top:36px;border-bottom:1px solid #e5e7eb;padding-bottom:6px}
.sub{color:#6b7280}.kpis{display:flex;gap:12px;margin:22px 0;flex-wrap:wrap}
.kpi{flex:1;min-width:150px;border:1px solid #e5e7eb;border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:20px;white-space:nowrap}.kpi span{color:#6b7280;font-size:13px}
.verdict{background:#eef3fa;border-left:4px solid #1f3a5f;padding:14px 18px;border-radius:6px;margin:20px 0}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid #eef0f3}
th{color:#6b7280;font-weight:600}img{width:100%;max-width:640px;display:block;margin:10px 0}
footer{color:#9ca3af;font-size:12px;margin:40px 0}
"""


def pct(x, d=2):
    return f"{x * 100:.{d}f} %".replace(".", ",")


def pp(x, d=2):
    return f"{x * 100:+.{d}f}".replace(".", ",") + " p.p."


def intf(n):
    return f"{n:,}".replace(",", ".")


def num(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def render(r: dict) -> str:
    d, c, rev, b, pk = r["design"], r["conversion"], r["revenue"], r["bayes"], r["peeking"]
    seg_rows = "".join(
        f"<tr><td>{s['segment'].capitalize()}</td><td>{intf(s['n'])}</td><td>{pct(s['p_a'])}</td><td>{pct(s['p_b'])}</td><td>{pp(s['diff'])}</td>"
        f"<td>[{pp(s['ci_low'])}, {pp(s['ci_high'])}]</td><td>{num(s['p_value'])}</td><td><b>{num(s['p_holm'])}</b></td></tr>"
        for s in r["segments"])
    mob = next(s for s in r["segments"] if s["segment"] == "móvil")
    desk = next(s for s in r["segments"] if s["segment"] == "escritorio")
    kpis = [("Conversión A → B", f"{pct(c['p_a'])} → {pct(c['p_b'])}"), ("Lift relativo", f"{c['rel_lift'] * 100:+.1f} %".replace(".", ",")),
            ("p-valor (z-test)", num(c["p_value"])), ("P(B > A) bayesiana", pct(b["prob_b_better"], 1))]
    kpi_html = "".join(f"<div class='kpi'><b>{v}</b><span>{k}</span></div>" for k, v in kpis)
    return f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Test A/B · nuevo checkout</title><style>{CSS}</style></head><body>
<h1>Test A/B: rediseño del checkout</h1>
<div class="sub">{intf(d['n_a'] + d['n_b'])} usuarios asignados al azar 50/50 · métrica principal: conversión · secundaria: ingreso por usuario</div>
<div class="kpis">{kpi_html}</div>
<div class="verdict"><b>Recomendación:</b> lanzar el nuevo checkout. La conversión sube {pp(c['diff'])} (IC 95 % [{pp(c['ci_low'])}, {pp(c['ci_high'])}]),
la probabilidad posterior de que B sea mejor es {pct(b['prob_b_better'], 1)} y la pérdida esperada de elegir B es casi nula
({num(b['expected_loss_choose_b'] * 100)} p.p.). El efecto se concentra en <b>móvil</b>. El impacto en ingreso por usuario
<b>todavía no es concluyente</b> (p = {num(rev['p_value'], 2)}), así que conviene seguir midiéndolo tras el lanzamiento.</div>

<h2>1. Diseño y validez</h2>
<p>Con una conversión base del {pct(d['baseline'], 0)} y un efecto mínimo de interés de +{d['mde_pp'] * 100:.0f} p.p. (α = 5 %, potencia 80 %)
se necesitan <b>{intf(d['n_required_per_arm'])} usuarios por grupo</b>. El experimento alcanzó {intf(d['n_a'])} (A) y {intf(d['n_b'])} (B).
El test de <i>Sample Ratio Mismatch</i> (χ²) da p = {num(d['srm_p_value'], 2)}: la asignación es coherente con 50/50, sin indicios de fallos de randomización.</p>
<img src="figures/power_curve.png" alt="Curvas de potencia">

<h2>2. Conversión: enfoque frecuentista</h2>
<p>z-test de dos proporciones: diferencia {pp(c['diff'])}, IC 95 % [{pp(c['ci_low'])}, {pp(c['ci_high'])}], z = {num(c['z'], 2)}, p = {num(c['p_value'])}.
Significativo al 5 %, aunque el límite inferior del intervalo está cerca de 0: el efecto es real pero su tamaño es incierto.</p>

<h2>3. Conversión: enfoque bayesiano</h2>
<p>Modelo Beta-Binomial con prior uniforme Beta(1,1). P(B &gt; A) = <b>{pct(b['prob_b_better'], 1)}</b>; IC creíble 95 % del lift relativo
[{f"{b['lift_ci'][0] * 100:+.1f}".replace(".", ",")} %, {f"{b['lift_ci'][1] * 100:+.1f}".replace(".", ",")} %]. Pérdida esperada eligiendo B: {num(b['expected_loss_choose_b'] * 100)} p.p.;
eligiendo A: {num(b['expected_loss_choose_a'] * 100)} p.p. La pérdida esperada es más útil para negocio que el p-valor porque mide cuánto se arriesga con cada decisión.</p>
<img src="figures/posterior.png" alt="Posteriores">

<h2>4. Ingreso por usuario</h2>
<p>El ingreso tiene muchos ceros y una cola larga, así que la t de Welch se complementa con un IC bootstrap (4.000 remuestreos).
Media A {num(rev['mean_a'], 2)} € vs B {num(rev['mean_b'], 2)} €; diferencia {num(rev['diff'], 2)} €, IC bootstrap
[{num(rev['bootstrap_ci'][0], 2)}, {num(rev['bootstrap_ci'][1], 2)}] €, p (Welch) = {num(rev['p_value'], 2)}. <b>No concluyente</b>:
el intervalo es compatible tanto con pérdidas como con ganancias. Con la varianza del ticket, detectar +1 % en ingreso exigiría muchos más usuarios.</p>
<img src="figures/revenue_bootstrap.png" alt="Bootstrap ingreso">

<h2>5. Segmentos (corrección de Holm)</h2>
<p>Al mirar varios segmentos aumenta la probabilidad de encontrar un falso positivo, así que los p-valores se ajustan con Holm-Bonferroni.</p>
<table><thead><tr><th>Segmento</th><th>Usuarios</th><th>A</th><th>B</th><th>Dif.</th><th>IC 95 %</th><th>p</th><th>p Holm</th></tr></thead><tbody>{seg_rows}</tbody></table>
<p>En <b>móvil</b> el efecto es {pp(mob['diff'])} y sigue siendo significativo tras el ajuste (p Holm = {num(mob['p_holm'])}). En escritorio
({pp(desk['diff'])}) no se distingue de cero. Hipótesis: el rediseño reduce fricción sobre todo en pantallas pequeñas.</p>
<img src="figures/segments.png" alt="Segmentos">

<h2>6. Por qué no mirar el test cada día (<i>peeking</i>)</h2>
<p>Simulación de 2.000 tests A/A (sin efecto real). Con un único análisis al final, la tasa de falsos positivos es
{pct(pk['fixed_horizon'], 1)}, como se espera. Si se revisa el p-valor en 20 momentos y se para en cuanto p &lt; 0,05, sube a
<b>{pct(pk['with_peeking'], 1)}</b>. Por eso este experimento se analizó una sola vez, con el tamaño muestral fijado de antemano.
Si se quiere monitorizar durante el test, hay que usar métodos secuenciales (por ejemplo, O'Brien-Fleming).</p>
<img src="figures/peeking.png" alt="Peeking">
<footer>Datos sintéticos generados con <code>abtest/simulate.py</code> · Salma Allioui</footer>
</body></html>"""
