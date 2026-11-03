"""Informe HTML del experimento."""
from __future__ import annotations

CSS = """
body{font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;color:#1f2a44;max-width:880px;margin:40px auto;padding:0 20px;line-height:1.6}
h1{font-size:26px;margin:0}h2{font-size:17px;margin-top:36px;border-bottom:1px solid #e5e7eb;padding-bottom:6px}
.sub{color:#6b7280}.box{background:#eef3fa;border-left:4px solid #1f3a5f;padding:14px 18px;border-radius:6px;margin:20px 0}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:right;padding:7px 8px;border-bottom:1px solid #eef0f3}
th:first-child,td:first-child{text-align:left}th{color:#6b7280;font-weight:600}tr.hl td{font-weight:700;background:#f6f8fb}
img{width:100%;max-width:720px;display:block;margin:12px 0}code{background:#f3f4f6;padding:1px 5px;border-radius:4px}
footer{color:#9ca3af;font-size:12px;margin:40px 0}
"""


def f(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def render(rows: dict) -> str:
    trs = "".join(
        f"<tr class='{'hl' if n == 'ℓ∞ + VNS' else ''}'><td>{n}</td><td>{f(r['eF'])} %</td><td>{f(r['psnr'], 1)} dB</td>"
        f"<td>{f(r['ssim'], 3)}</td><td>{f(r['jzr'], 1)} %</td><td>{f(r['channel_zero_rate'], 1)} %</td>"
        f"<td>{f(r['joint_share'], 0)} %</td><td>{f(r['hoyer'], 3)}</td></tr>" for n, r in rows.items())
    l1, g, v, s = rows["ℓ1 separable"], rows["ℓ∞ + Greedy"], rows["ℓ∞ + VNS"], rows["NMF estándar"]
    return f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NMF dispersa conjunta ℓ∞ para imágenes en color</title><style>{CSS}</style></head><body>
<h1>NMF dispersa conjunta ℓ∞ para imágenes en color</h1>
<div class="sub">Greedy y VNS sobre el soporte · métrica JZR (Joint Zero Rate) · basado en mi Trabajo Fin de Máster
«Técnicas de Reducción de la Dimensionalidad. Aplicaciones a Imágenes»</div>

<div class="box"><b>Idea.</b> En una imagen RGB, una NMF dispersa normal (ℓ1) pone ceros canal a canal: un píxel puede quedar
"apagado" en rojo pero no en verde ni en azul, así que no desaparece de verdad. La penalización <b>ℓ∞ conjunta</b> trata
el trío (H<sub>R</sub>, H<sub>G</sub>, H<sub>B</sub>) de cada posición como un grupo y penaliza su <i>máximo</i>: la única
forma de bajar la penalización es apagar los tres canales a la vez. La <b>JZR</b> mide el porcentaje de posiciones con los
tres canales a cero.</div>

<h2>Resultados</h2>
<table><thead><tr><th>Método</th><th>ε<sub>F</sub></th><th>PSNR</th><th>SSIM</th><th>JZR</th><th>Ceros por canal</th>
<th>% de ceros que son conjuntos</th><th>Hoyer</th></tr></thead><tbody>{trs}</tbody></table>
<ul>
<li><b>ℓ1 separable</b> pone muchos ceros ({f(l1['channel_zero_rate'], 1)} %), pero solo el {f(l1['joint_share'], 0)} % son
conjuntos. Con la ℓ∞ conjunta + VNS lo son el <b>{f(v['joint_share'], 0)} %</b>.</li>
<li>Frente a la NMF estándar, VNS multiplica la JZR ({f(s['jzr'], 1)} % → {f(v['jzr'], 1)} %) y además reconstruye mejor
(ε<sub>F</sub> {f(s['eF'])} % → {f(v['eF'])} %, SSIM {f(s['ssim'], 3)} → {f(v['ssim'], 3)}).</li>
<li>VNS mejora a Greedy en calidad (ε<sub>F</sub> {f(g['eF'])} % → {f(v['eF'])} %, PSNR {f(g['psnr'], 1)} → {f(v['psnr'], 1)} dB)
manteniendo la JZR, porque escapa del óptimo local en el que se queda Greedy.</li>
</ul>

<h2>Soporte de los átomos</h2>
<p>Azul = los tres canales activos; amarillo = <b>cero parcial</b> (algún canal a cero, pero no todos); blanco = cero conjunto.
Con ℓ1 dominan las zonas amarillas: ceros que no eliminan el píxel. Con ℓ∞ + VNS los ceros pasan a ser mayoritariamente conjuntos (blancos) y el amarillo queda reducido a puntos sueltos.</p>
<img src="figures/supports.png" alt="Soportes">
<img src="figures/jzr.png" alt="JZR">

<h2>Reconstrucciones</h2>
<img src="figures/reconstructions.png" alt="Reconstrucciones">

<h2>Algoritmos</h2>
<p><b>Modelo.</b> Para cada canal c: X<sub>c</sub> ≈ W<sub>c</sub>H<sub>c</sub>, minimizando
Σ<sub>c</sub> ½‖X<sub>c</sub> − W<sub>c</sub>H<sub>c</sub>‖²<sub>F</sub> + λ Σ<sub>i,j</sub> max<sub>c</sub> H<sub>c</sub>[i,j],
con actualizaciones multiplicativas y columnas de W normalizadas (evita que la penalización se esquive reescalando).</p>
<p><b>Greedy (conjunto activo).</b> En cada ronda anula los grupos con max(H<sub>R</sub>, H<sub>G</sub>, H<sub>B</sub>) &lt; τ,
con τ creciente, en los tres canales a la vez; reajusta sobre el soporte restante y para cuando el criterio
(error + α·% de grupos activos) deja de mejorar.</p>
<p><b>VNS (Variable Neighbourhood Search).</b> Sacude la solución con vecindarios de tamaño κ creciente (perturbación de W y
reapertura del soporte); la búsqueda local es ajuste ℓ∞ + Greedy. Si mejora, acepta y vuelve a κ = 1.</p>
<img src="figures/vns_trace.png" alt="Convergencia VNS" style="max-width:520px">
<footer>Reimplementación para el portafolio con un dataset sintético de 12 imágenes RGB 32×32 · Salma Allioui</footer>
</body></html>"""
