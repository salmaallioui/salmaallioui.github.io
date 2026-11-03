# Análisis estadístico de un test A/B

Análisis completo de un experimento A/B sobre el rediseño del checkout de una tienda online (25.000 usuarios). Cubre todas las fases: **diseño (potencia y tamaño muestral) → validez (SRM) → análisis frecuentista y bayesiano → métrica secundaria con bootstrap → segmentos con corrección por comparaciones múltiples → recomendación de negocio**. Además incluye una simulación que muestra por qué no hay que "mirar" el test antes de tiempo.

👉 **[Ver el informe completo](reports/report.html)**

## Resultado

> **Lanzar el nuevo checkout.** La conversión pasa del 8,23 % al 8,96 % (+0,73 p.p., IC 95 % [+0,03, +1,42], p = 0,040). P(B > A) = 98 % y la pérdida esperada de elegir B es prácticamente nula. El efecto se concentra en **móvil** (+1,07 p.p., p Holm = 0,028). El ingreso por usuario **no es concluyente** (p = 0,86) y hay que seguir midiéndolo.

| Fase | Método | Resultado |
|---|---|---|
| Tamaño muestral | z-test de dos proporciones, α = 5 %, potencia 80 %, MDE +1 p.p. | 12.208 por grupo (se alcanzaron 12.429 / 12.571) |
| Validez | Sample Ratio Mismatch (χ²) | p = 0,37: asignación correcta |
| Conversión | z-test + IC de Wald | +0,73 p.p., p = 0,040 |
| Conversión | Beta-Binomial, prior Beta(1,1) | P(B>A) = 98 %, lift [+0,4 %, +18 %] |
| Ingreso/usuario | t de Welch + IC bootstrap (4.000 remuestreos) | +0,05 €, IC [−0,51, +0,64] €: no concluyente |
| Segmentos | z-test por dispositivo + Holm-Bonferroni | Móvil significativo; escritorio no |
| *Peeking* | 2.000 tests A/A simulados con 20 miradas | falsos positivos: 4,7 % → **24,2 %** |

Como los datos son simulados, se conoce el efecto real (+1,2 p.p. en móvil y +0,4 p.p. en escritorio), y el análisis lo recupera.

## Uso

```bash
pip install -r requirements.txt
python -m abtest.simulate        # genera data/experiment.csv
python -m abtest                 # reports/report.html, results.json y figures/
pytest                           # 9 tests: fórmulas, Holm, SRM, bootstrap, peeking...
```

## Estructura

```
abtest/stats.py      funciones estadísticas (NumPy + SciPy), testeadas
abtest/simulate.py   generador del experimento con efecto heterogéneo conocido
abtest/__main__.py   pipeline de análisis y figuras
abtest/report.py     informe HTML
tests/               pytest
```

**Stack:** Python · NumPy · SciPy · pandas · matplotlib · pytest

*Datos 100 % sintéticos.*
