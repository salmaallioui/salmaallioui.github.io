# Monitor de calidad de datos (SQL + Python)

Sistema ligero de **checks automáticos de calidad de datos** sobre una tabla de pedidos: perfila la tabla con SQL, detecta anomalías con estadística robusta y genera un informe HTML y un CSV de alertas. Devuelve un código de salida distinto de 0 si hay alertas de severidad alta, así que puede ejecutarse en un cron o en un pipeline de CI para bloquear cargas defectuosas.

👉 **[Ver el informe de ejemplo](reports/report.html)**

## Qué detecta

| Check | Cómo | Severidad |
|---|---|---|
| **Volumen** (caídas y picos) | z-score robusto `0.6745·(x − mediana)/MAD` contra los **mismos días de la semana** de las 6 semanas anteriores, más un cambio relativo mínimo de ±25 % | alta si \|z\| > 7 |
| **Duplicados** | `COUNT(*) − COUNT(DISTINCT order_id)` por día | alta |
| **Completitud** | % de `customer_id` nulos > 2 % | media |
| **Rango** | importes < 0 € o > 5.000 € | media |
| **Frescura** | último día cargado con más de 1 día de retraso respecto a la fecha de corte | alta |

### ¿Por qué mediana + MAD y no media + desviación típica?
La línea base se calcula sobre datos históricos que pueden contener anomalías. Una sola carga duplicada infla la media y la desviación típica, y el check deja de ver la siguiente. La mediana y la MAD tienen un punto de ruptura del 50 %, así que siguen siendo fiables. Comparar con el mismo día de la semana absorbe la estacionalidad semanal (los sábados hay un 35 % más de pedidos). Como la MAD de 6 observaciones puede ser casi 0, se le pone un suelo igual al ruido de Poisson esperado (√mediana).

## Resultado sobre los datos de ejemplo

El generador inyecta 6 anomalías en 120 días de datos sintéticos (~54.000 pedidos). El monitor las detecta **todas sin falsos positivos**:

- Pico de +99 % y 344 `order_id` duplicados (lote cargado dos veces)
- Caída de volumen del −63 %
- 3 días con ~30 % de `customer_id` nulos
- 25 importes negativos
- 2 días sin cargar datos (frescura)

## Uso

```bash
pip install -r requirements.txt
python -m dqmonitor.generate_data            # crea data/orders.db (SQLite)
python -m dqmonitor --as-of 2026-06-28       # genera reports/report.html y reports/alerts.csv
pytest                                       # tests: detecta todo, sin falsos positivos, tabla limpia = 0 alertas
```

## Estructura

```
sql/                    consultas de perfilado (SQL parametrizado)
dqmonitor/checks.py     lógica de los checks y z-score robusto
dqmonitor/report.py     informe HTML autocontenido (gráficos matplotlib)
dqmonitor/__main__.py   CLI
tests/                  tests con pytest
```

**Stack:** Python · SQL (SQLite) · pandas · NumPy · matplotlib · pytest

*Datos 100 % sintéticos generados por `generate_data.py`.*
