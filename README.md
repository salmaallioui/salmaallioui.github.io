# Salma Allioui · Portafolio de datos

Ingeniera de Datos · SQL, Python y estadística.

[LinkedIn](https://www.linkedin.com/in/salma-allioui-845a72190/) · [GitHub](https://github.com/salmaallioui)

🌐 **Web:** [salmaallioui.github.io](https://salmaallioui.github.io)

## Proyectos

| Proyecto | Qué hace | Stack |
|---|---|---|
| [Monitor de calidad de datos](projects/data-quality-monitor) | Checks automáticos de volumen, duplicados, nulos, rangos y frescura con SQL y estadística robusta, más un informe HTML de alertas | SQL · Python · pandas · pytest |
| [Análisis de un test A/B](projects/ab-test-analysis) | Experimento analizado de principio a fin: potencia, SRM, inferencia frecuentista y bayesiana, bootstrap, Holm y simulación de *peeking* | Python · SciPy · NumPy · pytest |
| [NMF dispersa conjunta ℓ∞ (TFM)](projects/sparse-nmf-color) | NMF que fuerza ceros conjuntos en los canales RGB, con algoritmos Greedy y VNS y la métrica JZR | Python · NumPy · SciPy |

Cada proyecto incluye su README, tests e informe reproducible. Todos los datos son sintéticos.

## Ejecutar los proyectos

```bash
cd projects/data-quality-monitor && pip install -r requirements.txt && python -m dqmonitor.generate_data && python -m dqmonitor && pytest
cd ../ab-test-analysis && pip install -r requirements.txt && python -m abtest.simulate && python -m abtest && pytest
cd ../sparse-nmf-color && pip install -r requirements.txt && python -m snmf && pytest
```
