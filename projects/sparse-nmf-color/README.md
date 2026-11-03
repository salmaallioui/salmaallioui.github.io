# NMF dispersa conjunta ℓ∞ para imágenes en color

Este proyecto parte de mi **Trabajo Fin de Máster**, «Técnicas de Reducción de la Dimensionalidad. Aplicaciones a Imágenes». Es una factorización no negativa (NMF) dispersa pensada para imágenes RGB: penaliza el **máximo entre canales**, de modo que los ceros se producen **en los tres canales a la vez**. Para medirlo uso la métrica **JZR (Joint Zero Rate)**, y el soporte disperso se busca con dos algoritmos: **Greedy** (conjunto activo) y **VNS** (Variable Neighbourhood Search).

👉 **[Ver el informe](reports/report.html)**

## El problema

Una NMF dispersa con ℓ1 pone los ceros canal a canal. Así, un píxel puede quedar a cero en rojo pero seguir activo en verde y azul, y en realidad no desaparece de la imagen. La penalización ℓ∞ conjunta agrupa (H<sub>R</sub>, H<sub>G</sub>, H<sub>B</sub>) en cada posición:

$$\min_{W_c,H_c\ge 0}\ \sum_{c\in\{R,G,B\}} \tfrac12\|X_c-W_cH_c\|_F^2 \;+\; \lambda\sum_{i,j}\max_c H_c[i,j]$$

Bajar el máximo es la única forma de reducir la penalización, así que la solución tiende a anular el grupo completo.

## Resultados (12 imágenes RGB sintéticas de 32×32, k = 6)

| Método | ε<sub>F</sub> | PSNR | SSIM | JZR | % de los ceros que son conjuntos |
|---|---|---|---|---|---|
| NMF estándar | 4,22 % | 39,7 dB | 0,954 | 3,7 % | 16 % |
| ℓ1 separable | 4,66 % | 38,8 dB | 0,940 | 28,6 % | 51 % |
| ℓ∞ conjunta | 4,54 % | 39,0 dB | 0,928 | 14,0 % | 57 % |
| ℓ∞ + Greedy | 3,79 % | 40,6 dB | 0,944 | 29,4 % | 81 % |
| **ℓ∞ + VNS** | **3,37 %** | **41,7 dB** | **0,958** | **29,9 %** | **86 %** |

- Con ℓ1, la mitad de los ceros son **parciales** y no apagan el píxel. Con ℓ∞ + VNS, el 86 % de los ceros son conjuntos.
- Comparado con la NMF estándar, VNS pasa la JZR del 3,7 % al 29,9 % y además **reconstruye mejor**.
- La NMF no es convexa, y Greedy se queda en el óptimo local más cercano a su punto de partida. VNS sale de él perturbando la solución con vecindarios de tamaño creciente, y así mejora la calidad sin perder JZR.

## Algoritmos

- **Ajuste:** actualizaciones multiplicativas (Lee y Seung), normalizando las columnas de W para que la penalización no se pueda esquivar reescalando. El subgradiente de `max` se reparte de forma proporcional entre los canales, porque usar solo el canal dominante hace que oscile cuando dos canales están casi empatados.
- **Greedy:** anula los grupos con `max(H_R, H_G, H_B) < τ`, subiendo τ poco a poco, reajusta, y se detiene cuando el criterio `ε_F + α·(% de grupos activos)` deja de mejorar.
- **VNS:** el vecindario N<sub>κ</sub> perturba W con un ruido log-normal de escala κ·s y reabre el soporte. La búsqueda local es ajuste ℓ∞ + Greedy. Si la solución mejora se acepta y κ vuelve a 1; si no, κ aumenta.

## Uso

```bash
pip install -r requirements.txt
python -m snmf      # genera reports/report.html, results.json y figures/
pytest              # 6 tests: JZR, gradiente de la penalización, convergencia, SSIM...
```

**Stack:** Python · NumPy · SciPy · matplotlib · pytest

*Es una reimplementación  con datos sintéticos (zonas de color con transiciones gaussianas). Las cifras corresponden a este experimento, no a las del TFM.*
