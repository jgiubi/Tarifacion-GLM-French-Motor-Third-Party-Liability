# Tarificación GLM — French Motor Third-Party Liability

Construcción de una tarifa de seguro automotor de responsabilidad civil sobre el dataset
**freMTPL2** de la Casualty Actuarial Society, modelando frecuencia y severidad de siniestros
por separado y derivando la **prima pura** como el producto de sus esperanzas.

## Datos

- `freMTPL2freq` — pólizas, exposición y cantidad de siniestros
- `freMTPL2sev` — montos de siniestro
- ~680.000 pólizas, 9 variables de riesgo (potencia, antigüedad y marca del vehículo, edad del
  conductor, combustible, área, densidad, región, bonus-malus)

## Enfoque

**1. Análisis exploratorio y limpieza**
Detección de outliers en cantidad de siniestros y severidad. Recorte de exposiciones mayores a 1
año (inconsistentes). Truncado de la cola de `ClaimAmount`. Inspección de la forma de cada
distribución vía histogramas para orientar la elección de familia.

**2. Preparación**
Agrupamiento (binning) de variables continuas. El binning es manual y no automático por cuantiles:
la elección de cortes se documenta y es ajustable, porque impacta directamente en la tarifa
resultante. Marcas poco representadas se consolidan en `Other Brand`.

**3. Selección de variables**
- Chi-cuadrado de independencia para relevancia frente a frecuencia y a presencia de siniestro
- VIF y matriz de correlaciones para multicolinealidad
- **`BonusMalus` se excluye por endogeneidad**: es un score que ya refleja la siniestralidad
  pasada del asegurado, por lo que incluirlo contamina el modelo. Corresponde aplicarlo como
  recargo/descuento *sobre* la tarifa, no como predictor dentro de ella.
- `Region` y `Densidad` se excluyen por quedar contenidas en `Area`, más abarcativa y con mejor
  exposición por categoría.

**4. Modelado**

| Componente | Familia elegida | Criterio |
|---|---|---|
| Frecuencia | Binomial Negativa (offset `log(Exposure)`) | Se detecta sobredispersión — la varianza excede la media, por lo que Poisson no ajusta. La BN mejora Deviance, AIC y log-likelihood. |
| Severidad | Inversa Gaussiana | Mejora sustancialmente el ajuste frente a las alternativas evaluadas. |

Cada decisión de inclusión/exclusión de variables se contrasta comparando Deviance, AIC, BIC y
log-likelihood entre modelos anidados, no por significatividad individual aislada: variables no
significativas por sí solas se conservan cuando su remoción empeora el modelo en conjunto.

**5. Diagnóstico**
Test RESET de especificación (H0: modelo bien especificado). El resultado indica que quedan
relaciones no capturadas — no invalida el modelo, pero señala margen de mejora vía términos
polinómicos, interacciones o tratamiento continuo de variables hoy discretizadas.

**6. Tarifa**
Predicción de frecuencia (ajustada por exposición) y severidad → prima pura. Traducción de los
coeficientes a **factores multiplicativos** por nivel de cada variable, e impresión de la
**grilla tarifaria** para las distintas combinaciones de perfil de riesgo.

## Resultados

- Grilla tarifaria completa por combinación de factores de riesgo
- Tabla de coeficientes multiplicativos interpretables por nivel
- Diagnóstico de calibración: el modelo **sobreestima la frecuencia** y **estima la severidad con
  buena precisión**

Interpretación económica de cada factor documentada en el notebook (por qué mayor potencia,
menor edad del conductor o determinadas áreas se asocian a mayor riesgo).

## Limitaciones y próximos pasos

- El modelo está mal especificado según RESET: faltan relaciones por capturar
- Comparar contra la vía directa **Tweedie** (modelar la prima en un solo paso)
- Extender a **GAM** y **gradient boosting** para contrastar poder predictivo frente al GLM
- Sumar validación cruzada (k-fold), lift curves y QQ-plots al diagnóstico

## Stack

Python — pandas, numpy, statsmodels, scipy, patsy, matplotlib, seaborn

## Reproducir

El notebook lee los CSV desde una ruta local. Para correrlo, descargar `freMTPL2freq` y
`freMTPL2sev` (disponibles en CASdatasets / OpenML) y ajustar las rutas en la celda de carga.

## Referencias

- [Tutorial freMTPL2 — Lorentzen & Mayer](https://github.com/lorentzenchr/Tutorial_freMTPL2/blob/master/glm_freMTPL2_example.ipynb)
- [Insurance pricing with machine learning — SSRN 3164764](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3164764)
