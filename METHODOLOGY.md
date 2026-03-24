# Metodologia Matematica

Fundamentos teoricos del analisis de valores extremos implementado en este repositorio.

---

## 1. Contexto: por que EVT

Los modelos clasicos de riesgo financiero asumen retornos con distribucion normal
o t-Student. Sin embargo, las distribuciones de retornos reales exhiben:

- **Colas pesadas** (mas masa de probabilidad en los extremos que la normal)
- **Asimetria** (la cola izquierda suele ser mas gruesa que la derecha)
- **Clustering de volatilidad** (los eventos extremos tienden a agruparse)

La **Teoria de Valores Extremos (EVT)** proporciona un marco teorico riguroso
para modelar exclusivamente el comportamiento en las colas, sin imponer supuestos
sobre la distribucion central.

---

## 2. Block Maxima y la distribucion GEV

### 2.1 Teorema de Fisher-Tippett-Gnedenko

Sea $X_1, X_2, \ldots$ una sucesion de variables aleatorias i.i.d. y sea
$M_n = \max(X_1, \ldots, X_n)$. Si existen sucesiones $a_n > 0$ y $b_n$ tales que:

$$\frac{M_n - b_n}{a_n} \xrightarrow{d} G$$

donde $G$ es una distribucion no degenerada, entonces $G$ pertenece a la familia
**GEV (Generalized Extreme Value)**:

$$G(x; \mu, \sigma, \xi) = \exp\left\{-\left[1 + \xi\left(\frac{x - \mu}{\sigma}\right)\right]^{-1/\xi}\right\}$$

definida para $1 + \xi(x - \mu)/\sigma > 0$, con parametros:

| Parametro | Significado |
|---|---|
| $\mu \in \mathbb{R}$ | **Locacion**: centro de la distribucion de maximos |
| $\sigma > 0$ | **Escala**: dispersion de los maximos |
| $\xi \in \mathbb{R}$ | **Forma**: peso de la cola |

### 2.2 Subclases segun $\xi$

| Condicion | Nombre | Comportamiento de la cola |
|---|---|---|
| $\xi > 0$ | **Frechet** | Cola pesada (polinomial). Sin cota superior. Comun en finanzas. |
| $\xi = 0$ | **Gumbel** | Cola ligera (exponencial). Caso limite. |
| $\xi < 0$ | **Weibull** | Cola acotada. Tiene un maximo finito en $\mu - \sigma/\xi$. |

### 2.3 Implementacion: Block Maxima

1. Dividir la serie de perdidas $L_1, \ldots, L_N$ en $k$ bloques de tamano $m$
   (tipicamente $m = 50$ dias)
2. Calcular el maximo de cada bloque: $M_i = \max(L_{(i-1)m+1}, \ldots, L_{im})$
3. Ajustar la GEV a la muestra $M_1, \ldots, M_k$ via **Maximum Likelihood Estimation (MLE)**

> **Nota sobre SciPy:** `scipy.stats.genextreme` usa la convencion $c = -\xi$,
> por lo que internamente se convierte con `xi = -c`.

### 2.4 Nivel de retorno GEV

El nivel de retorno $x_T$ para un periodo de $T$ bloques es el cuantil:

$$x_T = G^{-1}\left(1 - \frac{1}{T}\right) = \mu - \frac{\sigma}{\xi}\left[1 - \left(-\log\left(1 - \frac{1}{T}\right)\right)^{-\xi}\right]$$

Interpretacion: la probabilidad de que el maximo de un bloque supere $x_T$ es $1/T$.

### 2.5 VaR y ES del maximo de bloque

**VaR al nivel $p$** (cuantil de la distribucion GEV):

$$\text{VaR}_p = G^{-1}(p) = \mu - \frac{\sigma}{\xi}\left[1 - (-\log p)^{-\xi}\right]$$

**Expected Shortfall** (formula cerrada para $\xi < 1$):

$$\text{ES}_p = \frac{\text{VaR}_p}{1 - \xi} + \frac{\sigma - \xi \mu}{1 - \xi}$$

Para el caso Gumbel ($\xi \approx 0$), se usa integracion numerica:

$$\text{ES}_p = \frac{1}{1-p} \int_p^1 G^{-1}(u) \, du$$

Para $\xi \geq 1$, el ES no existe (la esperanza de la cola es infinita).

---

## 3. Peaks Over Threshold y la distribucion GPD

### 3.1 Teorema de Pickands-Balkema-de Haan

Sea $X$ una variable aleatoria con funcion de distribucion $F$ y sea $u$ un umbral alto.
La distribucion de los **excesos** $Y = X - u \mid X > u$ converge, para $u$
suficientemente grande, a la **GPD (Generalized Pareto Distribution)**:

$$H(y; \xi, \beta) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi}$$

para $y \geq 0$ y $1 + \xi y / \beta > 0$, con:

| Parametro | Significado |
|---|---|
| $\beta > 0$ | **Escala**: magnitud tipica de los excesos |
| $\xi \in \mathbb{R}$ | **Forma**: mismo rol que en la GEV |

### 3.2 Implementacion: POT

1. Elegir un umbral $u$ (tipicamente el cuantil 95% de las perdidas)
2. Calcular los excesos: $Y_i = L_i - u$ para todo $L_i > u$
3. Ajustar la GPD a los excesos via MLE con `scipy.stats.genpareto`

### 3.3 Eleccion del umbral

La eleccion de $u$ implica un trade-off:

- **$u$ muy alto**: pocos excesos → alta varianza del estimador
- **$u$ muy bajo**: sesgo por incluir datos que no pertenecen a la cola

Este proyecto usa un percentil configurable (default: 95%). Para un analisis
mas riguroso, se recomienda complementar con el **mean excess plot** (pendiente
de implementacion).

### 3.4 VaR y ES via POT

Combinando la GPD con la probabilidad empirica de exceder el umbral $\hat{p}_u = P(X > u)$:

**VaR al nivel $p$** ($\xi \neq 0$):

$$\text{VaR}_p = u + \frac{\beta}{\xi}\left[\left(\frac{\hat{p}_u}{1-p}\right)^{\xi} - 1\right]$$

**VaR al nivel $p$** (limite $\xi \to 0$):

$$\text{VaR}_p = u + \beta \log\left(\frac{\hat{p}_u}{1-p}\right)$$

**Expected Shortfall** ($\xi < 1$):

$$\text{ES}_p = \frac{\text{VaR}_p}{1 - \xi} + \frac{\beta - \xi u}{1 - \xi}$$

### 3.5 Nivel de retorno POT

El nivel de retorno $z_T$ tal que $P(X > z_T) \approx 1/T$:

$$z_T = u + \frac{\beta}{\xi}\left[(T \cdot \hat{p}_u)^{\xi} - 1\right] \qquad (\xi \neq 0)$$

$$z_T = u + \beta \log(T \cdot \hat{p}_u) \qquad (\xi \approx 0)$$

---

## 4. Relacion entre GEV y GPD

Los parametros de forma $\xi$ de la GEV y la GPD son **el mismo** cuando ambos
modelos se aplican a la misma distribucion subyacente. Esto es una consecuencia
directa de la dualidad entre los teoremas de Fisher-Tippett y Pickands-Balkema-de Haan.

En la practica, las estimaciones pueden diferir ligeramente porque:

- Usan subconjuntos distintos de los datos (maximos vs excesos)
- El MLE tiene propiedades de muestra finita diferentes en cada caso
- La eleccion del block size y del umbral introduce sesgos diferentes

---

## 5. Diagnosticos y validacion

### QQ-Plot

Se comparan los cuantiles empiricos contra los teoricos del modelo ajustado.
Si el modelo es adecuado, los puntos deben alinearse sobre la diagonal.
Desviaciones sistematicas en las colas indican que el modelo subestima o
sobreestima el riesgo extremo.

### Advertencias automaticas

| Condicion | Significado |
|---|---|
| $\xi \geq 1$ | La media de la distribucion no existe. El ES es infinito. |
| $\xi \geq 0.5$ | Cola muy pesada. Los estimadores MLE pueden ser inestables. |
| $< 10$ bloques | Muestra insuficiente para ajuste GEV confiable. |
| $< 30$ excesos | Muestra insuficiente para ajuste GPD confiable. |
| $< 200$ observaciones | Serie original corta para analisis de colas. |

---

## 6. Limitaciones conocidas

### Supuestos del modelo

- **Estacionariedad**: EVT asume que la distribucion subyacente no cambia en el tiempo.
  En mercados financieros, esto raramente se cumple estrictamente (regimenes de
  volatilidad, cambios estructurales). Aplicar EVT a periodos homogeneos mitiga
  parcialmente este problema.

- **Independencia**: los teoremas requieren observaciones i.i.d. Los retornos
  financieros exhiben autocorrelacion en la volatilidad (efecto ARCH/GARCH).
  EVT sigue siendo aplicable, pero las tasas de convergencia pueden ser mas lentas.

- **Eleccion de hiperparametros**: tanto el tamano del bloque como el umbral POT
  son elecciones del usuario que afectan los resultados. No hay un criterio
  universalmente optimo.

### Limitaciones de la implementacion

- **Estimacion puntual**: no se proporcionan intervalos de confianza para los
  parametros ni para VaR/ES. Un analisis robusto deberia incluir bootstrap
  o profile likelihood.

- **Umbral fijo**: se usa un percentil fijo en lugar de metodos adaptativos
  como el mean excess plot o el metodo de Wadsworth.

- **Unilateral**: solo se modela la cola derecha de las perdidas. Para activos
  con riesgo de short squeeze, seria necesario un analisis bilateral.

- **Sin modelos de volatilidad**: no se pre-filtra la serie con GARCH.
  La combinacion EVT + GARCH (enfoque McNeil-Frey) mejoraria las estimaciones
  condicionales.

---

## 7. Referencias

1. **McNeil, A. J., Frey, R., & Embrechts, P.** (2015).
   *Quantitative Risk Management: Concepts, Techniques and Tools* (Revised Edition).
   Princeton University Press.
   — Referencia principal para VaR/ES via EVT, formulas de GPD, y el enfoque
   POT en gestion de riesgos.

2. **Coles, S.** (2001).
   *An Introduction to Statistical Modeling of Extreme Values*.
   Springer.
   — Tratamiento accesible de la teoria GEV y GPD con ejemplos aplicados.

3. **Embrechts, P., Kluppelberg, C., & Mikosch, T.** (1997).
   *Modelling Extremal Events for Insurance and Finance*.
   Springer.
   — Fundamentos teoricos rigurosos de EVT aplicada.

4. **Fisher, R. A., & Tippett, L. H. C.** (1928).
   "Limiting forms of the frequency distribution of the largest or smallest
   member of a sample."
   *Proceedings of the Cambridge Philosophical Society*, 24, 180–190.
   — Teorema original de tipos extremos.

5. **Pickands, J.** (1975).
   "Statistical inference using extreme order statistics."
   *Annals of Statistics*, 3, 119–131.
   — Teorema de convergencia para excesos (base del metodo POT).

6. **Balkema, A. A., & de Haan, L.** (1974).
   "Residual life time at great age."
   *Annals of Probability*, 2, 792–804.
   — Resultado complementario al de Pickands para la GPD.
