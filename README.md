# EVT Analyzer — Extreme Value Theory para Finanzas

> Analisis de riesgo extremo con Block Maxima (GEV) y Peaks Over Threshold (GPD).
> Modulo Python + aplicacion web que corre en el browser via Pyodide.

[SCREENSHOT]

**[Abrir demo en vivo (GitHub Pages)](https://IgnacioMorard.github.io/EVT-Analysis-ES/web/)**

---

## Que es esto y para que sirve

Los modelos financieros tradicionales asumen que los retornos siguen una distribucion
normal, donde los eventos extremos (crashes, cisnes negros) son practicamente imposibles.
La realidad muestra lo contrario: las colas de la distribucion de perdidas son mucho mas
gruesas. La **Teoria de Valores Extremos (EVT)** es la rama de la estadistica dedicada
exclusivamente a modelar estos eventos raros pero de alto impacto.

Este proyecto implementa los dos metodos principales de EVT — **Block Maxima** y
**Peaks Over Threshold** — y los aplica a datos financieros reales. En vez de asumir
normalidad, ajusta distribuciones matematicas disenadas para capturar colas pesadas,
y con ellas calcula metricas de riesgo como el VaR extremo, el Expected Shortfall y
niveles de retorno. Todo se puede usar desde Python o directamente en el browser sin
instalar nada.

---

## Que puedo hacer con esto?

| Pregunta | El analisis responde |
|---|---|
| **"Si invierto en SPY, cual es la peor perdida diaria que puedo esperar en el 99.5% de los dias?"** | El VaR al 99.5% via GPD te da ese numero exacto, usando la cola real de la distribucion — no una campana de Gauss. |
| **"Cada cuanto ocurre un crash tipo marzo 2020 (-10% en un dia)?"** | El nivel de retorno te dice: "una perdida de X% ocurre aproximadamente una vez cada N anios", calibrado con datos historicos reales. |
| **"Si el VaR se rompe, que tan mal puede ponerse?"** | El Expected Shortfall (ES) mide exactamente eso: la perdida promedio en los dias que superan el VaR. Es la metrica regulatoria de Basilea III. |

---

## Inicio rapido

### Opcion 1: Web (sin instalar Python)

Abrí la [demo en GitHub Pages](https://IgnacioMorard.github.io/EVT-Analysis-ES/web/),
**subi un CSV** y hacé click en "Analizar".
Todo corre en tu browser via Pyodide — tus datos no salen de tu maquina.

> **Primera carga:** descarga ~60 MB de paquetes Python (numpy, pandas, scipy,
> matplotlib). Las siguientes cargas usan la cache del browser y son mas rapidas.
>
> **Ticker de Yahoo Finance:** solo funciona corriendo la app en un servidor
> local (ver abajo). Desde GitHub Pages, Yahoo bloquea las peticiones por CORS.

#### Correr la web en local (habilita tickers)

```bash
git clone https://github.com/IgnacioMorard/EVT-Analysis-ES.git
cd Extreme-Value-Theory-EVT-/web
python -m http.server 8000
```

Abri [http://localhost:8000](http://localhost:8000) — desde ahi el tab "Ticker"
funciona con cualquier simbolo de Yahoo Finance (SPY, BTC-USD, ^GSPC, etc.).

### Opcion 2: Python

```bash
git clone https://github.com/IgnacioMorard/EVT-Analysis-ES.git
cd Extreme-Value-Theory-EVT-
pip install -r requirements.txt
```

#### Ejemplo minimo

```python
from data_loader import get_returns
from evt_engine import run_evt_analysis
from report_generator import generate_report

# Cargar datos (ticker, CSV, o DataFrame)
returns = get_returns("SPY", start="2020-01-01", end="2025-01-01")

# Correr analisis completo
results = run_evt_analysis(returns)

# Ver resultados
print(f"VaR 99.5% (GPD): {results['gpd']['var']:.4f}")
print(f"ES  99.5% (GPD): {results['gpd']['es']:.4f}")

# Generar reporte HTML
generate_report(results, output_path="mi_reporte.html")
```

#### Ejemplo con CSV local y parametros custom

```python
returns = get_returns("SPY_US - Cotizaciones historicas.csv",
                      start="2023-01-01", end="2025-01-01")

results = run_evt_analysis(returns, config={
    "block_size": 25,           # bloques mas chicos
    "quantile_threshold": 0.90, # umbral mas bajo → mas excesos
    "p_conf": 0.99,             # VaR/ES al 99%
})
```

#### Ejemplo con cripto

```python
returns = get_returns("BTC-USD", start="2021-01-01", end="2025-01-01")
results = run_evt_analysis(returns)
# xi alto → cola muy pesada (esperado en cripto)
```

---

## Arquitectura

```
┌─────────────────┐     ┌──────────────┐     ┌────────────────────┐
│  data_loader.py │────>│ evt_engine.py│────>│ report_generator.py│
│                 │     │              │     │                    │
│ get_returns()   │     │ run_evt_     │     │ generate_report()  │
│ - CSV local     │     │ analysis()   │     │ - HTML standalone  │
│ - yfinance      │     │ - GEV + GPD  │     │ - base64 charts    │
│ - datareader    │     │ - VaR/ES     │     │ - explicaciones    │
│ - DataFrame     │     │ - 8 figuras  │     │   no-tecnicas      │
└─────────────────┘     └──────────────┘     └────────────────────┘
        │                       │
        v                       v
  ┌──────────┐          ┌────────────┐
  │config.py │          │Utilidades_ │
  │ defaults │          │GEV_BM_2.py │
  └──────────┘          │ math core  │
                        └────────────┘
```

### Modulos

| Archivo | Responsabilidad |
|---|---|
| `config.py` | Constantes por defecto (block_size, umbrales, niveles de confianza) |
| `data_loader.py` | Carga unificada: CSV, ticker, DataFrame. Fallback chain: yfinance → pandas_datareader. Validaciones de calidad. |
| `Utilidades_GEV_BM_2.py` | Funciones matematicas puras: ajuste GEV/GPD, VaR, ES, niveles de retorno |
| `evt_engine.py` | Orquestador: toma returns + config, ejecuta GEV + GPD, devuelve dict con resultados + figuras |
| `report_generator.py` | Genera HTML standalone con explicaciones en lenguaje simple y graficos embebidos |
| `web/` | SPA con Pyodide: corre el analisis en el browser via Web Worker. Tickers solo en local (CORS). |

### Estructura del dict de resultados

```python
results = run_evt_analysis(returns)

results["metadata"]     # source, n_obs, start, end, config
results["gev"]          # params(xi,mu,sigma), var, es, return_level, figures
results["gpd"]          # params(xi,beta), threshold, var, es, return_level, figures
results["diagnostics"]  # lista de advertencias (xi alto, pocos excesos, etc.)
```

---

## Parametros configurables

| Parametro | Default | Descripcion |
|---|---|---|
| `block_size` | 50 | Observaciones por bloque para Block Maxima |
| `quantile_threshold` | 0.95 | Percentil para el umbral POT |
| `p_conf` | 0.995 | Nivel de confianza para VaR y ES |
| `t_blocks` | 100 | Periodo de retorno en bloques (GEV) |
| `t_days` | 5000 | Periodo de retorno en dias (GPD) |

Se pueden sobreescribir pasando un dict a `run_evt_analysis()` o ajustandolos
en la UI web.

---

## Activos soportados

`data_loader.get_returns()` acepta cualquier ticker de Yahoo Finance:

| Tipo | Ejemplos |
|---|---|
| Acciones US | `AAPL`, `MSFT`, `GOOGL` |
| Acciones latam | `GGAL.BA` (Merval), `VALE3.SA` (Bovespa) |
| ETFs | `SPY`, `QQQ`, `EWZ`, `GLD` |
| Indices | `^GSPC`, `^VIX`, `^MERV` |
| Criptos | `BTC-USD`, `ETH-USD` |
| Commodities | `GLD` (oro), `USO` (petroleo) |
| CSV local | Cualquier CSV con columna de fecha y cierre |

---

## Limitaciones de la version web

| Limitacion | Causa | Workaround |
|---|---|---|
| **Ticker depende de proxy externo** | Yahoo Finance no envia headers CORS; se usa corsproxy.io como proxy | Si el proxy esta caido, subir CSV manualmente |
| **Primera carga lenta (~60 MB)** | Pyodide + numpy/pandas/scipy/matplotlib se descargan del CDN | Las siguientes cargas usan cache del browser |
| **No funciona offline** | Pyodide se carga desde CDN (cross-origin, no se cachea por el Service Worker) | Requiere conexion a internet en cada sesion nueva |

---

## Metodologia

Ver **[METHODOLOGY.md](METHODOLOGY.md)** para la explicacion matematica completa
con formulas, referencias y limitaciones conocidas.

---

## Estructura del repositorio

```
.
├── config.py                    # Constantes por defecto
├── data_loader.py               # Carga unificada de datos financieros
├── Utilidades_GEV_BM_2.py      # Funciones matematicas EVT
├── evt_engine.py                # Orquestador de analisis
├── report_generator.py          # Generador de reportes HTML
├── requirements.txt             # Dependencias Python
├── METHODOLOGY.md               # Explicacion matematica detallada
├── SPY_US - Cotizaciones historicas.csv  # Datos de ejemplo
└── web/
    ├── index.html               # SPA principal
    ├── evt_core.py              # Modulo EVT para Pyodide
    ├── worker.js                # Web Worker (Pyodide off-thread)
    └── sw.js                    # Service Worker (cache archivos locales)
```

---

## Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature: `git checkout -b feature/mi-mejora`
3. Las funciones matematicas van en `Utilidades_GEV_BM_2.py`; la orquestacion en `evt_engine.py`
4. Si agregas una metrica nueva, agregala tambien a `report_generator.py` y `web/evt_core.py`
5. Asegurate de no romper la logica matematica existente — la prioridad es la correccion numerica
6. Abrí un Pull Request con descripcion de que cambia y por que

### Ideas para contribuir

- Agregar bootstrap confidence intervals para VaR/ES
- Implementar el mean excess plot para seleccion automatica de umbral
- Agregar soporte para POT bilateral (colas izquierda y derecha)
- Tests unitarios con pytest
- Deploy automatico a GitHub Pages via GitHub Actions

---

## Licencia

Este proyecto es de uso educativo y de investigacion.
