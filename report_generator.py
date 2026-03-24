"""
Generador de reportes HTML standalone para resultados EVT.

Toma el dict devuelto por ``evt_engine.run_evt_analysis()`` y produce
un archivo HTML autocontenido (figuras embebidas como base64, sin
dependencias externas) legible para audiencias no técnicas.
"""

from __future__ import annotations

import base64
import io
import math
from datetime import datetime
from typing import Any, Dict

from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: Figure, dpi: int = 130) -> str:
    """Renderiza una Figure de matplotlib a una cadena base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    return encoded


def _pct(value: float) -> str:
    """Formatea un valor como porcentaje con 2 decimales."""
    return f"{value * 100:.2f}%"


def _f4(value: float) -> str:
    """Formatea un float a 4 decimales."""
    if math.isinf(value):
        return "&infin;"
    return f"{value:.4f}"


def _f6(value: float) -> str:
    """Formatea un float a 6 decimales."""
    if math.isinf(value):
        return "&infin;"
    return f"{value:.6f}"


def _xi_interpretation(xi: float) -> str:
    """Devuelve una interpretación en lenguaje simple del parámetro de forma."""
    if xi > 0.5:
        return ("Cola <strong>muy pesada</strong>. Los eventos extremos son "
                "significativamente más frecuentes de lo que sugiere una "
                "distribución normal. Se requiere cautela extra en la "
                "gestión de riesgo.")
    if xi > 0:
        return ("Cola <strong>pesada</strong>. Hay más riesgo de eventos "
                "extremos del que asumiría un modelo con distribución normal.")
    if abs(xi) < 0.05:
        return ("Cola similar a una distribución exponencial (tipo Gumbel). "
                "El riesgo extremo es moderado.")
    return ("Cola <strong>liviana</strong> (acotada). Los extremos tienen "
            "un techo natural, lo cual limita las pérdidas máximas posibles.")


def _return_level_years(t_days: int) -> str:
    """Convierte un periodo de retorno en días a una descripción en años."""
    years = t_days / 252  # días hábiles por año
    if years >= 2:
        return f"aproximadamente {years:.0f} años"
    if years >= 1:
        return f"aproximadamente {years:.1f} año(s)"
    return f"{t_days} días hábiles"


def _return_level_years_blocks(t_blocks: int, block_size: int) -> str:
    """Convierte T bloques a una descripción en años."""
    total_days = t_blocks * block_size
    years = total_days / 252
    if years >= 2:
        return f"aproximadamente {years:.0f} años"
    if years >= 1:
        return f"aproximadamente {years:.1f} año(s)"
    return f"{total_days} días hábiles"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """\
:root {
    --bg: #ffffff;
    --bg-alt: #f7f8fa;
    --text: #1e293b;
    --text-muted: #64748b;
    --accent: #2563eb;
    --accent-light: #dbeafe;
    --border: #e2e8f0;
    --warn-bg: #fef3c7;
    --warn-border: #f59e0b;
    --green: #059669;
    --red: #dc2626;
    --radius: 8px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    color: var(--text);
    background: var(--bg);
    line-height: 1.6;
    max-width: 960px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
}

h1 {
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

h2 {
    font-size: 1.35rem;
    font-weight: 600;
    margin-top: 2.5rem;
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid var(--accent);
    color: var(--accent);
}

h3 {
    font-size: 1.05rem;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}

p, li { margin-bottom: 0.5rem; }

.subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-bottom: 2rem;
}

/* Tarjetas de métricas */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1rem;
    margin: 1rem 0 1.5rem;
}

.metric-card {
    background: var(--bg-alt);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
}

.metric-card .label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 0.25rem;
}

.metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent);
}

.metric-card .explain {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
    line-height: 1.5;
}

/* Figuras */
.fig-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.fig-card {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--bg);
}

.fig-card img {
    width: 100%;
    height: auto;
    display: block;
}

.fig-card .caption {
    padding: 0.5rem 0.75rem;
    font-size: 0.82rem;
    color: var(--text-muted);
    background: var(--bg-alt);
}

/* Alertas / diagnósticos */
.alert-box {
    background: var(--warn-bg);
    border-left: 4px solid var(--warn-border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
}

.alert-box ul { padding-left: 1.2rem; }

/* Interpretación */
.interp {
    background: var(--accent-light);
    border-radius: var(--radius);
    padding: 0.9rem 1.2rem;
    margin: 0.75rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
}

.interp strong { color: var(--accent); }

/* Collapsible técnico */
details {
    margin-top: 2.5rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
}

details summary {
    background: var(--bg-alt);
    padding: 0.8rem 1.2rem;
    font-weight: 600;
    cursor: pointer;
    user-select: none;
    font-size: 0.95rem;
}

details summary:hover { background: var(--border); }

details .tech-content {
    padding: 1.2rem;
}

.param-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    margin-top: 0.75rem;
}

.param-table th,
.param-table td {
    text-align: left;
    padding: 0.45rem 0.75rem;
    border-bottom: 1px solid var(--border);
}

.param-table th {
    background: var(--bg-alt);
    font-weight: 600;
    color: var(--text-muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.param-table code {
    background: var(--bg-alt);
    padding: 0.1em 0.4em;
    border-radius: 3px;
    font-size: 0.9em;
}

footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    font-size: 0.8rem;
    color: var(--text-muted);
    text-align: center;
}

@media (max-width: 600px) {
    body { padding: 1rem; }
    .fig-grid { grid-template-columns: 1fr; }
    .metric-grid { grid-template-columns: 1fr; }
}
"""


# ---------------------------------------------------------------------------
# Generación de HTML
# ---------------------------------------------------------------------------

def generate_report(
    results: Dict[str, Any],
    output_path: str = "evt_report.html",
    title: str = "Reporte de Riesgo Extremo (EVT)",
) -> str:
    """
    Genera un reporte HTML standalone a partir de los resultados de EVT.

    Parameters
    ----------
    results : dict
        Resultado de ``evt_engine.run_evt_analysis()``.
    output_path : str
        Ruta donde guardar el archivo HTML.
    title : str
        Título del reporte.

    Returns
    -------
    str
        La ruta del archivo generado.
    """
    meta = results["metadata"]
    gev = results["gev"]
    gpd = results["gpd"]
    diag = results["diagnostics"]
    cfg = meta["config_used"]

    p_conf_pct = f"{gev['p_conf'] * 100:.1f}"
    block_size = cfg["block_size"]

    # Encode all figures
    gev_imgs = {k: _fig_to_base64(v) for k, v in gev["figures"].items()}
    gpd_imgs = {k: _fig_to_base64(v) for k, v in gpd["figures"].items()}

    # Return level descriptions
    rl_gev_desc = _return_level_years_blocks(gev["t_blocks"], block_size)
    rl_gpd_desc = _return_level_years(gpd["t_days"])

    # Build HTML sections
    parts = []
    parts.append(_html_head(title))
    parts.append(_section_header(title, meta))
    parts.append(_section_diagnostics(diag))
    parts.append(_section_summary(gev, gpd, p_conf_pct))
    parts.append(_section_gev(gev, gev_imgs, p_conf_pct, rl_gev_desc, block_size))
    q_thresh_pct = _pct(cfg["quantile_threshold"])
    parts.append(_section_gpd(gpd, gpd_imgs, p_conf_pct, rl_gpd_desc, q_thresh_pct))
    parts.append(_section_technical(gev, gpd, meta, cfg))
    parts.append(_html_footer())

    html = "\n".join(parts)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


# ---------------------------------------------------------------------------
# Secciones HTML individuales
# ---------------------------------------------------------------------------

def _html_head(title: str) -> str:
    return f"""\
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{_CSS}
</style>
</head>
<body>"""


def _section_header(title: str, meta: Dict[str, Any]) -> str:
    source = meta["source"]
    n = meta["n_observations"]
    start = meta["start"][:10] if meta["start"] else "?"
    end = meta["end"][:10] if meta["end"] else "?"
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    return f"""\
<h1>{title}</h1>
<p class="subtitle">
    Fuente: <strong>{source}</strong> &nbsp;|&nbsp;
    {n} observaciones &nbsp;|&nbsp;
    {start} &rarr; {end} &nbsp;|&nbsp;
    Generado: {now}
</p>"""


def _section_diagnostics(diag: list[str]) -> str:
    if not diag:
        return ""
    items = "".join(f"<li>{d}</li>" for d in diag)
    return f"""\
<div class="alert-box">
    <strong>Advertencias del modelo</strong>
    <ul>{items}</ul>
</div>"""


def _section_summary(
    gev: Dict[str, Any],
    gpd: Dict[str, Any],
    p_conf_pct: str,
) -> str:
    return f"""\
<h2>Resumen ejecutivo</h2>
<div class="metric-grid">
    <div class="metric-card">
        <div class="label">VaR {p_conf_pct}% (GPD diario)</div>
        <div class="value">{_pct(gpd['var'])}</div>
        <div class="explain">
            En el {p_conf_pct}% de los dias, la perdida diaria
            no supero este valor.
        </div>
    </div>
    <div class="metric-card">
        <div class="label">ES {p_conf_pct}% (GPD diario)</div>
        <div class="value">{_pct(gpd['es'])}</div>
        <div class="explain">
            En los peores dias (cuando se supera el VaR),
            la perdida promedio fue de este nivel.
        </div>
    </div>
    <div class="metric-card">
        <div class="label">Nivel de retorno POT ({gpd['t_days']} dias)</div>
        <div class="value">{_pct(gpd['return_level'])}</div>
        <div class="explain">
            Una perdida de este tamano ocurre aproximadamente
            una vez cada {_return_level_years(gpd['t_days'])}.
        </div>
    </div>
</div>"""


def _section_gev(
    gev: Dict[str, Any],
    imgs: Dict[str, str],
    p_conf_pct: str,
    rl_desc: str,
    block_size: int,
) -> str:
    xi = gev["params"]["xi"]
    xi_interp = _xi_interpretation(xi)

    return f"""\
<h2>Block Maxima (GEV)</h2>
<p>
    Este metodo divide la serie de perdidas diarias en bloques de
    <strong>{block_size} dias</strong> y analiza el maximo de cada bloque.
    Esto permite estimar que tan grandes pueden ser las peores perdidas
    en periodos equivalentes.
</p>

<div class="metric-grid">
    <div class="metric-card">
        <div class="label">VaR {p_conf_pct}% (max. de bloque)</div>
        <div class="value">{_pct(gev['var'])}</div>
        <div class="explain">
            En el {p_conf_pct}% de los bloques de {block_size} dias,
            la peor perdida diaria no supero este nivel.
        </div>
    </div>
    <div class="metric-card">
        <div class="label">ES {p_conf_pct}% (max. de bloque)</div>
        <div class="value">{_pct(gev['es'])}</div>
        <div class="explain">
            En los bloques mas extremos (cuando se supera el VaR),
            la peor perdida diaria promedio fue de este nivel.
        </div>
    </div>
    <div class="metric-card">
        <div class="label">Nivel de retorno (T={gev['t_blocks']} bloques)</div>
        <div class="value">{_pct(gev['return_level'])}</div>
        <div class="explain">
            Una perdida diaria maxima de este tamano dentro de un bloque
            ocurre aproximadamente una vez cada {rl_desc}.
        </div>
    </div>
</div>

<h3>Forma de la cola (parametro de forma)</h3>
<div class="interp">
    <strong>&xi; = {_f4(xi)}</strong> &mdash; {xi_interp}
</div>

<h3>Graficos del ajuste GEV</h3>
<div class="fig-grid">
    <div class="fig-card">
        <img src="data:image/png;base64,{imgs['fit']}" alt="GEV fit">
        <div class="caption">
            Densidad GEV ajustada vs histograma de maximos por bloque.
            Si la curva sigue de cerca al histograma, el modelo es adecuado.
        </div>
    </div>
    <div class="fig-card">
        <img src="data:image/png;base64,{imgs['qq']}" alt="GEV QQ">
        <div class="caption">
            QQ-plot: los puntos cerca de la diagonal indican buen ajuste.
            Desviaciones en los extremos sugieren que la cola real
            difiere del modelo.
        </div>
    </div>
    <div class="fig-card">
        <img src="data:image/png;base64,{imgs['block_structure']}" alt="Bloques">
        <div class="caption">
            Estructura de bloques. Cada color es un bloque de {block_size} dias;
            los puntos destacados son los maximos seleccionados.
        </div>
    </div>
    <div class="fig-card">
        <img src="data:image/png;base64,{imgs['histogram']}" alt="Hist maximos">
        <div class="caption">
            Distribucion de los maximos por bloque.
        </div>
    </div>
</div>"""


def _section_gpd(
    gpd: Dict[str, Any],
    imgs: Dict[str, str],
    p_conf_pct: str,
    rl_desc: str,
    q_thresh_pct: str,
) -> str:
    xi = gpd["params"]["xi"]
    xi_interp = _xi_interpretation(xi)
    u_pct = _pct(gpd["threshold"])

    return f"""\
<h2>Peaks Over Threshold (GPD)</h2>
<p>
    Este metodo analiza solamente las perdidas que superan un umbral alto
    (el percentil {q_thresh_pct} de las perdidas = <strong>{u_pct}</strong>).
    Se enfoca directamente en los eventos extremos, sin depender de la
    estructura de bloques.
</p>

<div class="metric-grid">
    <div class="metric-card">
        <div class="label">Umbral</div>
        <div class="value">{u_pct}</div>
        <div class="explain">
            Perdidas por encima de este valor se consideran "extremas".
            Hubo <strong>{gpd['n_excesos']}</strong> excesos
            en la muestra.
        </div>
    </div>
    <div class="metric-card">
        <div class="label">VaR {p_conf_pct}% (diario)</div>
        <div class="value">{_pct(gpd['var'])}</div>
        <div class="explain">
            En el {p_conf_pct}% de los dias, la perdida
            no supero este valor.
        </div>
    </div>
    <div class="metric-card">
        <div class="label">ES {p_conf_pct}% (diario)</div>
        <div class="value">{_pct(gpd['es'])}</div>
        <div class="explain">
            En los peores dias (cuando se supera el VaR),
            la perdida promedio fue de este nivel.
            Es la medida de riesgo mas conservadora.
        </div>
    </div>
    <div class="metric-card">
        <div class="label">Nivel de retorno ({gpd['t_days']} dias)</div>
        <div class="value">{_pct(gpd['return_level'])}</div>
        <div class="explain">
            Una perdida de este tamano ocurre aproximadamente
            una vez cada {rl_desc}.
        </div>
    </div>
</div>

<h3>Forma de la cola (parametro de forma)</h3>
<div class="interp">
    <strong>&xi; = {_f4(xi)}</strong> &mdash; {xi_interp}
</div>

<h3>Graficos del ajuste GPD</h3>
<div class="fig-grid">
    <div class="fig-card">
        <img src="data:image/png;base64,{imgs['fit']}" alt="GPD fit">
        <div class="caption">
            Densidad GPD ajustada vs histograma de excesos sobre el umbral.
        </div>
    </div>
    <div class="fig-card">
        <img src="data:image/png;base64,{imgs['qq']}" alt="GPD QQ">
        <div class="caption">
            QQ-plot GPD: los puntos sobre la diagonal indican buen ajuste.
        </div>
    </div>
    <div class="fig-card">
        <img src="data:image/png;base64,{imgs['pot_points']}" alt="POT points">
        <div class="caption">
            Serie completa de perdidas. Los puntos destacados estan por
            encima del umbral y son los que alimentan el modelo GPD.
        </div>
    </div>
    <div class="fig-card">
        <img src="data:image/png;base64,{imgs['histogram']}" alt="Hist excesos">
        <div class="caption">
            Distribucion de los excesos (perdida menos umbral).
        </div>
    </div>
</div>"""


def _section_technical(
    gev: Dict[str, Any],
    gpd: Dict[str, Any],
    meta: Dict[str, Any],
    cfg: Dict[str, Any],
) -> str:
    return f"""\
<details>
<summary>Parametros tecnicos completos (click para expandir)</summary>
<div class="tech-content">

<h3>Configuracion utilizada</h3>
<table class="param-table">
    <tr><th>Parametro</th><th>Valor</th></tr>
    <tr><td><code>block_size</code></td><td>{cfg['block_size']}</td></tr>
    <tr><td><code>quantile_threshold</code></td><td>{cfg['quantile_threshold']}</td></tr>
    <tr><td><code>p_conf</code></td><td>{cfg['p_conf']}</td></tr>
    <tr><td><code>t_blocks</code></td><td>{cfg['t_blocks']}</td></tr>
    <tr><td><code>t_days</code></td><td>{cfg['t_days']}</td></tr>
</table>

<h3>GEV (Generalized Extreme Value)</h3>
<table class="param-table">
    <tr><th>Parametro</th><th>Valor</th><th>Descripcion</th></tr>
    <tr>
        <td><code>&xi;</code> (forma)</td>
        <td>{_f6(gev['params']['xi'])}</td>
        <td>Controla el peso de la cola. &gt;0 = Frechet, =0 = Gumbel, &lt;0 = Weibull</td>
    </tr>
    <tr>
        <td><code>&mu;</code> (locacion)</td>
        <td>{_f6(gev['params']['mu'])}</td>
        <td>Centro de la distribucion de maximos por bloque</td>
    </tr>
    <tr>
        <td><code>&sigma;</code> (escala)</td>
        <td>{_f6(gev['params']['sigma'])}</td>
        <td>Dispersion de la distribucion de maximos</td>
    </tr>
    <tr>
        <td>N bloques</td>
        <td>{gev['n_blocks']}</td>
        <td>Cantidad de bloques usados en el ajuste</td>
    </tr>
    <tr>
        <td>VaR {gev['p_conf']*100:.1f}%</td>
        <td>{_f6(gev['var'])}</td>
        <td>Value at Risk del maximo de bloque</td>
    </tr>
    <tr>
        <td>ES {gev['p_conf']*100:.1f}%</td>
        <td>{_f6(gev['es'])}</td>
        <td>Expected Shortfall del maximo de bloque</td>
    </tr>
    <tr>
        <td>Nivel de retorno (T={gev['t_blocks']})</td>
        <td>{_f6(gev['return_level'])}</td>
        <td>Cuantil extremo para T={gev['t_blocks']} bloques</td>
    </tr>
</table>

<h3>GPD (Generalized Pareto Distribution)</h3>
<table class="param-table">
    <tr><th>Parametro</th><th>Valor</th><th>Descripcion</th></tr>
    <tr>
        <td><code>&xi;</code> (forma)</td>
        <td>{_f6(gpd['params']['xi'])}</td>
        <td>Peso de la cola de los excesos</td>
    </tr>
    <tr>
        <td><code>&beta;</code> (escala)</td>
        <td>{_f6(gpd['params']['beta'])}</td>
        <td>Dispersion de los excesos sobre el umbral</td>
    </tr>
    <tr>
        <td>Umbral <code>u</code></td>
        <td>{_f6(gpd['threshold'])}</td>
        <td>Cuantil {cfg['quantile_threshold']*100:.0f}% de las perdidas</td>
    </tr>
    <tr>
        <td>N excesos</td>
        <td>{gpd['n_excesos']}</td>
        <td>Observaciones por encima del umbral</td>
    </tr>
    <tr>
        <td>P(X &gt; u)</td>
        <td>{_f6(gpd['p_u'])}</td>
        <td>Probabilidad empirica de exceder el umbral</td>
    </tr>
    <tr>
        <td>VaR {gpd['p_conf']*100:.1f}%</td>
        <td>{_f6(gpd['var'])}</td>
        <td>Value at Risk diario via POT</td>
    </tr>
    <tr>
        <td>ES {gpd['p_conf']*100:.1f}%</td>
        <td>{_f6(gpd['es'])}</td>
        <td>Expected Shortfall diario via POT</td>
    </tr>
    <tr>
        <td>Nivel de retorno (T={gpd['t_days']})</td>
        <td>{_f6(gpd['return_level'])}</td>
        <td>Cuantil extremo para T={gpd['t_days']} dias</td>
    </tr>
</table>

<h3>Datos de entrada</h3>
<table class="param-table">
    <tr><th>Campo</th><th>Valor</th></tr>
    <tr><td>Fuente</td><td>{meta['source']}</td></tr>
    <tr><td>Observaciones</td><td>{meta['n_observations']}</td></tr>
    <tr><td>Inicio</td><td>{meta['start']}</td></tr>
    <tr><td>Fin</td><td>{meta['end']}</td></tr>
</table>

</div>
</details>"""


def _html_footer() -> str:
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    return f"""\
<footer>
    Reporte generado automaticamente &mdash; {now}
    &nbsp;|&nbsp; Extreme Value Theory (EVT) Analysis
</footer>
</body>
</html>"""
