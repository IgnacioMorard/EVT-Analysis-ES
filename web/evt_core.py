"""
EVT core for Pyodide — self-contained module.

Combines the mathematical functions from Utilidades_GEV_BM_2.py and
evt_engine.py into a single file with no local imports, suitable for
running inside a browser via Pyodide.

Figures are returned as base64-encoded PNG strings (not Figure objects)
so they can be passed directly to JavaScript via the Pyodide bridge.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from scipy.stats import genextreme, genpareto
from scipy.integrate import quad


# ===================================================================
# DEFAULTS
# ===================================================================

DEFAULTS: Dict[str, Any] = {
    "block_size": 50,
    "quantile_threshold": 0.95,
    "p_conf": 0.995,
    "t_blocks": 100,
    "t_days": 5000,
}


# ===================================================================
# MATH — Block Maxima / GEV
# ===================================================================

def compute_block_maxima(
    series: pd.Series, block_size: int
) -> Tuple[pd.Series, int, pd.Series]:
    n = len(series)
    n_blocks = n // block_size
    trimmed = series.iloc[: n_blocks * block_size]
    blocks = trimmed.values.reshape(n_blocks, block_size)
    maxima = blocks.max(axis=1)
    return pd.Series(maxima, name="BlockMax"), n_blocks, trimmed


def fit_gev(block_maxima: pd.Series) -> Dict[str, float]:
    c, loc, scale = genextreme.fit(block_maxima)
    return {
        "c_hat": float(c),
        "xi_hat": float(-c),
        "loc_hat": float(loc),
        "scale_hat": float(scale),
    }


def gev_return_level(T: float, c: float, loc: float, scale: float) -> float:
    return float(genextreme.ppf(1 - 1 / T, c, loc=loc, scale=scale))


def gev_var_block(p: float, c: float, loc: float, scale: float) -> float:
    return float(genextreme.ppf(p, c, loc=loc, scale=scale))


def gev_var_es_block(
    p: float, c: float, loc: float, scale: float
) -> Tuple[float, float]:
    xi = -c
    var_p = float(genextreme.ppf(p, c, loc=loc, scale=scale))
    if xi >= 1.0:
        return var_p, float("inf")
    if abs(xi) < 1e-6:
        integrand = lambda u: genextreme.ppf(u, c, loc=loc, scale=scale)
        integral, _ = quad(integrand, p, 1.0 - 1e-10, limit=500)
        return var_p, float(integral / (1.0 - p))
    es = var_p / (1.0 - xi) + (scale - xi * loc) / (1.0 - xi)
    return var_p, float(es)


# ===================================================================
# MATH — Peaks Over Threshold / GPD
# ===================================================================

def compute_exceedances(series: pd.Series, threshold: float) -> pd.Series:
    exc = series[series > threshold] - threshold
    return exc.rename("Excess")


def fit_gpd(exceedances: pd.Series) -> Dict[str, float]:
    c, loc, scale = genpareto.fit(exceedances, floc=0)
    return {"xi_gpd": float(c), "beta_gpd": float(scale), "loc_gpd": float(loc)}


def gpd_return_level(
    T: float, u: float, xi: float, beta: float, p_u: float
) -> float:
    if abs(xi) < 1e-6:
        return float(u + beta * np.log(T * p_u))
    return float(u + (beta / xi) * ((T * p_u) ** xi - 1))


def gpd_var(p: float, u: float, xi: float, beta: float, p_u: float) -> float:
    q = 1.0 - p
    if abs(xi) < 1e-6:
        return float(u + beta * np.log(p_u / q))
    return float(u + (beta / xi) * ((p_u / q) ** xi - 1.0))


def gpd_es(p: float, u: float, xi: float, beta: float, p_u: float) -> float:
    v = gpd_var(p, u, xi, beta, p_u)
    if xi >= 1.0:
        return float("inf")
    return float(v / (1.0 - xi) + (beta - xi * u) / (1.0 - xi))


# ===================================================================
# FIGURES — return base64 PNG strings
# ===================================================================

def _to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _fig_gev_fit(bm: pd.Series, c: float, loc: float, sc: float) -> str:
    x = np.linspace(bm.min(), bm.max(), 200)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(bm, bins=20, density=True, alpha=0.6, label="Datos")
    ax.plot(x, genextreme.pdf(x, c, loc=loc, scale=sc), lw=2, label="GEV")
    ax.set_title("GEV vs histograma de maximos por bloque")
    ax.set_xlabel("Maximo de bloque"); ax.set_ylabel("Densidad")
    ax.legend(); ax.grid(True)
    return _to_b64(fig)


def _fig_gev_qq(bm: pd.Series, c: float, loc: float, sc: float) -> str:
    n = len(bm)
    prob = (np.arange(1, n + 1) - 0.5) / n
    emp = np.sort(bm); theo = genextreme.ppf(prob, c, loc=loc, scale=sc)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(theo, emp, s=15)
    lo, hi = min(theo.min(), emp.min()), max(theo.max(), emp.max())
    ax.plot([lo, hi], [lo, hi], "--")
    ax.set_title("QQ-plot: datos vs GEV"); ax.set_xlabel("Teoricos GEV")
    ax.set_ylabel("Empiricos"); ax.grid(True)
    return _to_b64(fig)


def _fig_gpd_fit(exc: pd.Series, xi: float, beta: float) -> str:
    y = np.linspace(0, exc.max(), 200)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(exc, bins=20, density=True, alpha=0.6, label="Datos")
    ax.plot(y, genpareto.pdf(y, xi, loc=0, scale=beta), lw=2, label="GPD")
    ax.set_title("GPD vs histograma de excesos (POT)")
    ax.set_xlabel("Exceso y = X - u"); ax.set_ylabel("Densidad")
    ax.legend(); ax.grid(True)
    return _to_b64(fig)


def _fig_gpd_qq(exc: pd.Series, xi: float, beta: float) -> str:
    n = len(exc)
    prob = (np.arange(1, n + 1) - 0.5) / n
    emp = np.sort(exc); theo = genpareto.ppf(prob, xi, loc=0, scale=beta)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(theo, emp, s=15)
    lo, hi = min(theo.min(), emp.min()), max(theo.max(), emp.max())
    ax.plot([lo, hi], [lo, hi], "--")
    ax.set_title("QQ-plot: datos vs GPD"); ax.set_xlabel("Teoricos GPD")
    ax.set_ylabel("Empiricos"); ax.grid(True)
    return _to_b64(fig)


def _fig_pot_points(series: pd.Series, threshold: float) -> str:
    x = np.arange(len(series)); y = series.values; m = y > threshold
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(x[~m], y[~m], s=8, alpha=0.5, label="<= umbral")
    ax.scatter(x[m], y[m], s=18, alpha=0.9, label="> umbral")
    ax.axhline(threshold, ls="--", lw=1.5, label=f"u = {threshold:.4f}")
    ax.set_title("Serie con umbral POT"); ax.set_xlabel("Dia")
    ax.set_ylabel("Perdida"); ax.legend(); ax.grid(True)
    return _to_b64(fig)


def _fig_losses_hist(losses: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(losses, bins=50, density=True, alpha=0.6)
    ax.set_title("Distribucion de perdidas diarias")
    ax.set_xlabel("Perdida L = -R"); ax.set_ylabel("Densidad"); ax.grid(True)
    return _to_b64(fig)


# ===================================================================
# MAIN ANALYSIS FUNCTION
# ===================================================================

def run_analysis(csv_text: str, config: Optional[Dict[str, Any]] = None, source_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Entry point called from JavaScript.

    Parameters
    ----------
    csv_text : str
        Raw CSV text (either from file upload or fetched from Yahoo).
        Must have columns interpretable as date + close price.
    config : dict | None
        User overrides for analysis parameters.

    Returns
    -------
    dict with keys: metadata, gev, gpd, diagnostics.
        All figures are base64 PNG strings (not matplotlib objects).
    """
    cfg = {**DEFAULTS, **(config or {})}
    block_size = int(cfg["block_size"])
    q_thresh = float(cfg["quantile_threshold"])
    p_conf = float(cfg["p_conf"])
    t_blocks = int(cfg["t_blocks"])
    t_days = int(cfg["t_days"])

    diagnostics: List[str] = []

    # --- Parse CSV ---
    df = pd.read_csv(io.StringIO(csv_text))

    # Detect date column
    date_col = None
    for c in df.columns:
        low = c.strip().lower()
        if low in ("date", "fecha", "datetime", "timestamp", "time", "day"):
            date_col = c
            break
    if date_col is None:
        # Try first column
        date_col = df.columns[0]

    # Detect close column
    close_col = None
    for c in df.columns:
        low = c.strip().lower().replace(" ", "")
        if low in ("close", "cierre", "adjclose", "adj_close", "precio",
                    "price", "ultimo", "last", "settle"):
            close_col = c
            break
    if close_col is None:
        # Fallback: last numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            close_col = num_cols[-1]
        else:
            raise ValueError("No se encontro columna de cierre en el CSV.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    prices = df.set_index(date_col)[close_col].dropna()
    prices = prices[prices > 0]

    if len(prices) < 10:
        raise ValueError(f"Solo {len(prices)} precios validos. Se necesitan mas datos.")

    returns = np.log(prices / prices.shift(1)).dropna()
    returns.name = "R"
    losses = -returns
    losses.name = "L"

    n_obs = len(returns)
    if n_obs < 200:
        diagnostics.append(
            f"Serie corta: {n_obs} observaciones (se recomiendan >= 200)."
        )

    source_name = source_name or (str(prices.name) if prices.name else "CSV")
    start_dt = str(returns.index.min())[:10]
    end_dt = str(returns.index.max())[:10]

    # === GEV ===
    bm, n_blocks, trimmed = compute_block_maxima(losses, block_size)
    if n_blocks < 5:
        raise ValueError(
            f"Solo {n_blocks} bloques con block_size={block_size}. "
            "Se necesitan al menos 5. Reduce block_size o usa mas datos."
        )
    if n_blocks < 10:
        diagnostics.append(
            f"GEV: solo {n_blocks} bloques (se recomiendan >= 10)."
        )

    gev_raw = fit_gev(bm)
    c_hat = gev_raw["c_hat"]
    xi_gev = gev_raw["xi_hat"]
    mu_gev = gev_raw["loc_hat"]
    sigma_gev = gev_raw["scale_hat"]

    if xi_gev >= 1.0:
        diagnostics.append(f"GEV: xi={xi_gev:.4f} >= 1. ES será infinito.")
    elif xi_gev >= 0.5:
        diagnostics.append(f"GEV: xi={xi_gev:.4f} alto (>= 0.5). Los estimadores pueden ser inestables.")

    rl_gev = gev_return_level(t_blocks, c_hat, mu_gev, sigma_gev)
    var_gev, es_gev = gev_var_es_block(p_conf, c_hat, mu_gev, sigma_gev)

    gev_figs = {
        "fit": _fig_gev_fit(bm, c_hat, mu_gev, sigma_gev),
        "qq": _fig_gev_qq(bm, c_hat, mu_gev, sigma_gev),
    }

    # === GPD ===
    u = float(losses.quantile(q_thresh))
    exc = compute_exceedances(losses, u)
    n_exc = len(exc)
    p_u = float((losses > u).mean())

    if n_exc < 10:
        raise ValueError(
            f"Solo {n_exc} excesos con umbral al percentil {q_thresh}. "
            "Bajá el umbral o usá más datos."
        )
    if n_exc < 30:
        diagnostics.append(
            f"GPD: solo {n_exc} excesos (se recomiendan >= 30)."
        )

    gpd_raw = fit_gpd(exc)
    xi_gpd = gpd_raw["xi_gpd"]
    beta_gpd = gpd_raw["beta_gpd"]

    if xi_gpd >= 1.0:
        diagnostics.append(f"GPD: xi={xi_gpd:.4f} >= 1. ES será infinito.")
    elif xi_gpd >= 0.5:
        diagnostics.append(f"GPD: xi={xi_gpd:.4f} alto (>= 0.5). Los estimadores pueden ser inestables.")

    rl_gpd = gpd_return_level(t_days, u, xi_gpd, beta_gpd, p_u)
    var_gpd_val = gpd_var(p_conf, u, xi_gpd, beta_gpd, p_u)
    es_gpd_val = gpd_es(p_conf, u, xi_gpd, beta_gpd, p_u)

    gpd_figs = {
        "fit": _fig_gpd_fit(exc, xi_gpd, beta_gpd),
        "qq": _fig_gpd_qq(exc, xi_gpd, beta_gpd),
        "pot_points": _fig_pot_points(losses, u),
    }

    # Extra figure
    losses_hist = _fig_losses_hist(losses)

    # === Return ===
    def _safe(v: float) -> Any:
        if math.isinf(v):
            return None  # JSON-safe
        return round(v, 8)

    return {
        "metadata": {
            "source": source_name,
            "n_observations": n_obs,
            "start": start_dt,
            "end": end_dt,
            "config_used": cfg,
        },
        "gev": {
            "params": {"xi": _safe(xi_gev), "mu": _safe(mu_gev), "sigma": _safe(sigma_gev)},
            "n_blocks": n_blocks,
            "var": _safe(var_gev),
            "es": _safe(es_gev),
            "return_level": _safe(rl_gev),
            "t_blocks": t_blocks,
            "p_conf": p_conf,
            "figures": gev_figs,
        },
        "gpd": {
            "params": {"xi": _safe(xi_gpd), "beta": _safe(beta_gpd)},
            "threshold": _safe(u),
            "n_excesos": n_exc,
            "p_u": _safe(p_u),
            "var": _safe(var_gpd_val),
            "es": _safe(es_gpd_val),
            "return_level": _safe(rl_gpd),
            "t_days": t_days,
            "p_conf": p_conf,
            "figures": gpd_figs,
        },
        "figures": {
            "losses_hist": losses_hist,
        },
        "diagnostics": diagnostics,
    }
