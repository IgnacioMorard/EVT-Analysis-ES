"""
Motor de análisis EVT completo.

Orquesta las funciones de ``Utilidades_GEV_BM_2`` y ``data_loader``
para ejecutar un análisis Block-Maxima (GEV) + Peaks-Over-Threshold (GPD)
en una sola llamada, devolviendo todos los resultados y figuras en un dict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import genextreme, genpareto

import config as default_cfg
from Utilidades_GEV_BM_2 import (
    compute_block_maxima,
    compute_exceedances,
    fit_gev,
    fit_gpd,
    gev_return_level,
    gev_var_block,
    gev_var_es_block,
    gev_prob_exceed,
    gpd_return_level,
    gpd_prob_exceed,
)

# ---------------------------------------------------------------------------
# Configuración por defecto (extraída de config.py)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "block_size": default_cfg.BLOCK_SIZE,
    "quantile_threshold": default_cfg.QUANTILE_THRESHOLD,
    "p_conf": default_cfg.P_CONF,
    "t_blocks": default_cfg.T_BLOCKS,
    "t_days": default_cfg.T_DAYS,
}


def _resolve_config(user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Mezcla defaults con overrides del usuario."""
    merged = _DEFAULTS.copy()
    if user_config:
        merged.update(user_config)
    return merged


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  FIGURAS (generan y cierran, devuelven Figure)                        ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def _fig_histogram(
    series: pd.Series,
    bins: int,
    title: str,
    xlabel: str,
) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series, bins=bins, density=True, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densidad")
    ax.grid(True)
    plt.close(fig)
    return fig


def _fig_block_structure(
    trimmed: pd.Series,
    block_size: int,
    title: str,
) -> Figure:
    n = len(trimmed)
    n_blocks = n // block_size
    x_idx = np.arange(n)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(n_blocks):
        s = i * block_size
        e = s + block_size
        xb = x_idx[s:e]
        yb = trimmed.iloc[s:e]
        ax.plot(xb, yb, marker="o", linewidth=0.8, alpha=0.6)
        loc_max = yb.values.argmax()
        ax.scatter(xb[loc_max], yb.iloc[loc_max], s=50, edgecolor="k")

    ax.set_title(title)
    ax.set_xlabel("Índice temporal (días)")
    ax.set_ylabel("Valor")
    ax.grid(True)
    plt.close(fig)
    return fig


def _fig_gev_fit(
    block_maxima: pd.Series,
    c: float,
    loc: float,
    scale: float,
) -> Figure:
    x_grid = np.linspace(block_maxima.min(), block_maxima.max(), 200)
    pdf = genextreme.pdf(x_grid, c, loc=loc, scale=scale)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(block_maxima, bins=20, density=True, alpha=0.6, label="Datos")
    ax.plot(x_grid, pdf, linewidth=2, label="GEV ajustada")
    ax.set_title("GEV vs histograma de máximos por bloque")
    ax.set_xlabel("Máximo de bloque")
    ax.set_ylabel("Densidad")
    ax.legend()
    ax.grid(True)
    plt.close(fig)
    return fig


def _fig_gev_qq(
    block_maxima: pd.Series,
    c: float,
    loc: float,
    scale: float,
) -> Figure:
    n = len(block_maxima)
    prob = (np.arange(1, n + 1) - 0.5) / n
    empirical = np.sort(block_maxima)
    theoretical = genextreme.ppf(prob, c, loc=loc, scale=scale)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(theoretical, empirical, s=15)
    lo = min(theoretical.min(), empirical.min())
    hi = max(theoretical.max(), empirical.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_title("QQ-plot: datos vs GEV ajustada")
    ax.set_xlabel("Cuantiles teóricos GEV")
    ax.set_ylabel("Cuantiles empíricos")
    ax.grid(True)
    plt.close(fig)
    return fig


def _fig_gpd_fit(
    exceedances: pd.Series,
    xi: float,
    beta: float,
) -> Figure:
    y_grid = np.linspace(0, exceedances.max(), 200)
    pdf = genpareto.pdf(y_grid, xi, loc=0, scale=beta)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(exceedances, bins=20, density=True, alpha=0.6, label="Datos")
    ax.plot(y_grid, pdf, linewidth=2, label="GPD ajustada")
    ax.set_title("GPD vs histograma de excesos (POT)")
    ax.set_xlabel("Exceso y = X - u")
    ax.set_ylabel("Densidad")
    ax.legend()
    ax.grid(True)
    plt.close(fig)
    return fig


def _fig_pot_points(
    series: pd.Series,
    threshold: float,
) -> Figure:
    x_all = np.arange(len(series))
    y_all = series.values
    mask = y_all > threshold

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_all[~mask], y_all[~mask], s=10, alpha=0.6, label="≤ umbral")
    ax.scatter(x_all[mask], y_all[mask], s=20, alpha=0.9, label="> umbral")
    ax.axhline(
        threshold, linestyle="--", linewidth=1.5,
        label=f"Umbral u = {threshold:.4f}",
    )
    ax.set_title("POT: serie con umbral y excesos destacados")
    ax.set_xlabel("Índice temporal (días)")
    ax.set_ylabel("Valor")
    ax.legend()
    ax.grid(True)
    plt.close(fig)
    return fig


def _fig_gpd_qq(
    exceedances: pd.Series,
    xi: float,
    beta: float,
) -> Figure:
    n = len(exceedances)
    prob = (np.arange(1, n + 1) - 0.5) / n
    empirical = np.sort(exceedances)
    theoretical = genpareto.ppf(prob, xi, loc=0, scale=beta)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(theoretical, empirical, s=15)
    lo = min(theoretical.min(), empirical.min())
    hi = max(theoretical.max(), empirical.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_title("QQ-plot: datos vs GPD ajustada")
    ax.set_xlabel("Cuantiles teóricos GPD")
    ax.set_ylabel("Cuantiles empíricos")
    ax.grid(True)
    plt.close(fig)
    return fig


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  VaR / ES vía GPD (a nivel de la serie original, no solo excesos)     ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def _gpd_var(
    p: float,
    u: float,
    xi: float,
    beta: float,
    p_u: float,
) -> float:
    """
    VaR al nivel *p* usando la aproximación POT.

    Fórmula (McNeil, Frey & Embrechts):
        VaR_p = u + (β/ξ) · [(p_u / (1 - p))^ξ − 1]       ξ ≠ 0
        VaR_p = u + β · log(p_u / (1 - p))                  ξ ≈ 0
    """
    q = 1.0 - p  # probabilidad de excedencia
    if abs(xi) < 1e-6:
        return float(u + beta * np.log(p_u / q))
    return float(u + (beta / xi) * ((p_u / q) ** xi - 1.0))


def _gpd_es(
    p: float,
    u: float,
    xi: float,
    beta: float,
    p_u: float,
) -> float:
    """
    Expected Shortfall al nivel *p* usando POT.

    Fórmula (válida para ξ < 1):
        ES_p = VaR_p / (1 - ξ)  +  (β − ξ·u) / (1 − ξ)

    Para ξ ≥ 1 el ES no existe (cola de varianza infinita).
    """
    var_p = _gpd_var(p, u, xi, beta, p_u)
    if xi >= 1.0:
        return float("inf")
    return float(var_p / (1.0 - xi) + (beta - xi * u) / (1.0 - xi))


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  FUNCIÓN PRINCIPAL                                                     ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def run_evt_analysis(
    returns: pd.Series,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Ejecuta un análisis EVT completo (GEV + GPD) sobre log-returns.

    Parameters
    ----------
    returns : pd.Series
        Log-returns diarios.  Se trabaja con pérdidas ``L = -R``.
    config : dict | None
        Overrides sobre los defaults de ``config.py``.  Claves válidas:

        ============== ====== ============================================
        Clave          Tipo   Descripción
        ============== ====== ============================================
        block_size     int    Observaciones por bloque (GEV)
        quantile_threshold float  Cuantil para el umbral POT
        p_conf         float  Nivel de confianza para VaR / ES
        t_blocks       int    Periodo de retorno en bloques (GEV)
        t_days         int    Periodo de retorno en días (GPD)
        ============== ====== ============================================

    Returns
    -------
    dict
        Estructura::

            {
                "metadata": { ... },
                "gev":      { "params", "var", "es", "return_level",
                              "n_blocks", "figures" },
                "gpd":      { "params", "threshold", "n_excesos",
                              "var", "es", "return_level", "figures" },
                "diagnostics": [ lista de advertencias ],
            }

        Cada entrada en ``figures`` es un ``matplotlib.figure.Figure``
        ya cerrado (no se muestra).  Para visualizarlas después::

            results["gev"]["figures"]["fit"].show()
            # o guardarlas:
            results["gev"]["figures"]["fit"].savefig("gev_fit.png")
    """
    cfg = _resolve_config(config)
    block_size: int = cfg["block_size"]
    q_thresh: float = cfg["quantile_threshold"]
    p_conf: float = cfg["p_conf"]
    t_blocks: int = cfg["t_blocks"]
    t_days: int = cfg["t_days"]

    diagnostics: List[str] = []

    # --- Pérdidas ---
    losses = -returns
    losses.name = "L"

    # --- Metadata ---
    metadata: Dict[str, Any] = {
        "source": returns.name or "unknown",
        "n_observations": len(returns),
        "start": str(returns.index.min()) if len(returns) > 0 else None,
        "end": str(returns.index.max()) if len(returns) > 0 else None,
        "config_used": cfg,
    }

    # ================================================================
    # BLOCK MAXIMA  (GEV)
    # ================================================================
    block_maxima, n_blocks, trimmed = compute_block_maxima(losses, block_size)

    if n_blocks < 10:
        diagnostics.append(
            f"GEV: solo {n_blocks} bloques (se recomiendan >= 10). "
            "Considerá reducir block_size o usar más datos."
        )

    gev_raw = fit_gev(block_maxima)
    c_hat = gev_raw["c_hat"]
    xi_gev = gev_raw["xi_hat"]
    mu_gev = gev_raw["loc_hat"]
    sigma_gev = gev_raw["scale_hat"]

    if xi_gev >= 1.0:
        diagnostics.append(
            f"GEV: ξ = {xi_gev:.4f} ≥ 1 → la media del máximo de bloque "
            "no existe.  El ES será infinito."
        )
    elif xi_gev >= 0.5:
        diagnostics.append(
            f"GEV: ξ = {xi_gev:.4f} es alto (≥ 0.5).  "
            "Los estimadores pueden ser inestables."
        )

    ret_level_gev = gev_return_level(t_blocks, c_hat, mu_gev, sigma_gev)
    var_gev = gev_var_block(p_conf, c_hat, mu_gev, sigma_gev)
    var_gev_val, es_gev_val = gev_var_es_block(p_conf, c_hat, mu_gev, sigma_gev)

    # Figuras GEV
    gev_figures: Dict[str, Figure] = {
        "histogram": _fig_histogram(
            block_maxima, bins=20,
            title="Histograma de máximos por bloque",
            xlabel="Máximo del bloque",
        ),
        "block_structure": _fig_block_structure(
            trimmed, block_size,
            title=f"Block Maxima: bloques de {block_size}",
        ),
        "fit": _fig_gev_fit(block_maxima, c_hat, mu_gev, sigma_gev),
        "qq": _fig_gev_qq(block_maxima, c_hat, mu_gev, sigma_gev),
    }

    gev_result: Dict[str, Any] = {
        "params": {
            "xi": xi_gev,
            "mu": mu_gev,
            "sigma": sigma_gev,
        },
        "n_blocks": n_blocks,
        "var": var_gev_val,
        "es": es_gev_val,
        "return_level": ret_level_gev,
        "t_blocks": t_blocks,
        "p_conf": p_conf,
        "figures": gev_figures,
    }

    # ================================================================
    # PEAKS OVER THRESHOLD  (GPD)
    # ================================================================
    u = float(losses.quantile(q_thresh))
    exceedances = compute_exceedances(losses, u)
    n_excesos = len(exceedances)
    p_u = float((losses > u).mean())

    if n_excesos < 30:
        diagnostics.append(
            f"GPD: solo {n_excesos} excesos sobre el umbral "
            f"(se recomiendan >= 30).  Considerá bajar quantile_threshold."
        )

    gpd_raw = fit_gpd(exceedances)
    xi_gpd = gpd_raw["xi_gpd"]
    beta_gpd = gpd_raw["beta_gpd"]

    if xi_gpd >= 1.0:
        diagnostics.append(
            f"GPD: ξ = {xi_gpd:.4f} ≥ 1 → la media de los excesos "
            "no existe.  El ES será infinito."
        )
    elif xi_gpd >= 0.5:
        diagnostics.append(
            f"GPD: ξ = {xi_gpd:.4f} es alto (≥ 0.5).  "
            "Los estimadores pueden ser inestables."
        )

    ret_level_gpd = gpd_return_level(t_days, u, xi_gpd, beta_gpd, p_u)
    var_gpd = _gpd_var(p_conf, u, xi_gpd, beta_gpd, p_u)
    es_gpd = _gpd_es(p_conf, u, xi_gpd, beta_gpd, p_u)

    # Figuras GPD
    gpd_figures: Dict[str, Figure] = {
        "histogram": _fig_histogram(
            exceedances, bins=20,
            title="Histograma de excesos sobre el umbral",
            xlabel="Exceso y = X - u",
        ),
        "fit": _fig_gpd_fit(exceedances, xi_gpd, beta_gpd),
        "qq": _fig_gpd_qq(exceedances, xi_gpd, beta_gpd),
        "pot_points": _fig_pot_points(losses, u),
    }

    gpd_result: Dict[str, Any] = {
        "params": {
            "xi": xi_gpd,
            "beta": beta_gpd,
        },
        "threshold": u,
        "n_excesos": n_excesos,
        "p_u": p_u,
        "var": var_gpd,
        "es": es_gpd,
        "return_level": ret_level_gpd,
        "t_days": t_days,
        "p_conf": p_conf,
        "figures": gpd_figures,
    }

    # ================================================================
    # RESULTADO FINAL
    # ================================================================
    return {
        "metadata": metadata,
        "gev": gev_result,
        "gpd": gpd_result,
        "diagnostics": diagnostics,
    }
