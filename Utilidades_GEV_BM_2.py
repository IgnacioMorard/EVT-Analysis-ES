#!/usr/bin/env python
# coding: utf-8

"""
Teoría de Valores Extremos (EVT) en Python
=========================================

Este módulo implementa ejemplos de EVT usando:

1) Block Maxima (GEV)
2) Peaks Over Threshold (POT – GPD)

Incluye:
- Ejemplo con datos simulados de colas pesadas (t-Student).
- Ejemplo con datos reales de mercado (via CSV).
- Cálculo de niveles de retorno, VaR y ES para extremos.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, Dict, Optional

from scipy.stats import genextreme, genpareto, t

from config import BLOCK_SIZE, QUANTILE_THRESHOLD, P_CONF, T_BLOCKS, T_DAYS

# Opcionales: solo si usás descarga online
try:
    import yfinance as yf  # type: ignore
except ImportError:
    yf = None

# Config global de gráficos
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True

# =============================================================================
# UTILIDADES GENERALES
# =============================================================================

def simulate_t_series(
    n: int = 5000,
    df: int = 3,
    name: str = "X",
    random_state: int = 42
) -> pd.Series:
    """
    Simula una serie con colas pesadas usando una t-Student.

    Parameters
    ----------
    n : int
        Número de observaciones.
    df : int
        Grados de libertad (más chico => colas más pesadas).
    name : str
        Nombre de la serie resultante.
    random_state : int
        Semilla para reproducibilidad.

    Returns
    -------
    pd.Series
        Serie simulada con colas pesadas.
    """
    rng = np.random.default_rng(random_state)
    data = t.rvs(df, size=n, random_state=rng)
    return pd.Series(data, name=name)


def plot_histogram(
    series: pd.Series,
    bins: int = 50,
    title: str = "",
    xlabel: str = "",
    density: bool = True
) -> None:
    """Grafica un histograma sencillo de una serie."""
    fig, ax = plt.subplots()
    ax.hist(series, bins=bins, density=density, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densidad" if density else "Frecuencia")
    plt.show()


def qq_plot(
    theoretical: np.ndarray,
    empirical: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str
) -> None:
    """QQ-plot entre cuantiles teóricos y empíricos."""
    fig, ax = plt.subplots()
    ax.scatter(theoretical, empirical, s=15)
    min_val = min(theoretical.min(), empirical.min())
    max_val = max(theoretical.max(), empirical.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


# =============================================================================
# BLOCK MAXIMA (GEV)
# =============================================================================

def compute_block_maxima(
    series: pd.Series,
    block_size: int
) -> Tuple[pd.Series, int, pd.Series]:
    """
    Divide la serie en bloques de tamaño fijo y obtiene el máximo por bloque.

    Parameters
    ----------
    series : pd.Series
        Serie original (por ejemplo, pérdidas diarias).
    block_size : int
        Cantidad de observaciones por bloque.

    Returns
    -------
    block_maxima : pd.Series
        Serie de máximos por bloque.
    n_blocks : int
        Cantidad de bloques usados.
    trimmed_series : pd.Series
        Serie recortada para ser múltiplo exacto de block_size.
    """
    n = len(series)
    n_blocks = n // block_size
    trimmed = series.iloc[: n_blocks * block_size]
    blocks = trimmed.values.reshape(n_blocks, block_size)
    maxima = blocks.max(axis=1)
    block_maxima = pd.Series(maxima, name="BlockMax")
    return block_maxima, n_blocks, trimmed


def plot_block_structure(
    trimmed_series: pd.Series,
    block_size: int,
    title: str = "Block Maxima: bloques con sus máximos"
) -> None:
    """
    Muestra visualmente los bloques y marca el máximo en cada uno.
    """
    n = len(trimmed_series)
    n_blocks = n // block_size
    x_idx = np.arange(n)

    fig, ax = plt.subplots()

    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size

        x_block = x_idx[start:end]
        y_block = trimmed_series.iloc[start:end]

        ax.plot(x_block, y_block, marker="o", linewidth=0.8, alpha=0.6)

        local_argmax = y_block.values.argmax()
        x_max = x_block[local_argmax]
        y_max = y_block.iloc[local_argmax]
        ax.scatter(x_max, y_max, s=50, edgecolor="k")

    ax.set_title(title)
    ax.set_xlabel("Índice temporal (días)")
    ax.set_ylabel("Valor")
    plt.show()


def fit_gev(block_maxima: pd.Series) -> Dict[str, float]:
    """
    Ajusta una distribución GEV a los máximos de bloque mediante MLE.

    Returns
    -------
    dict con:
        c_hat   : parámetro de SciPy
        xi_hat  : parámetro de forma estándar EVT (xi = -c_hat)
        loc_hat : parámetro de locación (mu)
        scale_hat : parámetro de escala (sigma)
    """
    c_hat, loc_hat, scale_hat = genextreme.fit(block_maxima)
    xi_hat = -c_hat  # SciPy usa c = -xi
    params = {
        "c_hat": float(c_hat),
        "xi_hat": float(xi_hat),
        "loc_hat": float(loc_hat),
        "scale_hat": float(scale_hat),
    }
    return params


def plot_gev_fit(
    block_maxima: pd.Series,
    c_hat: float,
    loc_hat: float,
    scale_hat: float
) -> None:
    """Superpone la densidad GEV ajustada sobre el histograma de máximos."""
    x_grid = np.linspace(block_maxima.min(), block_maxima.max(), 200)
    gev_pdf = genextreme.pdf(x_grid, c_hat, loc=loc_hat, scale=scale_hat)

    fig, ax = plt.subplots()
    ax.hist(block_maxima, bins=20, density=True, alpha=0.6, label="Datos")
    ax.plot(x_grid, gev_pdf, linewidth=2, label="GEV ajustada")
    ax.set_title("GEV vs histograma de máximos por bloque")
    ax.set_xlabel("Máximo de bloque")
    ax.set_ylabel("Densidad")
    ax.legend()
    plt.show()


def plot_gev_qq(block_maxima: pd.Series, c_hat: float, loc_hat: float, scale_hat: float) -> None:
    """QQ-plot datos vs GEV ajustada."""
    prob = (np.arange(1, len(block_maxima) + 1) - 0.5) / len(block_maxima)
    block_sorted = np.sort(block_maxima)

    gev_theoretical = genextreme.ppf(prob, c_hat, loc=loc_hat, scale=scale_hat)

    qq_plot(
        theoretical=gev_theoretical,
        empirical=block_sorted,
        title="QQ-plot: datos vs GEV ajustada",
        xlabel="Cuantiles teóricos GEV",
        ylabel="Cuantiles empíricos"
    )


def gev_return_level(
    T_blocks: float,
    c_hat: float,
    loc_hat: float,
    scale_hat: float
) -> float:
    """
    Nivel de retorno para la distribución de máximos por bloque.

    T_blocks : periodo de retorno en número de bloques
    Devuelve x_T tal que P(Máximo del bloque > x_T) = 1 / T_blocks.
    """
    p = 1 - 1 / T_blocks
    x_T = genextreme.ppf(p, c_hat, loc=loc_hat, scale=scale_hat)
    return float(x_T)


def gev_var_block(
    p: float,
    c_hat: float,
    loc_hat: float,
    scale_hat: float
) -> float:
    """
    VaR del máximo de bloque al nivel de confianza p.
    P(Máximo ≤ x_p) = p.
    """
    x_p = genextreme.ppf(p, c_hat, loc=loc_hat, scale=scale_hat)
    return float(x_p)


def gev_prob_exceed(
    z: float,
    c_hat: float,
    loc_hat: float,
    scale_hat: float
) -> float:
    """
    Probabilidad de que el máximo de bloque exceda un nivel z.
    Devuelve P(Máximo > z).
    """
    F_z = genextreme.cdf(z, c_hat, loc=loc_hat, scale=scale_hat)
    return float(1 - F_z)


def gev_var_es_block(
    p: float,
    c_hat: float,
    loc_hat: float,
    scale_hat: float,
) -> Tuple[float, float]:
    """
    Calcula VaR y ES del máximo por bloque al nivel p de forma analítica.

    Usa la fórmula cerrada válida para ξ < 1 (c_hat > -1):

        ES_p = VaR_p / (1 - ξ) + (σ - ξ·μ) / (1 - ξ)

    donde ξ = -c_hat (convención SciPy: c = -ξ).

    Para |ξ| < 1e-6 (límite Gumbel) se usa integración numérica estable.
    La fórmula requiere ξ < 1; para ξ ≥ 1 el ES no existe (cola infinita).

    Referencia: McNeil, Frey & Embrechts – «Quantitative Risk Management» (2005).
    """
    from scipy.integrate import quad

    xi = -c_hat  # parámetro de forma EVT estándar
    var_p = float(genextreme.ppf(p, c_hat, loc=loc_hat, scale=scale_hat))

    if xi >= 1.0:
        return var_p, float("inf")

    if abs(xi) < 1e-6:
        # Límite Gumbel: integración numérica de alta precisión
        integrand = lambda u: genextreme.ppf(u, c_hat, loc=loc_hat, scale=scale_hat)
        integral, _ = quad(integrand, p, 1.0 - 1e-10, limit=500)
        es_p = float(integral / (1.0 - p))
    else:
        es_p = float(var_p / (1.0 - xi) + (scale_hat - xi * loc_hat) / (1.0 - xi))

    return var_p, es_p


# =============================================================================
# POT (GPD)
# =============================================================================

def compute_exceedances(
    series: pd.Series,
    threshold: float
) -> pd.Series:
    """
    Calcula excesos sobre un umbral: Y = X - u, para X > u.
    """
    exceedances = series[series > threshold] - threshold
    return exceedances.rename("Excess")


def fit_gpd(exceedances: pd.Series) -> Dict[str, float]:
    """
    Ajusta una distribución GPD a los excesos.

    Devuelve dict con:
        xi_gpd   : parámetro de forma
        beta_gpd : parámetro de escala
        loc_gpd  : parámetro de locación (normalmente 0 en POT)
    """
    c_gpd, loc_gpd, scale_gpd = genpareto.fit(exceedances, floc=0)
    xi_gpd = c_gpd
    beta_gpd = scale_gpd
    return {
        "xi_gpd": float(xi_gpd),
        "beta_gpd": float(beta_gpd),
        "loc_gpd": float(loc_gpd),
    }


def plot_gpd_fit(
    exceedances: pd.Series,
    xi_gpd: float,
    beta_gpd: float,
    loc_gpd: float = 0.0
) -> None:
    """Superpone la densidad GPD ajustada sobre el histograma de excesos."""
    y_grid = np.linspace(0, exceedances.max(), 200)
    gpd_pdf = genpareto.pdf(y_grid, xi_gpd, loc=loc_gpd, scale=beta_gpd)

    fig, ax = plt.subplots()
    ax.hist(exceedances, bins=20, density=True, alpha=0.6, label="Datos")
    ax.plot(y_grid, gpd_pdf, linewidth=2, label="GPD ajustada")
    ax.set_title("GPD vs histograma de excesos (POT)")
    ax.set_xlabel("Exceso y = X - u")
    ax.set_ylabel("Densidad")
    ax.legend()
    plt.show()


def plot_pot_points(
    series: pd.Series,
    threshold: float
) -> None:
    """
    Muestra la serie, marcando qué puntos están por encima del umbral.
    """
    x_all = np.arange(len(series))
    y_all = series.values
    mask = y_all > threshold

    fig, ax = plt.subplots()
    ax.scatter(x_all[~mask], y_all[~mask], s=10, alpha=0.6, label="≤ umbral")
    ax.scatter(x_all[mask], y_all[mask], s=20, alpha=0.9, label="> umbral")
    ax.axhline(threshold, linestyle="--", linewidth=1.5,
               label=f"Umbral u = {threshold:.4f}")
    ax.set_title("POT: serie con umbral y excesos destacados")
    ax.set_xlabel("Índice temporal (días)")
    ax.set_ylabel("Valor")
    ax.legend()
    plt.show()


def gpd_prob_exceed(
    z: float,
    u: float,
    xi_gpd: float,
    beta_gpd: float,
    p_u: float
) -> float:
    """
    Probabilidad P(X > z) usando POT, con z > u.

    p_u = P(X > u) estimada empíricamente.
    """
    if z <= u:
        raise ValueError("Se requiere z > u para POT.")
    y = z - u
    tail_cond = (1 + xi_gpd * y / beta_gpd) ** (-1 / xi_gpd)
    return float(p_u * tail_cond)


def gpd_return_level(
    T_days: float,
    u: float,
    xi_gpd: float,
    beta_gpd: float,
    p_u: float
) -> float:
    """
    Nivel de retorno z_T tal que P(X > z_T) ≈ 1 / T_days usando POT.

    Fórmula general (ξ ≠ 0):
        z_T = u + (beta / xi) * [(T * p_u)^xi - 1]

    Límite estable para ξ → 0 (distribución Exponencial):
        z_T = u + beta * log(T * p_u)

    Se usa el límite cuando |ξ| < 1e-6 para evitar inestabilidad numérica.
    """
    if abs(xi_gpd) < 1e-6:
        # Límite exponencial: expansión de primer orden en xi
        z_T = u + beta_gpd * np.log(T_days * p_u)
    else:
        z_T = u + (beta_gpd / xi_gpd) * ((T_days * p_u) ** xi_gpd - 1)
    return float(z_T)


# =============================================================================
# DATOS REALES: DESCARGA O CSV
# =============================================================================

def load_returns_from_csv(
    csv_path: str,
    col_fecha: str = "fecha",
    col_cierre: str = "cierre",
    start: Optional[str] = None,
    end: Optional[str] = None
) -> pd.Series:
    """
    Carga precios desde un CSV y devuelve rendimientos logarítmicos diarios.
    """
    df = pd.read_csv(csv_path)
    if start is not None:
        df = df[df[col_fecha] >= start]
    if end is not None:
        df = df[df[col_fecha] <= end]

    close = df[col_cierre].dropna()
    returns = np.log(close / close.shift(1)).dropna()
    returns.name = "R"
    return returns


# =============================================================================
# FLUJOS COMPLETOS: EJEMPLOS
# =============================================================================

def run_simulated_evt_example() -> None:
    """
    Ejecuta el ejemplo completo de EVT con datos simulados (t-Student).
    """
    print("=== Ejemplo con datos simulados (t-Student) ===")

    data = simulate_t_series(n=5000, df=3, name="X", random_state=42)
    plot_histogram(
        data,
        bins=50,
        title="Histograma de la serie simulada (colas pesadas)",
        xlabel="Valor",
    )

    # --- Block Maxima ---
    block_maxima, n_blocks, trimmed = compute_block_maxima(data, BLOCK_SIZE)
    plot_histogram(
        block_maxima,
        bins=20,
        title="Histograma de máximos por bloque (simulado)",
        xlabel="Máximo del bloque",
    )
    plot_block_structure(trimmed, BLOCK_SIZE,
                         title=f"Block Maxima: bloques de {BLOCK_SIZE} con máximos (simulado)")

    gev_params = fit_gev(block_maxima)
    c_hat = -gev_params["xi_hat"]  # volver a convención de SciPy
    loc_hat = gev_params["loc_hat"]
    scale_hat = gev_params["scale_hat"]

    print("Parámetros GEV estimados (simulado):")
    print(gev_params)

    plot_gev_fit(block_maxima, c_hat, loc_hat, scale_hat)
    plot_gev_qq(block_maxima, c_hat, loc_hat, scale_hat)

    x_T = gev_return_level(T_BLOCKS, c_hat, loc_hat, scale_hat)
    print(f"Nivel de retorno GEV para T={T_BLOCKS} bloques (simulado): {x_T:.4f}")

    # --- POT ---
    u = data.quantile(QUANTILE_THRESHOLD)
    exceedances = compute_exceedances(data, u)
    print(f"Umbral u (simulado) = {u:.4f}")
    print(f"Cantidad de excesos: {len(exceedances)}")
    plot_histogram(
        exceedances,
        bins=20,
        title="Histograma de excesos sobre el umbral (simulado)",
        xlabel="Exceso y = X - u",
    )

    gpd_params = fit_gpd(exceedances)
    print("Parámetros GPD estimados (simulado):")
    print(gpd_params)

    plot_gpd_fit(exceedances, gpd_params["xi_gpd"], gpd_params["beta_gpd"])
    plot_pot_points(data, u)

    z = data.quantile(P_CONF)
    p_u = (data > u).mean()
    p_exceed_z = gpd_prob_exceed(
        z=z,
        u=u,
        xi_gpd=gpd_params["xi_gpd"],
        beta_gpd=gpd_params["beta_gpd"],
        p_u=p_u,
    )
    print(f"Nivel z (simulado) = {z:.4f}")
    print(f"P(X > z) estimada por POT (simulado) = {p_exceed_z:.6f}")


def run_real_data_evt_example(
    ticker: Optional[str] = None,
    csv_path: Optional[str] = None,
    start: str = "2023-01-01",
    end: str = "2025-01-01"
) -> None:
    """
    Ejecuta el ejemplo de EVT con datos reales (mercado).

    Puedes usar:
    - ticker de yfinance, o
    - csv_path a un archivo local con columnas [fecha, cierre].
    """
    print("=== Ejemplo con datos reales ===")

    if csv_path is not None:
        print(f"Usando datos desde CSV: {csv_path}")
        returns = load_returns_from_csv(csv_path, start=start, end=end)
        source_label = csv_path
    elif ticker is not None:
        if yf is None:
            raise ImportError(
                "yfinance no está instalado. Ejecutá: pip install yfinance"
            )
        print(f"Descargando datos de yfinance: {ticker} ({start} → {end})")
        raw = yf.download(ticker, start=start, end=end, progress=False)
        if raw.empty:
            raise ValueError(f"No se obtuvieron datos para el ticker '{ticker}'.")
        close = raw["Close"].squeeze().dropna()
        returns = np.log(close / close.shift(1)).dropna()
        returns.name = "R"
        source_label = ticker
    else:
        raise ValueError("Debes especificar 'ticker' o 'csv_path' para datos reales.")

    # Definimos pérdidas L = -R
    losses = -returns
    losses.name = "L"
    print(losses.describe())

    plot_histogram(
        losses,
        bins=50,
        title=f"Histograma de pérdidas diarias ({source_label})",
        xlabel="Pérdida diaria L = -R",
    )

    # --- Block Maxima ---
    block_maxima, n_blocks, trimmed = compute_block_maxima(losses, BLOCK_SIZE)
    plot_block_structure(
        trimmed,
        BLOCK_SIZE,
        title=f"Block Maxima: bloques de {BLOCK_SIZE} pérdidas ({source_label})",
    )
    plot_histogram(
        block_maxima,
        bins=20,
        title="Histograma de máximos de pérdidas por bloque",
        xlabel="Máxima pérdida en el bloque",
    )

    gev_params = fit_gev(block_maxima)
    c_hat = -gev_params["xi_hat"]
    loc_hat = gev_params["loc_hat"]
    scale_hat = gev_params["scale_hat"]

    print("Parámetros GEV estimados (reales):")
    print(gev_params)

    plot_gev_fit(block_maxima, c_hat, loc_hat, scale_hat)
    plot_gev_qq(block_maxima, c_hat, loc_hat, scale_hat)

    x_T = gev_return_level(T_BLOCKS, c_hat, loc_hat, scale_hat)
    print(f"Nivel de retorno GEV para T={T_BLOCKS} bloques (reales): {x_T:.4f}")

    var_block = gev_var_block(P_CONF, c_hat, loc_hat, scale_hat)
    print(f"VaR GEV (máximo de bloque) al {P_CONF*100:.2f}% = {var_block:.4f}")
    prob_ex_z = gev_prob_exceed(0.06, c_hat, loc_hat, scale_hat)
    print(f"P(Máximo de bloque > 6% pérdida) = {prob_ex_z:.6f}")

    var_p, es_p = gev_var_es_block(P_CONF, c_hat, loc_hat, scale_hat)
    print(f"VaR GEV (bloque) 99.5% = {var_p:.4f}")
    print(f"ES  GEV (bloque) 99.5% = {es_p:.4f}")

    # --- POT ---
    u = losses.quantile(QUANTILE_THRESHOLD)
    exceedances = compute_exceedances(losses, u)
    print(f"Umbral u (reales) = {u:.4f}")
    print(f"Cantidad de excesos: {len(exceedances)}")

    gpd_params = fit_gpd(exceedances)
    print("Parámetros GPD estimados (reales):")
    print(gpd_params)

    plot_gpd_fit(exceedances, gpd_params["xi_gpd"], gpd_params["beta_gpd"])
    plot_pot_points(losses, u)

    z = losses.quantile(P_CONF)
    p_u = (losses > u).mean()
    p_exceed_z = gpd_prob_exceed(
        z=z,
        u=u,
        xi_gpd=gpd_params["xi_gpd"],
        beta_gpd=gpd_params["beta_gpd"],
        p_u=p_u,
    )
    print(f"Nivel z (reales) = {z:.4f}")
    print(f"P(L > z) estimada por POT (reales) = {p_exceed_z:.6f}")

    z_T_pot = gpd_return_level(
        T_days=T_DAYS,
        u=u,
        xi_gpd=gpd_params["xi_gpd"],
        beta_gpd=gpd_params["beta_gpd"],
        p_u=p_u,
    )
    print(f"Nivel de retorno POT para T={T_DAYS} días: z_T = {z_T_pot:.4f}")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    # Ejemplo 1: datos simulados
    # run_simulated_evt_example()

    # Ejemplo 2: datos reales
    run_real_data_evt_example(csv_path="SPY_US - Cotizaciones historicas.csv",
                              start="2023-01-01", end="2025-01-01")


