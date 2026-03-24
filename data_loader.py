"""
Carga unificada de datos financieros para análisis EVT.

Soporta múltiples fuentes (yfinance, pandas_datareader, CSV local)
y múltiples tipos de activos (acciones, ETFs, índices, criptos, commodities).
Siempre devuelve pd.Series de log-returns con DatetimeIndex.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes internas
# ---------------------------------------------------------------------------

MIN_OBSERVATIONS: int = 200

MAX_NAN_RATIO: float = 0.05

MAX_GAP_CALENDAR_DAYS: int = 30

# Patrones para detección automática de asset_type
_CRYPTO_RE = re.compile(
    r"^[A-Z]{2,10}-(?:USD|EUR|GBP|BTC|ETH|USDT|BUSD)$", re.IGNORECASE
)
_INDEX_RE = re.compile(r"^\^")

_KNOWN_ETFS: frozenset[str] = frozenset({
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "EEM", "EWZ", "EFA",
    "GLD", "SLV", "USO", "UNG", "TLT", "HYG", "XLF", "XLE", "XLK",
    "ARKK", "IEMG", "VWO", "AGG", "BND", "LQD", "SCHD", "JEPI",
})

# Nombres comunes de columnas de cierre / fecha en CSVs financieros
_CLOSE_ALIASES: list[str] = [
    "close", "cierre", "adj close", "adj_close", "adjclose",
    "precio", "price", "ultimo", "last", "settle",
]
_DATE_ALIASES: list[str] = [
    "date", "fecha", "datetime", "timestamp", "time", "day", "periodo",
]


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  DETECCIÓN AUTOMÁTICA DE TIPO DE ACTIVO                               ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def detect_asset_type(ticker: str) -> str:
    """
    Infiere el tipo de activo a partir de la cadena del ticker.

    Returns
    -------
    str
        Uno de: ``"crypto"``, ``"index"``, ``"etf"``, ``"equity"``.
    """
    if _CRYPTO_RE.match(ticker):
        return "crypto"
    if _INDEX_RE.match(ticker):
        return "index"
    symbol = ticker.split(".")[0].upper()
    if symbol in _KNOWN_ETFS:
        return "etf"
    return "equity"


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  FUNCIONES DE CARGA POR FUENTE                                        ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def _load_yfinance(
    ticker: str,
    start: str,
    end: str,
) -> pd.Series:
    """Descarga precios de cierre ajustado vía *yfinance*."""
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        raise ImportError(
            "yfinance no está instalado.  Ejecutá:  pip install yfinance"
        )

    raw = yf.download(ticker, start=start, end=end, progress=False)

    if raw.empty:
        raise ValueError(
            f"yfinance no devolvió datos para '{ticker}' "
            f"en el rango {start} → {end}."
        )

    # yfinance puede devolver MultiIndex de columnas para un solo ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    for col_name in ("Adj Close", "Close"):
        if col_name in raw.columns:
            return raw[col_name].squeeze().dropna().rename(ticker)

    raise KeyError(
        f"No se encontró columna de cierre en los datos de yfinance para '{ticker}'."
    )


def _load_datareader(
    ticker: str,
    start: str,
    end: str,
) -> pd.Series:
    """Intenta obtener precios vía *pandas_datareader* (Yahoo o FRED)."""
    try:
        import pandas_datareader.data as web  # type: ignore
    except ImportError:
        raise ImportError(
            "pandas_datareader no está instalado.  "
            "Ejecutá:  pip install pandas-datareader"
        )

    # FRED usa tickers como DGS10, DEXUSEU, etc.
    for source in ("yahoo", "fred"):
        try:
            df = web.DataReader(ticker, source, start, end)
        except Exception:
            continue

        if df.empty:
            continue

        # FRED devuelve una sola columna con el nombre del ticker
        if df.shape[1] == 1:
            return df.iloc[:, 0].dropna().rename(ticker)

        for col_name in ("Adj Close", "Close"):
            if col_name in df.columns:
                return df[col_name].squeeze().dropna().rename(ticker)

    raise ValueError(
        f"pandas_datareader no pudo obtener datos para '{ticker}' "
        f"en el rango {start} → {end}."
    )


def _find_column(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    """Busca la primera columna cuyo nombre (normalizado) coincida con *aliases*."""
    for col in df.columns:
        normalized = col.strip().lower().replace(" ", "_")
        if normalized in aliases:
            return col
    return None


def _load_csv(
    path: str,
    start: Optional[str],
    end: Optional[str],
) -> pd.Series:
    """
    Carga precios desde un CSV local con detección automática de columnas.

    Busca una columna de fecha y una de cierre usando alias comunes.
    Si las encuentra, filtra por rango y devuelve la serie de precios.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_csv(path)

    # --- Detectar columna de fecha ---
    date_col = _find_column(df, _DATE_ALIASES)
    if date_col is None:
        raise KeyError(
            f"No se detectó columna de fecha en '{path}'.  "
            f"Columnas disponibles: {list(df.columns)}.  "
            f"Se esperaba alguna de: {_DATE_ALIASES}"
        )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    if start is not None:
        df = df[df[date_col] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df[date_col] <= pd.Timestamp(end)]

    # --- Detectar columna de cierre ---
    close_col = _find_column(df, _CLOSE_ALIASES)
    if close_col is None:
        raise KeyError(
            f"No se detectó columna de cierre en '{path}'.  "
            f"Columnas disponibles: {list(df.columns)}.  "
            f"Se esperaba alguna de: {_CLOSE_ALIASES}"
        )

    series = df.set_index(date_col)[close_col].dropna()
    series.index.name = None
    series.name = Path(path).stem
    return series


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  VALIDACIONES                                                          ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class DataQualityWarning(UserWarning):
    """Advertencia emitida cuando los datos tienen problemas de calidad."""


def _validate_returns(returns: pd.Series, source_label: str) -> pd.Series:
    """
    Valida la serie de log-returns y emite advertencias / errores claros.

    Chequeos:
    1. Mínimo de observaciones.
    2. Proporción de NaN.
    3. Gaps temporales grandes (solo si el índice es DatetimeIndex).
    """
    import warnings

    # 1) Serie demasiado corta
    n_obs = len(returns)
    if n_obs < MIN_OBSERVATIONS:
        raise ValueError(
            f"'{source_label}' tiene solo {n_obs} observaciones "
            f"(mínimo requerido: {MIN_OBSERVATIONS}).  "
            "Probá un rango de fechas más amplio."
        )

    # 2) Demasiados NaN
    n_nan = int(returns.isna().sum())
    ratio = n_nan / len(returns) if len(returns) > 0 else 0.0
    if ratio > MAX_NAN_RATIO:
        warnings.warn(
            f"'{source_label}': {n_nan} NaN ({ratio:.1%}) superan el umbral "
            f"de {MAX_NAN_RATIO:.0%}.  Se eliminarán, pero revisá la calidad "
            "del dato subyacente.",
            DataQualityWarning,
            stacklevel=3,
        )

    # 3) Gaps temporales grandes
    if isinstance(returns.index, pd.DatetimeIndex) and len(returns) > 1:
        deltas = returns.index.to_series().diff().dropna()
        max_gap = deltas.max()
        if max_gap.days > MAX_GAP_CALENDAR_DAYS:
            warnings.warn(
                f"'{source_label}': gap máximo de {max_gap.days} días calendario "
                f"(umbral: {MAX_GAP_CALENDAR_DAYS}).  "
                "Puede indicar datos faltantes o un activo ilíquido.",
                DataQualityWarning,
                stacklevel=3,
            )

    # Limpiar NaN restantes
    returns = returns.dropna()
    return returns


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  FUNCIÓN PRINCIPAL                                                     ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def get_returns(
    source: Union[str, Path, pd.DataFrame],
    start: str = "2020-01-01",
    end: str = "2025-01-01",
    asset_type: Optional[str] = None,
) -> pd.Series:
    """
    Punto de entrada único para obtener log-returns de cualquier fuente.

    Parameters
    ----------
    source : str | Path | pd.DataFrame
        - **Path a CSV local**: se detecta si el string apunta a un archivo
          existente (extensión ``.csv``).
        - **Ticker string**: ``"AAPL"``, ``"^GSPC"``, ``"BTC-USD"``, etc.
        - **DataFrame**: debe contener al menos una columna de cierre
          reconocible; se espera un DatetimeIndex o una columna de fecha.
    start, end : str
        Rango de fechas en formato ``"YYYY-MM-DD"``.
    asset_type : str | None
        ``"equity"``, ``"etf"``, ``"index"``, ``"crypto"`` o ``None``
        para detección automática.  Se usa solo con fines informativos
        (log) ya que yfinance acepta cualquier ticker válido de Yahoo.

    Returns
    -------
    pd.Series
        Log-returns diarios con ``DatetimeIndex`` limpio y nombre
        descriptivo.  Sin NaN.

    Raises
    ------
    ValueError
        Si no se puede obtener datos de ninguna fuente, o la serie
        resultante tiene menos de ``MIN_OBSERVATIONS`` puntos.
    FileNotFoundError
        Si *source* es un path que no existe.

    Examples
    --------
    >>> returns = get_returns("AAPL", start="2022-01-01", end="2024-01-01")
    >>> returns = get_returns("BTC-USD")
    >>> returns = get_returns("SPY_US - Cotizaciones historicas.csv")
    >>> returns = get_returns(my_dataframe)
    """

    # --- Caso 1: DataFrame ya cargado ---
    if isinstance(source, pd.DataFrame):
        prices = _prices_from_dataframe(source)
        label = "DataFrame"
        returns = _prices_to_log_returns(prices, label)
        return _validate_returns(returns, label)

    source_str = str(source)

    # --- Caso 2: CSV local ---
    if _looks_like_csv_path(source_str):
        prices = _load_csv(source_str, start, end)
        label = Path(source_str).stem
        returns = _prices_to_log_returns(prices, label)
        return _validate_returns(returns, label)

    # --- Caso 3: Ticker → fallback chain ---
    ticker = source_str.strip()

    if asset_type is None:
        asset_type = detect_asset_type(ticker)
    logger.info("Tipo de activo detectado para '%s': %s", ticker, asset_type)

    errors: list[str] = []

    # 3a. yfinance
    try:
        prices = _load_yfinance(ticker, start, end)
        returns = _prices_to_log_returns(prices, ticker)
        return _validate_returns(returns, ticker)
    except Exception as exc:
        errors.append(f"yfinance: {exc}")
        logger.debug("yfinance falló para '%s': %s", ticker, exc)

    # 3b. pandas_datareader
    try:
        prices = _load_datareader(ticker, start, end)
        returns = _prices_to_log_returns(prices, ticker)
        return _validate_returns(returns, ticker)
    except Exception as exc:
        errors.append(f"pandas_datareader: {exc}")
        logger.debug("pandas_datareader falló para '%s': %s", ticker, exc)

    # 3c. Ninguna fuente funcionó
    error_detail = "\n  - ".join(errors)
    raise ValueError(
        f"No se pudieron obtener datos para '{ticker}' de ninguna fuente.\n"
        f"Errores:\n  - {error_detail}\n"
        "Verificá que el ticker sea válido o proporcioná un CSV local."
    )


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS INTERNOS                                                      ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def _looks_like_csv_path(source: str) -> bool:
    """Devuelve True si *source* parece ser un path a un archivo CSV."""
    if source.lower().endswith(".csv"):
        return True
    if os.path.sep in source or "/" in source:
        return Path(source).suffix.lower() in (".csv", ".tsv", ".txt")
    return False


def _prices_from_dataframe(df: pd.DataFrame) -> pd.Series:
    """
    Extrae la columna de precios de cierre de un DataFrame genérico.

    Si el DataFrame ya tiene DatetimeIndex, lo preserva.
    Si no, busca una columna de fecha y la usa como índice.
    """
    work = df.copy()

    # Intentar asegurar DatetimeIndex
    if not isinstance(work.index, pd.DatetimeIndex):
        date_col = _find_column(work, _DATE_ALIASES)
        if date_col is not None:
            work[date_col] = pd.to_datetime(work[date_col])
            work = work.set_index(date_col).sort_index()

    close_col = _find_column(work, _CLOSE_ALIASES)
    if close_col is None:
        # Último recurso: si solo queda una columna numérica, usarla
        numeric_cols = work.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 1:
            close_col = numeric_cols[0]
        else:
            raise KeyError(
                "No se detectó columna de cierre en el DataFrame.  "
                f"Columnas disponibles: {list(work.columns)}.  "
                f"Se esperaba alguna de: {_CLOSE_ALIASES}"
            )

    series = work[close_col].dropna()
    series.name = close_col
    return series


def _prices_to_log_returns(prices: pd.Series, label: str) -> pd.Series:
    """Convierte serie de precios a log-returns, preservando el índice."""
    if (prices <= 0).any():
        n_neg = int((prices <= 0).sum())
        logger.warning(
            "'%s': %d precios ≤ 0 encontrados; se eliminarán antes del log.",
            label, n_neg,
        )
        prices = prices[prices > 0]

    returns = np.log(prices / prices.shift(1)).dropna()
    returns.name = f"R_{label}"

    # Asegurar DatetimeIndex si es posible
    if isinstance(returns.index, pd.DatetimeIndex):
        returns.index = returns.index.normalize()

    return returns
