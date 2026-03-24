"""
Constantes de configuración para el análisis EVT.

Centraliza los valores que antes estaban hardcodeados en
Utilidades_GEV_BM_2.py para facilitar la experimentación.
"""

# Block Maxima: cantidad de observaciones por bloque
BLOCK_SIZE: int = 50

# POT: cuantil para definir el umbral u = series.quantile(QUANTILE_THRESHOLD)
QUANTILE_THRESHOLD: float = 0.95

# Nivel de confianza para VaR / ES
P_CONF: float = 0.995

# Periodo de retorno en bloques (GEV)
T_BLOCKS: int = 100

# Periodo de retorno en días (POT)
T_DAYS: int = 5000
