import pandas as pd
from src.config import FEATURE_PARAMS

def calculate_ema(df: pd.DataFrame, period: int, column: str) -> pd.Series:
    """Вычисляет Exponential Moving Average (EMA)."""
    return df[column].ewm(span=period, adjust=False).mean()

def calculate_sma(df: pd.DataFrame, period: int, column: str) -> pd.Series:
    """Вычисляет Simple Moving Average (SMA)."""
    return df[column].rolling(window=period).mean()

def calculate_rsi(df: pd.DataFrame, period: int, column: str) -> pd.Series:
    """Вычисляет Relative Strength Index (RSI)."""
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Расширенный конвейер генерации признаков.
    Все константы берутся централизованно из config.py.
    """
    df_features = df.copy()

    col = FEATURE_PARAMS['target_column']
    ema_fast = FEATURE_PARAMS['ema_fast_period']
    ema_slow = FEATURE_PARAMS['ema_slow_period']
    sma_per = FEATURE_PARAMS['sma_period']
    rsi_per = FEATURE_PARAMS['rsi_period']

    # 1. Трендовые индикаторы (База)
    df_features[f'EMA_{ema_fast}'] = calculate_ema(df_features, ema_fast, col)
    df_features[f'EMA_{ema_slow}'] = calculate_ema(df_features, ema_slow, col)
    df_features[f'SMA_{sma_per}'] = calculate_sma(df_features, sma_per, col)

    # 2. Осцилляторы
    df_features[f'RSI_{rsi_per}'] = calculate_rsi(df_features, rsi_per, col)

    # 3. Относительные фичи (Идеально для нейросетей)
    # Расстояние между скользящими средними
    df_features['EMA_spread'] = df_features[f'EMA_{ema_fast}'] - df_features[f'EMA_{ema_slow}']
    # На сколько процентов цена отклонилась от средней
    df_features['Price_to_SMA'] = df_features[col] / df_features[f'SMA_{sma_per}']

    # 4. Доходность и Волатильность
    # Дневная доходность (log returns или обычный pct_change)
    df_features['Daily_Return'] = df_features[col].pct_change().fillna(0)
    # Историческая волатильность за период SMA
    df_features['Volatility'] = df_features['Daily_Return'].rolling(window=sma_per).std().fillna(0)

    # Из-за rolling-функций первые значения (равные периоду SMA) будут NaN.
    # Дропаем их, чтобы у нейросети был чистый датасет без пустот.
    return df_features.dropna()