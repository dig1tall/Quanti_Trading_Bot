import logging
import numpy as np
import pandas as pd
from src.config import FEATURE_PARAMS

# Инициализируем логгер для модуля генерации фич
logger = logging.getLogger(__name__)

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

    # Защита от деления на ноль: если avg_loss == 0, rs станет inf
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(df: pd.DataFrame, fast: int, slow: int, signal: int, column: str) -> tuple[pd.Series, pd.Series]:
    """Вычисляет линии MACD и Signal."""
    macd_line = calculate_ema(df, fast, column) - calculate_ema(df, slow, column)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calculate_bollinger_bands(df: pd.DataFrame, period: int, std_dev: int, column: str) -> tuple[pd.Series, pd.Series]:
    """Вычисляет Верхнюю и Нижнюю полосы Боллинджера."""
    sma = calculate_sma(df, period, column)
    std = df[column].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Расширенный конвейер генерации признаков с валидацией аномалий.
    Все константы берутся централизованно из config.py.
    """
    initial_rows = len(df)
    logger.info("Запуск Feature Engineering пайплайна...")

    try:
        df_features = df.copy()

        col = FEATURE_PARAMS['target_column']
        ema_fast = FEATURE_PARAMS['ema_fast_period']
        ema_slow = FEATURE_PARAMS['ema_slow_period']
        sma_per = FEATURE_PARAMS['sma_period']
        rsi_per = FEATURE_PARAMS['rsi_period']
        macd_sig = FEATURE_PARAMS['macd_signal_period']
        bb_per = FEATURE_PARAMS['bb_period']
        bb_std = FEATURE_PARAMS['bb_std_dev']

        logger.info(
            f"Конфигурация фич: Target={col}, EMA({ema_fast}/{ema_slow}), "
            f"SMA={sma_per}, RSI={rsi_per}, MACD_Signal={macd_sig}, BB({bb_per}, {bb_std})"
        )

        # 1. Трендовые индикаторы (База)
        df_features[f'EMA_{ema_fast}'] = calculate_ema(df_features, ema_fast, col)
        df_features[f'EMA_{ema_slow}'] = calculate_ema(df_features, ema_slow, col)
        df_features[f'SMA_{sma_per}'] = calculate_sma(df_features, sma_per, col)

        # 2. Осцилляторы & Импульс
        df_features[f'RSI_{rsi_per}'] = calculate_rsi(df_features, rsi_per, col)

        # Добавляем MACD и гистограмму (разница между линиями)
        macd, signal = calculate_macd(df_features, ema_fast, ema_slow, macd_sig, col)
        df_features['MACD'] = macd
        df_features['MACD_Signal'] = signal
        df_features['MACD_Hist'] = macd - signal

        # 3. Волатильность & Границы (Bollinger Bands)
        upper, lower = calculate_bollinger_bands(df_features, bb_per, bb_std, col)
        # Относительное положение цены внутри полос (от 0 до 1) — идеально для нейросетей
        df_features['BB_Position'] = (df_features[col] - lower) / (upper - lower)
        df_features['BB_Position'] = df_features['BB_Position'].fillna(0.5)

        # 4. Относительные фичи
        df_features['EMA_spread'] = df_features[f'EMA_{ema_fast}'] - df_features[f'EMA_{ema_slow}']
        df_features['Price_to_SMA'] = df_features[col] / df_features[f'SMA_{sma_per}']

        # 5. Доходность и Волатильность
        df_features['Daily_Return'] = df_features[col].pct_change().fillna(0)
        df_features['Volatility'] = df_features['Daily_Return'].rolling(window=sma_per).std().fillna(0)

        # --- БЛОК ЗАЩИТЫ И ВАЛИДАЦИИ ФИЧ ---

        # Проверка на бесконечные значения (inf)
        inf_counts = np.isinf(df_features).sum().sum()
        if inf_counts > 0:
            logger.warning(f"Обнаружено {inf_counts} значений +/- c бесконечностью (inf). Производится замена на корректные числа.")
            df_features = df_features.replace([np.inf, -np.inf], np.nan)

        # Дропаем временные NaN, возникшие из-за rolling-окон
        df_clean = df_features.dropna()
        dropped_rows = initial_rows - len(df_clean)

        if len(df_clean) == 0:
            logger.warning("После удаления NaN матрица признаков пуста! Проверь длину истории или периоды скользящих окон.")
        elif len(df_clean) < 100:
            logger.warning(f"Критически мало данных для обучения модели после расчета фич: всего {len(df_clean)} строк.")
        else:
            logger.info(
                f"Генерация фич завершена успешно. Удалено строк с NaN: {dropped_rows}. "
                f"Строк на выходе: {len(df_clean)}"
            )

        return df_clean

    except Exception as e:
        logger.error(f"Критический сбой при расчете математических признаков: {e}", exc_info=True)
        return pd.DataFrame()