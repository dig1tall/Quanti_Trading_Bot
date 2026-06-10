import logging
import numpy as np
import pandas as pd
import os

from src import config
from src.config import FEATURE_PARAMS

# Инициализируем логгер для модуля генерации фич
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Класс генерации признаков (Feature Engineering пайплайн) с валидацией аномалий.
    Все константы по умолчанию берутся централизованно из config.py.
    """
    def __init__(self):
        # Распаковываем конфигурацию для удобного использования в методах класса
        self.col = FEATURE_PARAMS['target_column']
        self.ema_fast = FEATURE_PARAMS['ema_fast_period']
        self.ema_slow = FEATURE_PARAMS['ema_slow_period']
        self.sma_per = FEATURE_PARAMS['sma_period']
        self.rsi_per = FEATURE_PARAMS['rsi_period']
        self.macd_sig = FEATURE_PARAMS['macd_signal_period']
        self.bb_per = FEATURE_PARAMS['bb_period']
        self.bb_std = FEATURE_PARAMS['bb_std_dev']

    def _calculate_ema(self, df: pd.DataFrame, period: int, column: str) -> pd.Series:
        """Вычисляет Exponential Moving Average (EMA)."""
        return df[column].ewm(span=period, adjust=False).mean()

    def _calculate_sma(self, df: pd.DataFrame, period: int, column: str) -> pd.Series:
        """Вычисляет Simple Moving Average (SMA)."""
        return df[column].rolling(window=period).mean()

    def _calculate_rsi(self, df: pd.DataFrame, period: int, column: str) -> pd.Series:
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

    def _calculate_macd(self, df: pd.DataFrame, fast: int, slow: int, signal: int, column: str) -> tuple[pd.Series, pd.Series]:
        """Вычисляет линии MACD и Signal."""
        macd_line = self._calculate_ema(df, fast, column) - self._calculate_ema(df, slow, column)
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int, std_dev: int, column: str) -> tuple[pd.Series, pd.Series]:
        """Вычисляет Верхнюю и Нижнюю полосы Боллинджера."""
        sma = self._calculate_sma(df, period, column)
        std = df[column].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расширенный конвейер генерации признаков с валидацией аномалий.
        Все константы берутся централизованно из config.py.
        """
        initial_rows = len(df)
        logger.info("Запуск Feature Engineering пайплайна...")

        try:
            df_features = df.copy()

            logger.info(
                f"Конфигурация фич: Target={self.col}, EMA({self.ema_fast}/{self.ema_slow}), "
                f"SMA={self.sma_per}, RSI={self.rsi_per}, MACD_Signal={self.macd_sig}, BB({self.bb_per}, {self.bb_std})"
            )

            # 1. Трендовые индикаторы (База)
            df_features[f'EMA_{self.ema_fast}'] = self._calculate_ema(df_features, self.ema_fast, self.col)
            df_features[f'EMA_{self.ema_slow}'] = self._calculate_ema(df_features, self.ema_slow, self.col)
            df_features[f'SMA_{self.sma_per}'] = self._calculate_sma(df_features, self.sma_per, self.col)

            # 2. Осцилляторы & ...мпульс
            df_features[f'RSI_{self.rsi_per}'] = self._calculate_rsi(df_features, self.rsi_per, self.col)

            # MACD и гистограмма (разница между линиями)
            macd, signal = self._calculate_macd(df_features, self.ema_fast, self.ema_slow, self.macd_sig, self.col)
            df_features['MACD'] = macd
            df_features['MACD_Signal'] = signal
            df_features['MACD_Hist'] = macd - signal

            # 3. Волатильность & Границы (Bollinger Bands)
            upper, lower = self._calculate_bollinger_bands(df_features, self.bb_per, self.bb_std, self.col)
            # Относительное положение цены внутри полос (от 0 до 1) — идеально для нейросетей
            df_features['BB_Position'] = (df_features[self.col] - lower) / (upper - lower)
            df_features['BB_Position'] = df_features['BB_Position'].fillna(0.5)

            # 4. Относительные фичи
            df_features['EMA_spread'] = df_features[f'EMA_{self.ema_fast}'] - df_features[f'EMA_{self.ema_slow}']
            df_features['Price_to_SMA'] = df_features[self.col] / df_features[f'SMA_{self.sma_per}']

            # 5. Доходность и Волатильность
            df_features['Daily_Return'] = df_features[self.col].pct_change().fillna(0)
            df_features['Volatility'] = df_features['Daily_Return'].rolling(window=self.sma_per).std().fillna(0)

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

# --- АВТОНОМНЫЙ ТЕСТ МОДУЛЯ ---
if __name__ == "__main__":
    from src.config import setup_logging

    # Инициализируем логгер через конфигурацию
    setup_logging(level=logging.INFO)

    logger.info("=== Запуск FeatureExtractor в автономном режиме ===")

    # Берем путь к сырому базовому файлу цен из конфига
    file_path = config.DATA_FILE_PATH

    if os.path.exists(file_path):
        # Загружаем сохраненный DataLoader'ом Parquet
        base_df = pd.read_parquet(file_path)
        logger.info(f"Успешно загружен базовый файл: {file_path} (Размерность: {base_df.shape})")

        # Прогоняем экстрактор
        extractor = FeatureExtractor()
        df_with_features = extractor.extract_features(base_df)

        print("\nПревью сгенерированных индикаторов (последние 3 строки):")
        # Показываем только новые сгенерированные колонки, чтобы не захламлять консоль базовыми OHLCV
        new_cols = [col for col in df_with_features.columns if col not in base_df.columns]
        print(df_with_features[new_cols].tail(3))
    else:
        logger.error(f"Базовый файл с ценами не найден по пути: {file_path}")
        logger.error("Сначала запусти модуль data_loader.py или main.py, чтобы скачать сырые данные.")