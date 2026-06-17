import logging
import numpy as np
import pandas as pd
import os

from src import config
from src.config import FEATURE_PARAMS

# Инициализация логгера для модуля генерации фич
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Класс генерации признаков (Feature Engineering пайплайн) с валидацией аномалий.
    Все константы по умолчанию берутся централизованно из config.py.
    """
    def __init__(self):
        # Распаковка конфигурации для удобного использования в методах класса
        self.col      = FEATURE_PARAMS['target_column']
        self.ema_fast = FEATURE_PARAMS['ema_fast_period']
        self.ema_slow = FEATURE_PARAMS['ema_slow_period']
        self.sma_per  = FEATURE_PARAMS['sma_period']
        self.rsi_per  = FEATURE_PARAMS['rsi_period']
        self.macd_sig = FEATURE_PARAMS['macd_signal_period']
        self.bb_per   = FEATURE_PARAMS['bb_period']
        self.bb_std   = FEATURE_PARAMS['bb_std_dev']
        self.adx_per        = FEATURE_PARAMS['adx_period']
        self.obv_window     = FEATURE_PARAMS['obv_rolling_window']
        self.chaikin_window = FEATURE_PARAMS['chaikin_rolling_window']

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

    def _calculate_obv(self, df: pd.DataFrame, price_col: str, vol_col: str) -> pd.Series:
        """Безопасный On-Balance Volume (изменение за окно, без бесконечного cumsum)."""
        direction = np.sign(df[price_col].diff()).fillna(0)
        # Динамически используем окно из конфигурации
        obv_window = (direction * df[vol_col]).rolling(window=self.obv_window).sum().fillna(0)
        return obv_window

    def _calculate_chaikin_oscillator(self, df: pd.DataFrame, price_col: str, vol_col: str) -> pd.Series:
        """Безопасный и нормированный Осциллятор Чайкина."""
        close = df[price_col]
        high = df['High']
        low = df['Low']
        vol = df[vol_col]

        denom = (high - low).replace(0, 1e-8)
        mf_multiplier = ((close - low) - (high - close)) / denom
        mf_volume = mf_multiplier * vol

        # Динамически используем окно накопления денег из конфигурации
        ad_volume_window = mf_volume.rolling(window=self.chaikin_window).sum().fillna(0)

        # Разница EMA от нормированного окна
        chaikin = ad_volume_window.ewm(span=3, adjust=False).mean() - ad_volume_window.ewm(span=10, adjust=False).mean()
        return chaikin

    def _calculate_adx_simplified(self, df: pd.DataFrame, period: int, price_col: str) -> pd.Series:
        """
        Вычисляет упрощенную силу тренда (аналог ADX) с жестким сохранением исходного индекса дат.
        Защищен от неявного приведения типов в numpy.ndarray и деления на ноль.
        """
        high = df['High']
        low = df['Low']

        # 1. Направленные движения (дифференциалы)
        up_move = high.diff()
        down_move = low.diff()

        # np.where возвращает голый массив numpy — сразу заворачиваем в Series с нужным индексом
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0),
            index=df.index
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0),
            index=df.index
        )

        # 2. Сглаживание направлений через EMA
        plus_di = plus_dm.ewm(span=period, adjust=False).mean()
        minus_di = minus_dm.ewm(span=period, adjust=False).mean()

        # 3. Расчет базового индекса направления (DX)
        denom = (plus_di + minus_di).replace(0, 1e-8)

        # Чтобы np.abs() не сбросил Series в ndarray, используем встроенный метод .abs() от Pandas
        raw_dx = ((plus_di - minus_di).abs() / denom) * 100

        # На всякий случай жестко контролируем, что dx остался Series с индексом
        dx_series = pd.Series(raw_dx, index=df.index)

        # 4. Финальное сглаживание — получаем итоговую силу тренда (ADX)
        adx = dx_series.ewm(span=period, adjust=False).mean()

        return adx

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Модернизированный конвейер генерации признаков.
        Исключает абсолютные цены из матрицы признаков для предотвращения утечки данных (Data Leakage).
        """
        initial_rows = len(df)
        logger.info("Запуск Feature Engineering пайплайна...")

        try:
            df_features = df.copy()

            # Базовые колонки, которые мы сохраним для бэктеста, но удалим из X перед моделью
            # Считаем Daily_Return в самом начале, так как теперь он — основа всего
            df_features['Daily_Return'] = df_features[self.col].pct_change().fillna(0)

            logger.info(
                f"Конфигурация фич (на базе Returns): EMA({self.ema_fast}/{self.ema_slow}), "
                f"SMA={self.sma_per}, RSI={self.rsi_per}, MACD_Signal={self.macd_sig}, BB({self.bb_per}, {self.bb_std}), "
                f"ADX_Per={self.adx_per}, OBV_Win={self.obv_window}, Chaikin_Win={self.chaikin_window}"
            )

            # 1. Трендовые индикаторы доходностей (теперь они колеблются около нуля!)
            df_features[f'EMA_{self.ema_fast}'] = self._calculate_ema(df_features, self.ema_fast, 'Daily_Return')
            df_features[f'EMA_{self.ema_slow}'] = self._calculate_ema(df_features, self.ema_slow, 'Daily_Return')
            df_features[f'SMA_{self.sma_per}'] = self._calculate_sma(df_features, self.sma_per, 'Daily_Return')

            # 2. Осцилляторы & импульс (RSI от доходностей работает как нормированный импульс)
            df_features[f'RSI_{self.rsi_per}'] = self._calculate_rsi(df_features, self.rsi_per, 'Daily_Return')

            # MACD доходностей
            macd, signal = self._calculate_macd(df_features, self.ema_fast, self.ema_slow, self.macd_sig, 'Daily_Return')
            df_features['MACD'] = macd
            df_features['MACD_Signal'] = signal
            df_features['MACD_Hist'] = macd - signal

            # 3. Волатильность & Границы (Bollinger Bands строим вокруг доходностей)
            upper, lower = self._calculate_bollinger_bands(df_features, self.bb_per, self.bb_std, 'Daily_Return')
            df_features['BB_Position'] = (df_features['Daily_Return'] - lower) / (upper - lower).replace(0, 1e-8)
            df_features['BB_Position'] = df_features['BB_Position'].fillna(0.5)

            # 4. Относительные фичи доходностей
            df_features['EMA_spread'] = df_features[f'EMA_{self.ema_fast}'] - df_features[f'EMA_{self.ema_slow}']

            # 5. Историческая волатильность
            df_features['Volatility'] = df_features['Daily_Return'].rolling(window=self.sma_per).std().fillna(0)

            # 6. Трейдинговые объемы и Сила тренда
            vol_column = 'Volume' if 'Volume' in df_features.columns else 'volume'

            # OBV и Чайкин считаем по классике от Close, но нормируем их изменения
            raw_obv = self._calculate_obv(df_features, self.col, vol_column)
            df_features['OBV_Slope'] = raw_obv.pct_change(periods=5).fillna(0).replace([np.inf, -np.inf], 0)

            df_features['Chaikin_Osc'] = self._calculate_chaikin_oscillator(df_features, self.col, vol_column)

            # Сила тренда (ADX)
            df_features['Trend_Strength'] = self._calculate_adx_simplified(df_features, self.adx_per, self.col)
            df_features['Trend_Strength'] = df_features['Trend_Strength'].fillna(0)

            # --- ЖЕСТКАЯ ЗАЧИСТКА АБСОЛЮТНЫХ ЦЕН (Защита от подглядывания) ---
            # Удаляем сырые ценовые колонки из датасета фич, чтобы они физически не попали в X модели
            # Примечание: Мы НЕ удаляем Daily_Return, так как он нужен для y_target в датасете
            cols_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'volume']
            cols_to_drop = [c for c in cols_to_drop if c in df_features.columns]
            df_features = df_features.drop(columns=cols_to_drop)

            # Проверка на бесконечные значения (inf)
            inf_counts = np.isinf(df_features).sum().sum()
            if inf_counts > 0:
                logger.warning(f"Обнаружено {inf_counts} значений +/- c бесконечностью (inf). Производится замена на корректные числа.")
                df_features = df_features.replace([np.inf, -np.inf], 0)

            # Дропаем временные NaN, возникшие из-за rolling-окон
            df_clean = df_features.dropna()
            dropped_rows = initial_rows - len(df_clean)

            logger.info(
                f"Генерация фич завершена успешно. Удалено ценовых колонок: {cols_to_drop}. "
                f"Удалено строк с NaN: {dropped_rows}. Строк на выходе: {len(df_clean)}"
            )

            return df_clean

        except Exception as e:
            logger.error(f"Критический сбой при расчете математических признаков: {e}", exc_info=True)
            return pd.DataFrame()

# --- АВТОНОМНЫЙ ТЕСТ МОДУЛЯ ---
if __name__ == "__main__":
    from src.config import setup_logging

    # Инициализация логгера через конфигурацию
    setup_logging(level=logging.INFO)

    logger.info("=== Запуск FeatureExtractor в автономном режиме ===")

    # Путь к сырому базовому файлу цен из конфига
    file_path = config.DATA_FILE_PATH

    if os.path.exists(file_path):
        # Загрузка сохраненного DataLoader'ом Parquet
        base_df = pd.read_parquet(file_path)
        logger.info(f"Успешно загружен базовый файл: {file_path} (Размерность: {base_df.shape})")

        # Прогонка экстрактора
        extractor = FeatureExtractor()
        df_with_features = extractor.extract_features(base_df)

        print("\nПревью сгенерированных индикаторов (последние 3 строки):")
        # Вывод только новых сгенерированных колонок, чтобы не захламлять консоль базовыми OHLCV
        new_cols = [col for col in df_with_features.columns if col not in base_df.columns]
        print(df_with_features[new_cols].tail(3))
    else:
        logger.error(f"Базовый файл с ценами не найден по пути: {file_path}")
        logger.error("Сначала запусти модуль data_loader.py или main.py, чтобы скачать сырые данные.")