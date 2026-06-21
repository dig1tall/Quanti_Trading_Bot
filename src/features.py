import os
import logging
import numpy as np
import pandas as pd

from src.config import FEATURE_PARAMS

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Высокоэффективный интрадей Feature Engineering для BTC/ETH 1m.
    Генерирует мощный, очищенный от коллинеарности набор признаков для GRU.
    """

    def __init__(self):
        self.params = FEATURE_PARAMS
        self.col = self.params.get('target_column', 'Close')

    def _calculate_ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_sma(self, series, period):
        return series.rolling(window=period).mean()

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_rows = len(df)

        logger.info("Запуск прокачанного интрадей Feature Engineering для QuantiGRU...")

        try:
            df_f = pd.DataFrame(index=df.index)

            close = df[self.col]
            open_p = df['Open']
            high = df['High']
            low = df['Low']
            volume = df['Volume']

            # =====================================================
            # 1. RETURNS & MOMENTUM (Очищено от дублей)
            # =====================================================
            logret_1 = np.log(close / close.shift(1))
            df_f['logret_1'] = logret_1

            # Вместо кучи ret_X оставляем только один среднесрочный импульс (momentum)
            df_f['momentum_5'] = np.log(close / close.shift(5))

            # =====================================================
            # 2. VOLATILITY & REGIME (Режимные индикаторы)
            # =====================================================
            df_f['vol_fast'] = logret_1.rolling(self.params.get('vol_fast_period', 15)).std()
            df_f['vol_slow'] = logret_1.rolling(self.params.get('vol_slow_period', 60)).std()
            df_f['vol_ratio'] = df_f['vol_fast'] / df_f['vol_slow'].replace(0, 1e-8)

            # Волатильность Паркинсона (Оставляем — это отличный признак скрытого зажатия цены)
            parkinson_raw = (np.log(high / low.replace(0, 1e-8)) ** 2) / (4 * np.log(2))
            df_f['parkinson_vol'] = parkinson_raw.rolling(self.params.get('parkinson_window', 15)).mean()

            # =====================================================
            # 3. MEAN REVERSION & STRUCTURE
            # =====================================================
            # Bollinger Position (Идеальный безразмерный осциллятор положения цены)
            sma_bb = self._calculate_sma(close, self.params.get('bb_period', 20))
            std_bb = close.rolling(self.params.get('bb_period', 20)).std()
            upper_bb = sma_bb + self.params.get('bb_std_dev', 2.0) * std_bb
            lower_bb = sma_bb - self.params.get('bb_std_dev', 2.0) * std_bb

            df_f['BB_Position'] = (close - lower_bb) / (upper_bb - lower_bb).replace(0, 1e-8)
            df_f['BB_Position'] = df_f['BB_Position'].clip(-1, 2)

            # Спред EMA (Показывает силу тренда без привязки к цене в баксах)
            ema_fast = self._calculate_ema(close, self.params.get('ema_fast_period', 12))
            ema_slow = self._calculate_ema(close, self.params.get('ema_slow_period', 26))
            df_f['EMA_spread'] = np.log(ema_fast / ema_slow.replace(0, 1e-8))

            # Относительное тело свечи
            df_f['body_pct'] = (close - open_p) / open_p.replace(0, 1e-8)

            # =====================================================
            # 4. VOLUME & LIQUIDITY DYNAMICS (Критическое усиление!)
            # =====================================================
            # Сила объема: Z-score объема на коротком окне
            vol_mean_f = volume.rolling(15).mean()
            vol_std_f = volume.rolling(15).std()
            df_f['volume_zscore_fast'] = (volume - vol_mean_f) / vol_std_f.replace(0, 1e-8)

            # Накопление объема (Аналог OBV, но в виде темпа прироста кумулятивного давления)
            # Показывает, куда давят крупные сайзы: на покупку или продажу
            direction = np.where(logret_1 > 0, 1, np.where(logret_1 < 0, -1, 0))
            directed_volume = volume * direction
            df_f['vol_pressure_5'] = directed_volume.rolling(5).sum() / vol_mean_f.replace(0, 1e-8)

            # =====================================================
            # 5. ИНТЕНСИВНОСТЬ РЫНКА (Микроструктура)
            # =====================================================
            # Отношение истинного диапазона хода к объему (Эффективность цены)
            # Помогает GRU отличать пустой ложный пробой от истинного импульса на объемах
            true_range = high - low
            df_f['market_efficiency'] = true_range / volume.replace(0, 1e-8)
            df_f['market_efficiency'] = df_f['market_efficiency'].rolling(5).mean()

            # =====================================================
            # CLEANING & DROPPING
            # =====================================================
            df_clean = df_f.replace([np.inf, -np.inf], np.nan).dropna()
            dropped_rows = initial_rows - len(df_clean)

            logger.info(
                f"Генерация фич успешно завершена.\n"
                f" -> Итого чистых признаков в матрице: {df_clean.shape[1]}\n"
                f" -> Удалено начальных строк разгона: {dropped_rows}"
            )

            return df_clean

        except Exception as e:
            logger.error(f"Ошибка в модуле FeatureExtractor: {e}", exc_info=True)
            return pd.DataFrame()