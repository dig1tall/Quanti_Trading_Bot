import logging
import numpy as np
import pandas as pd

from src.config import FEATURE_PARAMS

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Высокоэффективный интрадей Feature Engineering для BTC/ETH 1m.
    Итоговый набор из 15 изолированных признаков (консенсус ML-моделей).
    Архитектура полностью динамическая — все окна вынесены в конфигурационный файл.
    """

    def __init__(self):
        self.params = FEATURE_PARAMS
        self.col = self.params['target_column']  # Ожидаем 'Close'

    def _calculate_ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_sma(self, series, period):
        return series.rolling(window=period).mean()

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_rows = len(df)

        logger.info(
            "Запуск динамического интрадей Feature Engineering (Базис: 15 признаков)..."
        )

        try:
            df_f = pd.DataFrame(index=df.index)

            close = df[self.col]
            open_p = df['Open']
            high = df['High']
            low = df['Low']
            volume = df['Volume']

            # =====================================================
            # 1. RETURNS & MOMENTUM (5 признаков)
            # =====================================================
            logret_1 = np.log(close / close.shift(1))
            df_f['logret_1'] = logret_1

            # Мульти-горизонты логарифмических доходностей
            for h in self.params['ret_horizons']:
                if h != 1:
                    df_f[f'ret_{h}'] = np.log(close / close.shift(h))

            # =====================================================
            # 2. VOLATILITY & REGIME (4 признака)
            # =====================================================
            df_f['vol_fast'] = logret_1.rolling(self.params['vol_fast_period']).std()
            df_f['vol_slow'] = logret_1.rolling(self.params['vol_slow_period']).std()

            # Отношение волатильностей (Режимный индикатор шума)
            df_f['vol_ratio'] = df_f['vol_fast'] / df_f['vol_slow'].replace(0, 1e-8)

            # Волатильность Паркинсона (Размах внутрисвечевой борьбы)
            parkinson_raw = (np.log(high / low.replace(0, 1e-8)) ** 2) / (4 * np.log(2))
            df_f['parkinson_vol'] = parkinson_raw.rolling(self.params['parkinson_window']).mean()

            # =====================================================
            # 3. MEAN REVERSION & POSITION (3 признака)
            # =====================================================
            # Быстрый и медленный Price Z-Scores (Динамические имена фич)
            mean_fast = close.rolling(self.params['zscore_fast_period']).mean()
            std_fast = close.rolling(self.params['zscore_fast_period']).std()
            df_f['price_zscore_fast'] = (close - mean_fast) / std_fast.replace(0, 1e-8)

            mean_slow = close.rolling(self.params['zscore_slow_period']).mean()
            std_slow = close.rolling(self.params['zscore_slow_period']).std()
            df_f['price_zscore_slow'] = (close - mean_slow) / std_slow.replace(0, 1e-8)

            # Bollinger Position (Считается строго по Close)
            sma_bb = self._calculate_sma(close, self.params['bb_period'])
            std_bb = close.rolling(self.params['bb_period']).std()
            upper_bb = sma_bb + self.params['bb_std_dev'] * std_bb
            lower_bb = sma_bb - self.params['bb_std_dev'] * std_bb

            bb_pos = (close - lower_bb) / (upper_bb - lower_bb).replace(0, 1e-8)
            df_f['BB_Position'] = bb_pos.clip(-1, 2)

            # =====================================================
            # 4. TREND & STRUCTURE (2 признака)
            # =====================================================
            # Логарифмический EMA спред от цены
            ema_fast = self._calculate_ema(close, self.params['ema_fast_period'])
            ema_slow = self._calculate_ema(close, self.params['ema_slow_period'])
            df_f['EMA_spread'] = np.log(ema_fast / ema_slow.replace(0, 1e-8))

            # Направление давления внутри свечи (тело свечи)
            df_f['body_pct'] = (close - open_p) / open_p.replace(0, 1e-8)

            # =====================================================
            # 5. VOLUME DYNAMICS (1 признака)
            # =====================================================
            vol_mean = volume.rolling(self.params['volume_window']).mean()
            vol_std = volume.rolling(self.params['volume_window']).std()
            df_f['volume_zscore'] = (volume - vol_mean) / vol_std.replace(0, 1e-8)

            # =====================================================
            # CLEANING & DROPPING
            # =====================================================
            df_clean = df_f.replace([np.inf, -np.inf], np.nan).dropna()
            dropped_rows = initial_rows - len(df_clean)

            logger.info(
                f"Генерация фич успешно завершена.\n"
                f" -> Итого чистых признаков в матрице: {df_clean.shape[1]} (Ровно 15 без дубликатов)\n"
                f" -> Удалено начальных строк разгона: {dropped_rows}"
            )

            return df_clean

        except Exception as e:
            logger.error(f"Ошибка в модуле FeatureExtractor: {e}", exc_info=True)
            return pd.DataFrame()