import logging
import numpy as np
import pandas as pd

from src.config import FEATURE_PARAMS

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Feature Engineering для BTC 5m.

    Без Data Leakage.
    Все признаки используют только прошлую информацию.
    """

    def __init__(self):
        self.params = FEATURE_PARAMS
        self.col = self.params['target_column']

    def _calculate_ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_sma(self, series, period):
        return series.rolling(window=period).mean()

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_rows = len(df)

        logger.info(
            "Запуск гибридного Feature Engineering (Stats + Tech)..."
        )

        try:
            df_f = pd.DataFrame(index=df.index)

            close = df[self.col]

            # =====================================================
            # БАЗОВАЯ ДОХОДНОСТЬ
            # =====================================================

            ret = close.pct_change()

            df_f['Daily_Return'] = ret

            # =====================================================
            # MULTI-HORIZON RETURNS
            # =====================================================

            for h in self.params['ret_horizons']:
                df_f[f'ret_{h}'] = close.pct_change(h)

            # =====================================================
            # VOLATILITY
            # =====================================================

            df_f['vol_fast'] = (
                ret
                .rolling(self.params['vol_fast_period'])
                .std()
            )

            df_f['vol_slow'] = (
                ret
                .rolling(self.params['vol_slow_period'])
                .std()
            )

            # =====================================================
            # PRICE Z-SCORE
            # =====================================================

            mean_30 = close.rolling(30).mean()
            std_30 = close.rolling(30).std()

            df_f['price_zscore_30'] = (
                (close - mean_30)
                / std_30.replace(0, 1e-8)
            )

            mean_120 = close.rolling(120).mean()
            std_120 = close.rolling(120).std()

            df_f['price_zscore_120'] = (
                (close - mean_120)
                / std_120.replace(0, 1e-8)
            )

            # =====================================================
            # RANGE FEATURES
            # =====================================================

            range_pct = (
                (df['High'] - df['Low'])
                / close.replace(0, 1e-8)
            )

            df_f['range_pct'] = range_pct

            df_f['range_ma'] = range_pct.rolling(24).mean()

            # =====================================================
            # EMA TREND
            # =====================================================

            ema_fast = self._calculate_ema(
                ret,
                self.params['ema_fast_period']
            )

            ema_slow = self._calculate_ema(
                ret,
                self.params['ema_slow_period']
            )

            df_f['EMA_spread'] = ema_fast - ema_slow

            # =====================================================
            # RSI
            # =====================================================

            delta = ret.diff()

            gain = (
                delta.clip(lower=0)
                .ewm(
                    span=self.params['rsi_period'],
                    adjust=False
                )
                .mean()
            )

            loss = (
                -delta.clip(upper=0)
                .ewm(
                    span=self.params['rsi_period'],
                    adjust=False
                )
                .mean()
            )

            rs = gain / loss.replace(0, 1e-8)

            rsi = 100 - (100 / (1 + rs))

            df_f['RSI_Norm'] = (rsi - 50) / 100

            # =====================================================
            # MACD
            # =====================================================

            macd_line = ema_fast - ema_slow

            signal_line = macd_line.ewm(
                span=self.params['macd_signal_period'],
                adjust=False
            ).mean()

            df_f['MACD_Hist'] = macd_line - signal_line

            # =====================================================
            # BOLLINGER POSITION
            # =====================================================

            sma_ret = self._calculate_sma(
                ret,
                self.params['sma_period']
            )

            std_ret = ret.rolling(
                self.params['bb_period']
            ).std()

            upper_bb = (
                sma_ret
                + self.params['bb_std_dev'] * std_ret
            )

            lower_bb = (
                sma_ret
                - self.params['bb_std_dev'] * std_ret
            )

            bb_pos = (
                (ret - lower_bb)
                / (upper_bb - lower_bb)
                .replace(0, 1e-8)
            )

            df_f['BB_Position'] = bb_pos.clip(-1, 2)

            # =====================================================
            # CLEANING
            # =====================================================

            df_clean = (
                df_f
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            dropped_rows = initial_rows - len(df_clean)

            logger.info(
                f"Генерация фич завершена. "
                f"Признаков: {df_clean.shape[1]}. "
                f"Удалено строк: {dropped_rows}"
            )

            return df_clean

        except Exception as e:
            logger.error(
                f"Ошибка FeatureExtractor: {e}",
                exc_info=True
            )
            return pd.DataFrame()