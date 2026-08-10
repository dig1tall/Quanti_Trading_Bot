"""Module for extracting orthogonal features optimized for daily cryptocurrency sequences."""

import logging
import numpy as np
import pandas as pd

from src.config import FEATURE_PARAMS

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts stationary and orthogonal features for deep neural network processing."""

    def __init__(self):
        """Initializes target column settings and parameters from configuration."""

        self.params = FEATURE_PARAMS
        self.col = self.params.get('target_column', 'Close')

    def _calculate_ema(self, series, period):
        """Calculates Exponential Moving Average across a pandas series."""

        return series.ewm(span=period, adjust=False).mean()

    def _calculate_sma(self, series, period):
        """Calculates Simple Moving Average across a pandas series."""

        return series.rolling(window=period).mean()

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates technical, volatility, volume, and calendar feature set from OHLCV data."""

        initial_rows = len(df)

        logger.info("Extracting feature set optimized for 1d timeframe...")

        try:
            df_f = pd.DataFrame(index=df.index)

            close = df[self.col]
            open_p = df['Open']
            high = df['High']
            low = df['Low']
            volume = df['Volume']

            # 1. returns & momentum metrics
            logret_1 = np.log(close / close.shift(1))
            df_f['logret_1'] = logret_1

            df_f['momentum_5'] = np.log(close / close.shift(5))
            df_f['momentum_20'] = np.log(close / close.shift(20))

            df_f['momentum_accel'] = df_f['momentum_5'] - df_f['momentum_5'].shift(5)

            # 2. volatility & regime metrics
            df_f['vol_fast'] = logret_1.rolling(self.params.get('vol_fast_period', 15)).std()
            df_f['vol_slow'] = logret_1.rolling(self.params.get('vol_slow_period', 60)).std()
            df_f['vol_ratio'] = df_f['vol_fast'] / df_f['vol_slow'].replace(0, 1e-8)

            parkinson_raw = (np.log(high / low.replace(0, 1e-8)) ** 2) / (4 * np.log(2))
            df_f['parkinson_vol'] = parkinson_raw.rolling(self.params.get('parkinson_window', 15)).mean()

            true_range = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - close.shift(1)),
                    np.abs(low - close.shift(1))
                )
            )
            df_f['atr_14'] = true_range.rolling(14).mean() / close.replace(0, 1e-8)

            # 3. structure & mean reversion metrics
            sma_bb = self._calculate_sma(close, self.params.get('bb_period', 20))
            std_bb = close.rolling(self.params.get('bb_period', 20)).std()
            upper_bb = sma_bb + self.params.get('bb_std_dev', 2.0) * std_bb
            lower_bb = sma_bb - self.params.get('bb_std_dev', 2.0) * std_bb

            df_f['BB_Position'] = (close - lower_bb) / (upper_bb - lower_bb).replace(0, 1e-8)
            df_f['BB_Position'] = df_f['BB_Position'].clip(-1, 2)

            ema_fast = self._calculate_ema(close, self.params.get('ema_fast_period', 12))
            ema_slow = self._calculate_ema(close, self.params.get('ema_slow_period', 26))
            df_f['EMA_spread'] = np.log(ema_fast / ema_slow.replace(0, 1e-8))

            df_f['body_pct'] = (close - open_p) / open_p.replace(0, 1e-8)
            df_f['range_pct'] = (high - low) / close.replace(0, 1e-8)

            upper_wick = high - np.maximum(open_p, close)
            lower_wick = np.minimum(open_p, close) - low
            df_f['wick_ratio'] = (upper_wick - lower_wick) / (high - low).replace(0, 1e-8)

            rolling_high = high.rolling(252).max()
            rolling_low = low.rolling(252).min()
            df_f['year_position'] = (close - rolling_low) / (rolling_high - rolling_low).replace(0, 1e-8)

            # 4. volume & liquidity dynamics
            vol_mean_30 = volume.rolling(30).mean()
            vol_std_30 = volume.rolling(30).std()
            df_f['volume_zscore'] = (volume - vol_mean_30) / vol_std_30.replace(0, 1e-8)

            direction = np.sign(logret_1)
            df_f['vol_pressure_7'] = (volume * direction).rolling(7).sum() / volume.rolling(30).sum().replace(0, 1e-8)

            vol_fast_m = volume.rolling(7).mean()
            vol_slow_m = volume.rolling(60).mean()
            df_f['volume_regime'] = vol_fast_m / vol_slow_m.replace(0, 1e-8)

            # 5. regime & cyclic features
            dir_binary = (logret_1 > 0).astype(int)
            df_f['trend_persistence'] = (dir_binary.rolling(10).mean() - 0.5).abs()

            down_returns = logret_1.clip(upper=0)
            df_f['downside_variance_ratio'] = down_returns.rolling(20).var() / logret_1.rolling(20).var().replace(0, 1e-8)

            df_f['v_momentum_ratio'] = (close.diff(3) * volume.rolling(3).mean()) / close.replace(0, 1e-8)

            if isinstance(df.index, pd.DatetimeIndex):
                day_of_week = df.index.dayofweek
                df_f['day_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
                df_f['day_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)


            # clean infinity and drop warmup NaN values
            df_clean = df_f.replace([np.inf, -np.inf], np.nan).dropna()
            dropped_rows = initial_rows - len(df_clean)

            logger.info(
                f"Feature extraction complete:\n"
                f" -> features extracted,: {df_clean.shape[1]}\n"
                f" -> initial rows dropped: {dropped_rows}"
            )

            return df_clean

        except Exception as e:
            logger.error(f"Error occurred during feature extraction: {e}", exc_info=True)
            return pd.DataFrame()