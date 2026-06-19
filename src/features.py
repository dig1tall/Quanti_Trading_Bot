import logging
import numpy as np
import pandas as pd
import os

from src import config
from src.config import FEATURE_PARAMS

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Модернизированный Feature Engineering пайплайн для 1-минутных свечей.
    Полностью исключает Data Leakage и абсолютные ценовые уровни.
    """
    def __init__(self):
        self.col      = FEATURE_PARAMS['target_column'] # Ожидается 'Close'
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

    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    def _calculate_rsi_from_returns(self, returns: pd.Series, period: int) -> pd.Series:
        """RSI, адаптированный для стационарных доходностей."""
        delta = returns.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return (rsi.fillna(50) - 50) / 100

    def _calculate_adx_from_returns(self, df_norm: pd.DataFrame, period: int) -> pd.Series:
        """Безопасный расчет ADX на базе нормализованных приращений High/Low."""
        high_ret = df_norm['High_dist']
        low_ret = df_norm['Low_dist']

        up_move = high_ret.diff()
        down_move = low_ret.diff()

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df_norm.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df_norm.index)

        plus_di = plus_dm.ewm(span=period, adjust=False).mean()
        minus_di = minus_dm.ewm(span=period, adjust=False).mean()

        denom = (plus_di + minus_di).replace(0, 1e-8)
        raw_dx = (plus_di - minus_di).abs() / denom
        adx = raw_dx.ewm(span=period, adjust=False).mean()
        return adx

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_rows = len(df)
        logger.info("Запуск очищенного Feature Engineering пайплайна (HFT 1m)...")

        try:
            df_features = pd.DataFrame(index=df.index)

            # 1. Генерируем базовый стационарный фундамент (Процентные изменения)
            df_features['Daily_Return'] = df[self.col].pct_change().fillna(0)

            # ФИКС: Объявляем ret сразу, чтобы использовать его в волатильности ниже
            ret = df_features['Daily_Return']

            # Нормируем High и Low относительно Close текущей свечи (в % выражении)
            df_norm = pd.DataFrame(index=df.index)
            df_norm['High_dist'] = (df['High'] - df[self.col]) / df[self.col]
            df_norm['Low_dist']  = (df['Low'] - df[self.col]) / df[self.col]

            # Вместо объемов берем волатильность разной длины
            df_features['Volatility_Fast'] = ret.rolling(window=10).std().fillna(0)
            df_features['Volatility_Slow'] = ret.rolling(window=60).std().fillna(0)

            # 2. РАСЧЕТ ФОРВАРДНОГО ТАРГЕТА


            # 3. ТРЕНДЫ И ИМПУЛЬС НА БАЗЕ ДОХОДНОСТЕЙ
            df_features[f'EMA_{self.ema_fast}'] = self._calculate_ema(ret, self.ema_fast)
            df_features[f'EMA_{self.ema_slow}'] = self._calculate_ema(ret, self.ema_slow)
            df_features[f'SMA_{self.sma_per}']  = self._calculate_sma(ret, self.sma_per)
            df_features['EMA_spread'] = df_features[f'EMA_{self.ema_fast}'] - df_features[f'EMA_{self.ema_slow}']

            # RSI от доходностей (масштабированный)
            df_features['RSI_Norm'] = self._calculate_rsi_from_returns(ret, self.rsi_per)

            # MACD от доходностей
            macd_line = df_features[f'EMA_{self.ema_fast}'] - df_features[f'EMA_{self.ema_slow}']
            signal_line = macd_line.ewm(span=self.macd_sig, adjust=False).mean()
            df_features['MACD_Hist'] = macd_line - signal_line

            # Волатильность (Rolling Standard Deviation)
            df_features['Volatility'] = ret.rolling(window=self.sma_per).std().fillna(0)

            # Положение внутри Полос Боллинджера (для доходностей)
            sma_ret = df_features[f'SMA_{self.sma_per}']
            std_ret = ret.rolling(window=self.bb_per).std()
            upper_bb = sma_ret + (std_ret * self.bb_std)
            lower_bb = sma_ret - (std_ret * self.bb_std)
            df_features['BB_Position'] = (ret - lower_bb) / (upper_bb - lower_bb).replace(0, 1e-8)
            df_features['BB_Position'] = df_features['BB_Position'].fillna(0.5)

            # Сила тренда из нормализованных приращений
            df_features['Trend_Strength'] = self._calculate_adx_from_returns(df_norm, self.adx_per).fillna(0)

            # Финальная чистка
            df_clean = df_features.dropna()
            df_clean = df_clean.replace([np.inf, -np.inf], 0)

            dropped_rows = initial_rows - len(df_clean)
            logger.info(f"Генерация фич завершена. Удалено строк с NaN/Inf: {dropped_rows}. Строк на выходе: {len(df_clean)}")

            return df_clean

        except Exception as e:
            logger.error(f"Критический сбой в FeatureExtractor: {e}", exc_info=True)
            return pd.DataFrame()