import os
import logging
import numpy as np
import pandas as pd

from src.config import FEATURE_PARAMS

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Высокоэффективный Feature Engineering для BTC/ETH, оптимизированный под дневные таймфреймы (1d).
    Генерирует 21 ортогональный, стационарный признак для глубоких GRU-сетей.
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

        logger.info("Запуск Feature Engineering: сборка 21 ортогонального признака под 1d таймфрейм...")

        try:
            df_f = pd.DataFrame(index=df.index)

            close = df[self.col]
            open_p = df['Open']
            high = df['High']
            low = df['Low']
            volume = df['Volume']

            # =====================================================
            # 1. RETURNS & MOMENTUM (Импульсы и ускорение)
            # =====================================================
            logret_1 = np.log(close / close.shift(1))
            df_f['logret_1'] = logret_1

            df_f['momentum_5'] = np.log(close / close.shift(5))
            df_f['momentum_20'] = np.log(close / close.shift(20))

            # [ДОБАВЛЕНО] Ускорение тренда (производная от импульса)
            df_f['momentum_accel'] = df_f['momentum_5'] - df_f['momentum_5'].shift(5)

            # =====================================================
            # 2. VOLATILITY & REGIME (Режимы волатильности)
            # =====================================================
            df_f['vol_fast'] = logret_1.rolling(self.params.get('vol_fast_period', 15)).std()
            df_f['vol_slow'] = logret_1.rolling(self.params.get('vol_slow_period', 60)).std()
            df_f['vol_ratio'] = df_f['vol_fast'] / df_f['vol_slow'].replace(0, 1e-8)

            # Волатильность Паркинсона
            parkinson_raw = (np.log(high / low.replace(0, 1e-8)) ** 2) / (4 * np.log(2))
            df_f['parkinson_vol'] = parkinson_raw.rolling(self.params.get('parkinson_window', 15)).mean()

            # [ДОБАВЛЕНО] Относительный ATR-14 (нормированный на текущую цену)
            true_range = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - close.shift(1)),
                    np.abs(low - close.shift(1))
                )
            )
            df_f['atr_14'] = true_range.rolling(14).mean() / close.replace(0, 1e-8)

            # =====================================================
            # 3. MEAN REVERSION & STRUCTURE (Структура свечи и тренда)
            # =====================================================
            # Bollinger Position
            sma_bb = self._calculate_sma(close, self.params.get('bb_period', 20))
            std_bb = close.rolling(self.params.get('bb_period', 20)).std()
            upper_bb = sma_bb + self.params.get('bb_std_dev', 2.0) * std_bb
            lower_bb = sma_bb - self.params.get('bb_std_dev', 2.0) * std_bb

            df_f['BB_Position'] = (close - lower_bb) / (upper_bb - lower_bb).replace(0, 1e-8)
            df_f['BB_Position'] = df_f['BB_Position'].clip(-1, 2)

            # Спред EMA
            ema_fast = self._calculate_ema(close, self.params.get('ema_fast_period', 12))
            ema_slow = self._calculate_ema(close, self.params.get('ema_slow_period', 26))
            df_f['EMA_spread'] = np.log(ema_fast / ema_slow.replace(0, 1e-8))

            # Геометрия свечи (Тело + [ИСПРАВЛЕНО] Диапазон)
            df_f['body_pct'] = (close - open_p) / open_p.replace(0, 1e-8)
            df_f['range_pct'] = (high - low) / close.replace(0, 1e-8)

            # [ДОБАВЛЕНО] Свечная асимметрия (wick_ratio)
            upper_wick = high - np.maximum(open_p, close)
            lower_wick = np.minimum(open_p, close) - low
            df_f['wick_ratio'] = (upper_wick - lower_wick) / (high - low).replace(0, 1e-8)

            # [ДОБАВЛЕНО] Макро-режим: Положение внутри годового диапазона (252 торговых дня)
            rolling_high = high.rolling(252).max()
            rolling_low = low.rolling(252).min()
            df_f['year_position'] = (close - rolling_low) / (rolling_high - rolling_low).replace(0, 1e-8)

            # =====================================================
            # 4. VOLUME & LIQUIDITY DYNAMICS (Объемы и давление)
            # =====================================================
            # Z-score объема за месяц (30 дней)
            vol_mean_30 = volume.rolling(30).mean()
            vol_std_30 = volume.rolling(30).std()
            df_f['volume_zscore'] = (volume - vol_mean_30) / vol_std_30.replace(0, 1e-8)

            # [ИСПРАВЛЕНО] Накопление объема (vol_pressure_7 через np.sign)
            direction = np.sign(logret_1)
            df_f['vol_pressure_7'] = (volume * direction).rolling(7).sum() / volume.rolling(30).sum().replace(0, 1e-8)

            # [ДОБАВЛЕНО] Объемный режим (всплеск интереса к активу)
            vol_fast_m = volume.rolling(7).mean()
            vol_slow_m = volume.rolling(60).mean()
            df_f['volume_regime'] = vol_fast_m / vol_slow_m.replace(0, 1e-8)

            # =====================================================
            # 5. МАРКРУТНЫЕ РЕЖИМЫ И ЦИКЛЫ (Энтропия и Время)
            # =====================================================
            # [ДОБАВЛЕНО] Энтропия направления тренда (trend_persistence)
            dir_binary = (logret_1 > 0).astype(int)
            df_f['trend_persistence'] = (dir_binary.rolling(10).mean() - 0.5).abs()

            # [НОВОЕ] Индекс макро-давления продавцов (Tail Risk Metric)
            # Отношение волатильности падений к общей волатильности
            down_returns = logret_1.clip(upper=0)
            df_f['downside_variance_ratio'] = down_returns.rolling(20).var() / logret_1.rolling(20).var().replace(0, 1e-8)

            # [НОВОЕ] Нелинейный импульс объема (Индекс капитуляции)
            df_f['v_momentum_ratio'] = (close.diff(3) * volume.rolling(3).mean()) / close.replace(0, 1e-8)

            # Календарные фичи
            if isinstance(df.index, pd.DatetimeIndex):
                day_of_week = df.index.dayofweek
                df_f['day_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
                df_f['day_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)


            # =====================================================
            # CLEANING & DROPPING
            # =====================================================
            # Из-за годового диапазона (rolling 252) первые 252 строки уйдут на разгон. Для дневок это норма.
            df_clean = df_f.replace([np.inf, -np.inf], np.nan).dropna()
            dropped_rows = initial_rows - len(df_clean)

            logger.info(
                f"Генерация фич успешно завершена.\n"
                f" -> Итого чистых ортогональных признаков: {df_clean.shape[1]}\n"
                f" -> Удалено строк разгона (из-за окна 252): {dropped_rows}"
            )

            return df_clean

        except Exception as e:
            logger.error(f"Ошибка в модуле FeatureExtractor: {e}", exc_info=True)
            return pd.DataFrame()