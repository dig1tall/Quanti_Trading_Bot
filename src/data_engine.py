import os
import logging
import json
import numpy as np
import pandas as pd

from src import config
from src.config import DATA_LOAD_PARAMS, SCALING_PARAMS, FEATURE_PARAMS
from src.data_loader import DataLoader
from src.check_data import DataValidator
from src.features import FeatureExtractor
from src.data_preprocessing import Scaler

logger = logging.getLogger(__name__)


class DataEngine:
    """
    Класс-диспетчер (Engine), управляющий конвейером обработки данных Quanti.
    Связывает компоненты и размечает бинарный таргет (Up/Down) без утечек данных.
    """
    def __init__(self):
        self.ticker = DATA_LOAD_PARAMS['ticker']
        self.interval = DATA_LOAD_PARAMS['interval']
        self.period = DATA_LOAD_PARAMS['period']
        self.method = SCALING_PARAMS['method']

        self.loader = DataLoader(ticker=self.ticker, interval=self.interval, period=self.period)
        self.validator = DataValidator()
        self.extractor = FeatureExtractor()
        self.scaler = Scaler(method=self.method)

    # Замени методы разметки внутри класса DataEngine в src/data_engine.py

    def run_pipeline(self) -> None:
        logger.info("=== Запуск конвейера данных через DataEngine (Режим: 1d, ТРИНАРНАЯ классификация) ===")

        # [Код шагов 1, 2, 3 остается без изменений до расчета таргета...]
        df = self.loader.download_crypto_data()
        self.loader.save_to_parquet(df)
        df = self.validator.validate_dataset(df)
        df_features = self.extractor.extract_features(df)

        # 4. Расчет сырого таргета (горизонт 1d)
        forward_horizon = FEATURE_PARAMS.get('forward_horizon', 1)
        logger.info(f"Расчет сырых таргетов на горизонте {forward_horizon}d...")

        if 'logret_1' not in df_features.columns:
            raise KeyError("Критическая ошибка: колонка 'logret_1' не найдена!")

        if forward_horizon == 1:
            df_features['raw_target'] = df_features['logret_1'].shift(-1)
        else:
            df_features['raw_target'] = df_features['logret_1'].shift(-1).rolling(window=forward_horizon).sum().shift(-(forward_horizon-1))

        df_features = df_features.dropna(subset=['raw_target']).copy()

        # 5. СТРОГАЯ ТРИНАРНАЯ РАЗМЕТКА КЛАССОВ
        flat_th = FEATURE_PARAMS.get('flat_threshold', 0.005)
        logger.info(f"Разметка трех классов по порогу флэта: +-{flat_th*100}%")

        # Заводим вектор нулей (по умолчанию всё Flat = 1)
        conditions = [
            (df_features['raw_target'] < -flat_th),                  # Класс 0: жесткое падение (Short)
            (df_features['raw_target'].abs() <= flat_th),             # Класс 1: боковик (Flat / Вне рынка)
            (df_features['raw_target'] > flat_th)                    # Класс 2: жесткий рост (Long)
        ]
        choices = [0, 1, 2]
        df_features['target'] = np.select(conditions, choices, default=1)
        df_features.drop(columns=['raw_target'], inplace=True)

        # ---- РАЗДЕЛЕНИЕ НА TRAIN / VAL ----
        logger.info("Разделение данных на Train/Val выборки...")
        from src.config import TRAINING_PARAMS
        train_split = TRAINING_PARAMS.get('train_split', 0.8)

        split_idx = int(len(df_features) * train_split)
        df_train = df_features.iloc[:split_idx].copy()
        df_val = df_features.iloc[split_idx:].copy()

        # Мониторинг баланса 3-х классов
        for name, dataset in [("Train", df_train), ("Val", df_val)]:
            counts = dataset['target'].value_counts(normalize=True).sort_index()
            logger.info(
                f"Баланс классов {name}: "
                f"Short(0): {counts.get(0,0):.2%}, "
                f"Flat(1): {counts.get(1,0):.2%}, "
                f"Long(2): {counts.get(2,0):.2%}"
            )

        # [Остальной код скейлинга и сохранения в файле остается прежним...]
        columns_to_exclude = ['Date', 'date', 'target']
        cols_to_scale = [col for col in df_train.columns if col not in columns_to_exclude]
        self.scaler.fit(df_train, cols_to_scale)
        self.scaler.save()
        df_train_scaled = self.scaler.transform(df_train)
        df_val_scaled = self.scaler.transform(df_val)
        df_train_scaled.to_parquet(config.TRAIN_FEATURES_PATH)
        df_val_scaled.to_parquet(config.VAL_FEATURES_PATH)
        logger.info("Конвейер данных успешно завершен!")