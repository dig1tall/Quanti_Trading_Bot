import os
import logging
import numpy as np
import pandas as pd

from src import config
from src.config import DATA_LOAD_PARAMS, FEATURES_FILE_PATH, SCALING_PARAMS, FEATURE_PARAMS
from src.data_loader import DataLoader
from src.check_data import DataValidator
from src.features import FeatureExtractor
from src.data_preprocessing import Scaler

# Инициализация логгера для модуля движка данных
logger = logging.getLogger(__name__)


class DataEngine:
    """
    Класс-диспетчер (Engine), управляющий всем конвейером обработки данных Quanti.
    Связывает DataLoader, DataValidator, FeatureExtractor и Scaler в единый пайплайн.
    Генерирует и размечает таргеты до разделения во избежание Data Leakage.
    """
    def __init__(self):
        self.ticker = DATA_LOAD_PARAMS['ticker']
        self.interval = DATA_LOAD_PARAMS['interval']
        self.period = DATA_LOAD_PARAMS['period']
        self.method = SCALING_PARAMS['method']

        # Инициализируем компоненты конвейера
        self.loader = DataLoader(ticker=self.ticker, interval=self.interval, period=self.period)
        self.validator = DataValidator()
        self.extractor = FeatureExtractor()
        self.scaler = Scaler(method=self.method)

    def run_pipeline(self) -> None:
        """Запускает полный цикл генерации признаков, таргета и раздельного масштабирования."""
        logger.info("=== Запуск конвейера данных через DataEngine ===")

        # 1. Загрузка данных
        df = self.loader.download_crypto_data()
        if df.empty:
            logger.error("Сбой на этапе загрузки данных. Пайплайн остановлен.")
            return

        self.loader.save_to_parquet(df)

        # 2. Валидация сырых данных
        logger.info("Запуск валидатора целостности данных...")
        self.validator.validate_dataset(df)

        # 3. Генерация признаков (ML-пайплайн)
        logger.info("Запуск генерации математических признаков...")
        df_features = self.extractor.extract_features(df)
        if df_features.empty:
            logger.error("Сбой на этапе генерации признаков. Пайплайн остановлен.")
            return

        if 'logret_1' not in df_features.columns:
            raise KeyError("Критическая ошибка: колонка 'Daily_Return' не найдена после FeatureExtractor!")

        # 4. Расчет сырого интрадей-таргета (Будущее скользящее среднее доходностей)
        forward_horizon = FEATURE_PARAMS.get('forward_horizon', 5)
        logger.info(f"Расчет сырых таргетов на горизонте {forward_horizon}м...")

        returns = df_features['logret_1'].values
        raw_targets = np.zeros_like(returns, dtype=np.float32)

        # Считаем среднюю доходность на n свечей вперед
        for i in range(len(returns) - forward_horizon):
            raw_targets[i] = np.mean(returns[i + 1 : i + 1 + forward_horizon])

        df_features['raw_target'] = raw_targets

        # ---- ПРАВИЛЬНЫЙ РАЗДЕЛЬНЫЙ СКЕЙЛИНГ И ТАРГЕТИНГ ----
        logger.info("Разделение данных для честной обработки (Train/Val)...")

        from src.config import TRAINING_PARAMS
        train_split = TRAINING_PARAMS.get('train_split', 0.8)

        # Хронологический раскол матрицы фич
        split_idx = int(len(df_features) * train_split)
        df_train = df_features.iloc[:split_idx].copy()
        df_val = df_features.iloc[split_idx:].copy()

        # 5. Разметка классов на основе КВАНТИЛЕЙ ТРЕЙНА (Защита от утечки будущего)
        logger.info("Вычисление порогов классов по обучающей выборке...")
        # Убираем последние n свечей из расчета квантилей трейна, так как на хвосте сырой таргет равен 0
        train_raw_targets_clean = df_train['raw_target'].values[:-forward_horizon]

        lower_threshold = float(np.quantile(train_raw_targets_clean, 0.35))
        upper_threshold = float(np.quantile(train_raw_targets_clean, 0.65))
        logger.info(f"Пороги классов зафиксированы: Шорт < {lower_threshold:.6f}, Лонг > {upper_threshold:.6f}")

        # Создаем пустые массивы для меток (1 — Флэт/Боковик по умолчанию)
        df_train['target'] = np.ones(len(df_train), dtype=np.int64)
        df_val['target'] = np.ones(len(df_val), dtype=np.int64)

        # Размечаем Train
        df_train.loc[df_train['raw_target'] > upper_threshold, 'target'] = 2  # Класс 2: Лонг
        df_train.loc[df_train['raw_target'] < lower_threshold, 'target'] = 0  # Класс 0: Шорт

        # Размечаем Validation СТРОГО по порогам из Train
        df_val.loc[df_val['raw_target'] > upper_threshold, 'target'] = 2
        df_val.loc[df_val['raw_target'] < lower_threshold, 'target'] = 0

        # Удаляем временную колонку сырого таргета, чтобы модель на ней случайно не обучилась
        df_train.drop(columns=['raw_target'], inplace=True)
        df_val.drop(columns=['raw_target'], inplace=True)

        # 6. Масштабирование признаков (Исключаем служебные колонки и таргет)
        columns_to_exclude = ['Date', 'date', 'target']
        cols_to_scale = [col for col in df_train.columns if col not in columns_to_exclude]

        # Обучаем скейлер только на Train, сохраняем параметры на диск и трансформируем обе выборки
        self.scaler.fit(df_train, cols_to_scale)
        self.scaler.save() # Запись scaler_params.json в папку models

        df_train_scaled = self.scaler.transform(df_train)
        df_val_scaled = self.scaler.transform(df_val)

        # 7. Сохранение итоговых изолированных файлов
        train_path = config.TRAIN_FEATURES_PATH
        val_path = config.VAL_FEATURES_PATH

        df_train_scaled.to_parquet(train_path)
        df_val_scaled.to_parquet(val_path)

        logger.info(f"Конвейер данных завершен успешно!")
        logger.info(f" -> Train датасет: {df_train_scaled.shape[0]} строк")
        logger.info(f" -> Val датасет:   {df_val_scaled.shape[0]} строк")
        logger.info(f"Данные сохранены в:\n -> {train_path}\n -> {val_path}")


# --- АВТОНОМНЫЙ ТЕСТ ДИСПЕТЧЕРА ДАННЫХ ---
if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)

    logger.info("=== Запуск полного конвейера DataEngine в автономном режиме ===")
    engine = DataEngine()
    engine.run_pipeline()