import os
import logging
import pandas as pd

from src import config
from src.config import DATA_LOAD_PARAMS, FEATURES_FILE_PATH, SCALING_PARAMS
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
    """
    def __init__(self):
        # Извлечение параметров из конфигурации — теперь сразу видно, с чем работает бот
        self.ticker = DATA_LOAD_PARAMS['ticker']
        self.interval = DATA_LOAD_PARAMS['interval']
        self.period = DATA_LOAD_PARAMS['period']
        self.method = SCALING_PARAMS['method']

        # Инициализируем компоненты конвейера с явным пробросом зависимостей
        self.loader = DataLoader(ticker=self.ticker, interval=self.interval, period=self.period)
        self.validator = DataValidator()
        self.extractor = FeatureExtractor()
        self.scaler = Scaler(method=self.method)

    def run_pipeline(self) -> None:
        """Запускает полный цикл с раздельным масштабированием во избежание утечки данных."""
        logger.info("=== Запуск конвейера данных через DataEngine ===")

        df = self.loader.download_crypto_data()
        if df.empty:
            logger.error("Сбой на этапе загрузки данных. Пайплайн остановлен.")
            return

        self.loader.save_to_parquet(df)
        logger.info("Первая фаза проекта настроена и работает автономно.")

        logger.info("Запуск валидатора целостности данных...")
        self.validator.validate_dataset(df)

        logger.info("Запуск генерации математических признаков (ML-пайплайн)...")
        df_features = self.extractor.extract_features(df)
        if df_features.empty:
            logger.error("Сбой на этапе генерации признаков. Пайплайн остановлен.")
            return

        # ---- ПРАВИЛЬНЫЙ РАЗДЕЛЬНЫЙ СКЕЙЛИНГ ----
        logger.info("Разделение данных для честного масштабирования...")

        # Загружаем train_split из конфига обучения (или берем 0.8 по умолчанию)
        from src.config import TRAINING_PARAMS
        train_split = TRAINING_PARAMS.get('train_split', 0.8)

        # Хронологический раскол матрицы фич
        split_idx = int(len(df_features) * train_split)
        df_train = df_features.iloc[:split_idx].copy()
        df_val = df_features.iloc[split_idx:].copy()

        columns_to_exclude = ['Date', 'date', 'target_forward'] # Явно добавляем таргет в исключения!
        all_columns = [col for col in df_features.columns if col not in columns_to_exclude]

        # Обучаем скейлер и трансформируем ТОЛЬКО фичи
        self.scaler.fit(df_train, all_columns)
        df_train_scaled = self.scaler.transform(df_train)
        df_val_scaled = self.scaler.transform(df_val)

        train_path = os.path.join(config.DATA_DIR, "train_features.parquet")
        val_path = os.path.join(config.DATA_DIR, "val_features.parquet")

        df_train_scaled.to_parquet(train_path)
        df_val_scaled.to_parquet(val_path)

        logger.info(f"Масштабирование успешно применено. Train={len(df_train_scaled)} строк, Val={len(df_val_scaled)} строк.")
        logger.info(f"Данные сохранены в файлы:\n -> {train_path}\n -> {val_path}")

    def _save_features(self, df: pd.DataFrame) -> None:
        """Внутренний метод для сохранения итогового датасета по системному пути."""
        os.makedirs(config.DATA_DIR, exist_ok=True)
        df.to_parquet(FEATURES_FILE_PATH)

        logger.info(f"Датасет с фичами и скейлингом сохранен в отдельный файл: {FEATURES_FILE_PATH}")
        logger.info(f"Итоговый размер матрицы данных для нейросети: {df.shape}")


# --- АВТОНОМНЫЙ ТЕСТ ДИСПЕТЧЕРА ДАННЫХ ---
if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)

    logger.info("=== Запуск полного конвейера DataEngine в автономном режиме ===")
    engine = DataEngine()
    engine.run_pipeline()