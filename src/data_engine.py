import os
import logging
import pandas as pd

from src.config import DATA_LOAD_PARAMS
from src.data_loader import DataLoader
from src.check_data import DataValidator
from src.features import FeatureExtractor

# Инициализируем логгер для модуля движка данных
logger = logging.getLogger(__name__)


class DataEngine:
    """
    Класс-оркестратор (Engine), управляющий всем конвейером обработки данных Quanti.
    Связывает DataLoader, DataValidator и FeatureExtractor в единый пайплайн.
    """
    def __init__(self):
        # Извлекаем параметры из конфигурации для внутренних нужд движка
        self.ticker = DATA_LOAD_PARAMS['ticker']
        self.interval = DATA_LOAD_PARAMS['interval']
        self.period = DATA_LOAD_PARAMS['period']

        # Инициализируем компоненты конвейера
        self.loader = DataLoader(ticker=self.ticker, interval=self.interval, period=self.period)
        self.validator = DataValidator()
        self.extractor = FeatureExtractor()

    def run_pipeline(self) -> None:
        """Запускает полный цикл: загрузка -> валидация -> генерация фич -> сохранение."""
        logger.info("=== Запуск конвейера данных через DataEngine ===")

        # 1. Запускаем универсальный загрузчик, используя локальные переменные
        df = self.loader.download_crypto_data()

        # Безопасная проверка на пустоту
        if df.empty:
            logger.error("Сбой на этапе загрузки данных. Пайплайн остановлен.")
            return

        logger.info("Первая фаза проекта настроена и работает автономно.")

        # 2. Валидация датасета (теперь передаем df напрямую)
        logger.info("Запуск валидатора целостности данных...")
        self.validator.validate_dataset(df)

        # 3. Генерация признаков (фич)
        logger.info("Запуск генерации математических признаков (ML-пайплайн)...")
        df_features = self.extractor.extract_features(df)

        if df_features.empty:
            logger.error("Сбой на этапе генерации признаков. Файл не будет сохранен.")
            return

        # Выведем превью, чтобы глазами убедиться, что новые столбцы появились
        # Для вывода таблиц Pandas логгеры обычно используют чистый print или строку, сделаем красиво через логгер:
        logger.info("Новые признаки успешно сгенерированы. Превью датасета:\n%s", df_features.tail(3))

        # 4. Сохранение в НОВЫЙ файл в формате Parquet
        self._save_features(df_features)

    def _save_features(self, df: pd.DataFrame) -> None:
        """Внутренний метод для формирования имени и сохранения итогового датасета."""
        clean_ticker = self.ticker.replace('-', '_')
        features_filename = f"{clean_ticker}_{self.interval}_features.parquet"

        # Проверяем, существует ли папка data
        os.makedirs('data', exist_ok=True)
        save_path = os.path.join('data', features_filename)

        # Сохраняем расширенный датасет в бинарный Parquet
        df.to_parquet(save_path)
        logger.info(f"Датасет с фичами сохранен в отдельный файл: {save_path}")
        logger.info(f"Итоговый размер матрицы данных для нейросети: {df.shape}")