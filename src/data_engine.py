import os
import logging
import pandas as pd

from src import config
from src.config import DATA_LOAD_PARAMS, FEATURES_FILE_PATH, SCALING_PARAMS
from src.data_loader import DataLoader
from src.check_data import DataValidator
from src.features import FeatureExtractor
from src.data_preprocessing import Scaler

# Инициализируем логгер для модуля движка данных
logger = logging.getLogger(__name__)


class DataEngine:
    """
    Класс-оркестратор (Engine), управляющий всем конвейером обработки данных Quanti.
    Связывает DataLoader, DataValidator, FeatureExtractor и Scaler в единый пайплайн.
    """
    def __init__(self):
        # Явно извлекаем параметры из конфигурации — теперь сразу видно, с чем работает бот
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
        """Запускает полный цикл: загрузка -> сохранение сырых -> валидация -> фичи -> скейлинг -> сохранение финала."""
        logger.info("=== Запуск конвейера данных через DataEngine ===")

        # 1. Запуск загрузчика в память
        df = self.loader.download_crypto_data()

        if df.empty:
            logger.error("Сбой на этапе загрузки данных. Пайплайн остановлен.")
            return

        # Сохраняем сырой файл на диск
        self.loader.save_to_parquet(df)
        logger.info("Первая фаза проекта настроена и работает автономно.")

        # 2. Валидация датасета
        logger.info("Запуск валидатора целостности данных...")
        self.validator.validate_dataset(df)

        # 3. Генерация признаков (фич)
        logger.info("Запуск генерации математических признаков (ML-пайплайн)...")
        df_features = self.extractor.extract_features(df)

        if df_features.empty:
            logger.error("Сбой на этапе генерации признаков. Пайплайн остановлен.")
            return

        # 4. Масштабирование признаков (Scaling)
        logger.info("Запуск масштабирования ВСЕХ числовых признаков...")

        # Передаем абсолютно все колонки датафрейма в скейлер
        all_columns = list(df_features.columns)
        df_final = self.scaler.fit_transform(df_features, all_columns)
        logger.info(f"Масштабирование успешно применено к {len(df_final.columns)} признакам.")

        # Выводим превью финального датасета в лог
        logger.info("Итоговая матрица признаков подготовлена. Превью датасета:\n%s", df_final.tail(3))

        # 5. Сохранение в итоговый Parquet-файл для нейросети
        self._save_features(df_final)

    def _save_features(self, df: pd.DataFrame) -> None:
        """Внутренний метод для сохранения итогового датасета по системному пути."""
        os.makedirs(config.DATA_DIR, exist_ok=True)
        df.to_parquet(FEATURES_FILE_PATH)

        logger.info(f"Датасет с фичами и скейлингом сохранен в отдельный файл: {FEATURES_FILE_PATH}")
        logger.info(f"Итоговый размер матрицы данных для нейросети: {df.shape}")


# --- АВТОНОМНЫЙ ТЕСТ ОРКЕСТРАТОРА ---
if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)

    logger.info("=== Запуск полного конвейера DataEngine в автономном режиме ===")
    engine = DataEngine()
    engine.run_pipeline()