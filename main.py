import os
import logging

from src import config
from src.config import setup_logging
from src.config import DATA_LOAD_PARAMS, FEATURE_PARAMS
from src.data_loader import download_crypto_data
from src.check_data import validate_dataset
from src.features import extract_features

# Создаем логгер для текущего файла (main)
logger = logging.getLogger(__name__)

def main():
    # Инициализируем глобальные настройки логирования (вывод в консоль)
    setup_logging(level=logging.INFO)

    logger.info("=== Трейдинг-платформа Quanti: Сбор данных ===")

    # Извлекаем переменные из config для читаемости
    ticker = DATA_LOAD_PARAMS['ticker']
    interval = DATA_LOAD_PARAMS['interval']
    period = DATA_LOAD_PARAMS['period']

    # 1. Запускаем универсальный загрузчик, используя локальные переменные
    logger.info(f"Запуск DataLoader: загрузка {ticker} ({interval}, период: {period})...")
    df = download_crypto_data(
        ticker=ticker,
        interval=interval,
        period=period
    )

    # Безопасная проверка на пустоту
    if not df.empty:
        logger.info("Первая фаза проекта настроена и работает автономно.")
    else:
        logger.error("Сбой при инициализации базы данных. Скрипт остановлен.")
        return  # Прерываем выполнение, чтобы не ловить краш на расчёте фич

    # 2. Валидация датасета (теперь передаем df напрямую)
    logger.info("Запуск валидатора целостности данных...")
    validate_dataset(df)

    # 3. Генерация признаков (фич)
    logger.info("Запуск генерации математических признаков (ML-пайплайн)...")
    df_features = extract_features(df)

    # Выведем превью, чтобы глазами убедиться, что новые столбцы появились
    # Для вывода таблиц Pandas логгеры обычно используют чистый print или строку, сделаем красиво через логгер:
    logger.info("Новые признаки успешно сгенерированы. Превью датасета:\n%s", df_features.tail(3))

    # 4. Сохранение в НОВЫЙ файл в формате Parquet
    clean_ticker = ticker.replace('-', '_')
    features_filename = f"{clean_ticker}_{interval}_features.parquet"

    # Проверяем, существует ли папка data
    os.makedirs('data', exist_ok=True)
    save_path = os.path.join('data', features_filename)

    # Сохраняем расширенный датасет в бинарный Parquet
    df_features.to_parquet(save_path)
    logger.info(f"Датасет с фичами сохранен в отдельный файл: {save_path}")
    logger.info(f"Итоговый размер матрицы данных для нейросети: {df_features.shape}")

if __name__ == "__main__":
    main()