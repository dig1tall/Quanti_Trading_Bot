import os
import logging
import pandas as pd
import yfinance as yf

from src import config
from src.config import DATA_LOAD_PARAMS

# Инициализируем логгер для текущего модуля
logger = logging.getLogger(__name__)

# ticker (Тикер) — это краткое уникальное название актива на бирже.
# interval (Таймфрейм) — это размер одной свечи (одной строчки в таблице).

def download_crypto_data(
    ticker: str = DATA_LOAD_PARAMS['ticker'],
    period: str = DATA_LOAD_PARAMS['period'],
    interval: str = DATA_LOAD_PARAMS['interval']
) -> pd.DataFrame:
    """Скачивает исторические данные с yfinance, чистит их и сохраняет в Parquet."""
    logger.info(f"Запуск загрузки данных для {ticker} ({interval}, период: {period})...")

    try:
        # 1. Скачиваем данные ( progress=False отключает системный прогресс-бар в stderr)
        raw_data = yf.download(ticker, period=period, interval=interval, progress=False)

        if raw_data.empty:
            raise ValueError(
                f"Не удалось получить данные. Проверь тикер {ticker}."
            )

        df = raw_data.copy()

        # 2. Очистка структуры (убираем MultiIndex колонок, если yfinance его создал)
        df.columns = [
            col[0] if isinstance(col, tuple) else col for col in df.columns
        ]

        # Оставляем только классический OHLCV
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[required_cols]

        # Удаляем пропуски (NaN), если они есть
        df = df.dropna()

        # 3. Сохранение
        os.makedirs(config.DATA_DIR, exist_ok=True)

        # Сохраняем в parquet по готовому пути из config
        df.to_parquet(config.DATA_FILE_PATH, compression="snappy")

        logger.info(f"Данные сохранены в файл: {config.DATA_FILE_PATH}")
        logger.info(f"Размерность датасета: {df.shape[0]} строк, {df.shape[1]} колонок.")

        return df

    except Exception as e:
        # exc_info=True автоматически прикрепит traceback ошибки к логу
        logger.error(f"Ошибка в data_loader при обработке {ticker}: {e}", exc_info=True)
        return pd.DataFrame()


if __name__ == "__main__":
    from src.config import setup_logging
    # Если запускаем файл напрямую, инициализируем базовый логгер для тестов
    setup_logging(level=logging.INFO)
    download_crypto_data()