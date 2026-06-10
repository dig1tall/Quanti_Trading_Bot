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


class DataLoader:
    """
    Класс для загрузки, очистки и сохранения исторических рыночных данных.
    Разработан в соответствии с принципами чистой архитектуры (ООП).
    """
    def __init__(self, ticker: str = None, interval: str = None, period: str = None):
        self.ticker = ticker if ticker is not None else DATA_LOAD_PARAMS['ticker']
        self.interval = interval if interval is not None else DATA_LOAD_PARAMS['interval']
        self.period = period if period is not None else DATA_LOAD_PARAMS['period']
        self.required_cols = ["Open", "High", "Low", "Close", "Volume"]

    def download_crypto_data(self) -> pd.DataFrame:
        """Скачивает исторические данные с yfinance, чистит их и сохраняет в Parquet."""
        logger.info(f"Запуск загрузки данных для {self.ticker} ({self.interval}, период: {self.period})...")

        try:
            # 1. Скачиваем данные ( progress=False отключает системный прогресс-бар в stderr)
            raw_data = yf.download(self.ticker, period=self.period, interval=self.interval, progress=False)

            if raw_data.empty:
                raise ValueError(
                    f"Не удалось получить данные. Проверь тикер {self.ticker}."
                )

            df = raw_data.copy()

            # 2. Очистка структуры (убираем MultiIndex колонок, если yfinance его создал)
            df.columns = [
                col[0] if isinstance(col, tuple) else col for col in df.columns
            ]

            # Оставляем только классический OHLCV
            df = df[self.required_cols]

            # Удаляем пропуски (NaN), если они есть
            df = df.dropna()

            # 3. Сохранение
            self._save_to_parquet(df)

            return df

        except Exception as e:
            # exc_info=True автоматически прикрепит traceback ошибки к логу
            logger.error(f"Ошибка в data_loader при обработке {self.ticker}: {e}", exc_info=True)
            return pd.DataFrame()

    def _save_to_parquet(self, df: pd.DataFrame) -> None:
        """Внутренний изолированный метод для записи DataFrame на диск."""
        try:
            os.makedirs(config.DATA_DIR, exist_ok=True)

            # Сохраняем в parquet по готовому пути из config
            df.to_parquet(config.DATA_FILE_PATH, compression="snappy")

            logger.info(f"Данные сохранены в файл: {config.DATA_FILE_PATH}")
            logger.info(f"Размерность датасета: {df.shape[0]} строк, {df.shape[1]} колонок.")
        except Exception as e:
            logger.error(f"Не удалось сохранить Parquet файл на диск: {e}", exc_info=True)
            raise e


if __name__ == "__main__":
    from src.config import setup_logging
    # Если запускаем файл напрямую, инициализируем базовый логгер для тестов
    setup_logging(level=logging.INFO)

    loader = DataLoader()
    loader.download_crypto_data()