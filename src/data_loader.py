import os
import pandas as pd
import yfinance as yf
from src import config

#ticker (Тикер) — это краткое уникальное название актива на бирже.

#interval (Таймфрейм) — это размер одной свечи (одной строчки в таблице).
#1d (one day) означает, что одна строка в таблице — это итог одного торгового дня.
#Если поставить 1h (one hour), то yfinance скачает часовые свечи, и в таблице будет детальная история по каждому часу.

def download_crypto_data(
    ticker: str = config.TICKER, period: str = config.PERIOD, interval: str = config.INTERVAL
) -> pd.DataFrame:
    """Скачивает исторические данные с yfinance, чистит их и сохраняет в Parquet."""
    print(f" Запуск загрузки данных для {ticker} ({interval})...")

    try:
        # 1. Скачиваем данные
        raw_data = yf.download(ticker, period=period, interval=interval)

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

        print(f" Данные сохранены в файл: {config.DATA_FILE_PATH}")
        print(
            f"Размерность датасета: {df.shape[0]} строк, {df.shape[1]} колонок."
        )

        return df

    except Exception as e:
        print(f" Ошибка в data_loader: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    download_crypto_data()