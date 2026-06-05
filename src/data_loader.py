import os
import pandas as pd
import yfinance as yf


def download_crypto_data(
    ticker: str = "BTC-USD", period: str = "2y", interval: str = "1d"
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
        # Определяем путь к папке data относительно корня проекта
        base_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )  # Корень проекта
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        file_path = os.path.join(data_dir, f"{ticker}_{interval}.parquet")

        # Сохраняем в parquet
        df.to_parquet(file_path, compression="snappy")

        print(f" Данные сохранены в файл: {file_path}")
        print(f"Размерность датасета: {df.shape[0]} строк, {df.shape[1]} колонок.")

        return df

    except Exception as e:
        print(f" Ошибка в data_loader: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Тестовый запуск напрямую из модуля
    download_crypto_data()