import os
import time
import logging
import pandas as pd
import ccxt
from datetime import datetime, timedelta, timezone

from src import config
from src.config import DATA_LOAD_PARAMS

# Инициализация логгера для текущего модуля
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Класс для загрузки, очистки и сохранения исторических рыночных данных через CCXT (Bybit).
    Оптимизирован под различные таймфреймы, включая дневные (1d).
    """
    def __init__(self, ticker: str = None, interval: str = None, period: str = None):
        self.ticker   = ticker if ticker is not None else DATA_LOAD_PARAMS['ticker']
        self.interval = interval if interval is not None else DATA_LOAD_PARAMS['interval']
        self.period   = period if period is not None else DATA_LOAD_PARAMS['period']
        self.required_cols = ["Open", "High", "Low", "Close", "Volume"]

        # Маппинг тикеров (если в конфиге BTC-USD, для биржи конвертируем в BTC/USDT)
        self.symbol = self.ticker.replace("-", "/")
        if "/" not in self.symbol:
            self.symbol = "BTC/USDT" # Фолбэк по умолчанию

    def _convert_period_to_days(self, period_str: str) -> int:
        """Вспомогательный метод для конвертации строк вида '30d', '60d', '1y' в количество дней."""
        try:
            if period_str.endswith('d'):
                return int(period_str[:-1])
            elif period_str.endswith('mo'):
                return int(period_str[:-2]) * 30
            elif period_str.endswith('y'):
                return int(period_str[:-1]) * 365
        except Exception as e:
            logger.warning(f"Ошибка парсинга периода '{period_str}', откат к дефолту 365d. Ошибка: {e}")
        return 365  # Для дневок дефолт в 1 год выглядит разумнее, чем 60 дней

    def download_crypto_data(self) -> pd.DataFrame:
        """Скачивает историю свечей с Bybit чанками, сохраняя структуру OHLCV."""
        logger.info(f"Запуск загрузки данных для {self.symbol} ({self.interval}, период: {self.period}) через CCXT...")

        try:
            # Инициализируем Bybit
            exchange = ccxt.bybit({
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}  # Бессрочные фьючерсы
            })

            # Вычисляем глубину истории
            days_to_download = self._convert_period_to_days(self.period)
            now = datetime.now(timezone.utc)
            start_date = now - timedelta(days=days_to_download)

            # Переводим в таймстампы в миллисекундах для биржи
            since_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(now.timestamp() * 1000)

            all_candles = []
            logger.info(f"Запрос истории с {start_date.strftime('%Y-%m-%d')} по {now.strftime('%Y-%m-%d')}...")

            while since_timestamp < end_timestamp:
                # Скачиваем порцию из 1000 свечей (для 1d это ~2.7 года за раз)
                candles = exchange.fetch_ohlcv(self.symbol, self.interval, since=since_timestamp, limit=1000)

                if not candles:
                    break

                all_candles.extend(candles)

                # Универсальное определение шага между свечами
                if len(candles) > 1:
                    step = candles[-1][0] - candles[-2][0]
                else:
                    # Если вернулась 1 свеча, шаг зависит от таймфрейма (дефолт 1d = 86400000 мс)
                    step = 86400000 if self.interval == '1d' else 60000

                last_candle_time = candles[-1][0]
                since_timestamp = last_candle_time + step

                # Логируем каждый чанк, так как для 1d их будет очень мало (обычно 1-2 запроса)
                current_pipeline_date = datetime.fromtimestamp(last_candle_time / 1000, tz=timezone.utc)
                logger.info(f"Загружено свечей: {len(all_candles)} | Последняя точка: {current_pipeline_date.strftime('%Y-%m-%d')}")

                # Пауза между запросами
                time.sleep(exchange.rateLimit / 1000)

            if not all_candles:
                raise ValueError(f"Биржа не вернула данные для {self.symbol}.")

            # Формируем DataFrame
            df = pd.DataFrame(all_candles, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['Date'], unit='ms', utc=True)
            df.set_index('Date', inplace=True)

            # Фильтруем дубликаты и сортируем по времени
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()

            # Оставляем только нужные колонки и дропаем NaN
            df = df[self.required_cols]
            df = df.dropna()

            return df

        except Exception as e:
            logger.error(f"Ошибка в data_loader при обработке {self.ticker}: {e}", exc_info=True)
            return pd.DataFrame()

    def save_to_parquet(self, df: pd.DataFrame) -> None:
        """Публичный изолированный метод для записи DataFrame."""
        if df.empty:
            logger.warning("Попытка сохранить пустой DataFrame. Пропускаем запись.")
            return
        try:
            os.makedirs(config.DATA_DIR, exist_ok=True)

            # Сохранение в parquet по готовому пути из config
            df.to_parquet(config.DATA_FILE_PATH, compression="snappy")

            logger.info(f"Данные сохранены в файл: {config.DATA_FILE_PATH}")
            logger.info(f"Размерность датасета: {df.shape[0]} строк, {df.shape[1]} колонок.")
        except Exception as e:
            logger.error(f"Не удалось сохранить Parquet файл на диск: {e}", exc_info=True)
            raise e


# --- АВТОНОМНЫЙ ТЕСТ МОДУЛЯ ---
if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)

    logger.info("=== Запуск DataLoader в автономном режиме ===")

    loader = DataLoader()
    df_raw = loader.download_crypto_data()

    if not df_raw.empty:
        print("\nПревью скачанных сырых данных:")
        print(df_raw.tail(3))
        loader.save_to_parquet(df_raw)
    else:
        logger.error("Тестовая загрузка завершилась сбоем.")