"""Module for downloading, processing, and storing historical OHLCV market data using CCXT."""

import os
import time
import logging
import pandas as pd
import ccxt
from datetime import datetime, timedelta, timezone

from src import config
from src.config import DATA_LOAD_PARAMS
from src.config import API_PARAMS

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles fetching and saving crypto historical candlestick data via exchange API."""

    def __init__(self, ticker: str = None, interval: str = None, period: str = None):
        """Initializes data loader configuration and exchange pair formatting."""

        self.ticker   = ticker if ticker is not None else DATA_LOAD_PARAMS['ticker']
        self.interval = interval if interval is not None else DATA_LOAD_PARAMS['interval']
        self.period   = period if period is not None else DATA_LOAD_PARAMS['period']
        self.required_cols = ["Open", "High", "Low", "Close", "Volume"]

        # format pair for exchange compliance
        self.symbol = self.ticker.replace("-", "/")
        if "/" not in self.symbol:
            self.symbol = "BTC/USDT" # Фолбэк по умолчанию

    def _convert_period_to_days(self, period_str: str) -> int:
        """Converts human-readable time period string into number of days."""
        try:
            if period_str.endswith('d'):
                return int(period_str[:-1])
            elif period_str.endswith('mo'):
                return int(period_str[:-2]) * 30
            elif period_str.endswith('y'):
                return int(period_str[:-1]) * 365
        except Exception as e:
            logger.warning(f"Error parsing period '{period_str}', defaulting to 365 days: {e}")
        return 365

    def download_crypto_data(self) -> pd.DataFrame:
        """Fetches OHLCV candlestick historical data from Bybit in chunked requests.

            Returns:
                pd.DataFrame: Sorted, deduplicated market data indexed by UTC Datetime.
        """

        logger.info(f"Initiating data download for {self.symbol} (interval: {self.interval}, period: {self.period})...")

        try:
            exchange = ccxt.bybit({
                'enableRateLimit': True,
            })
            if API_PARAMS.get('enable_demo', False):
                exchange.set_sandbox_mode(True)
                logger.info("CCXT sandbox mode enabled.")

            days_to_download = self._convert_period_to_days(self.period)
            now = datetime.now(timezone.utc)
            start_date = now - timedelta(days=days_to_download)

            since_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(now.timestamp() * 1000)

            all_candles = []
            logger.info(f"Requesting OHLCV from {start_date.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}...")

            while since_timestamp < end_timestamp:
                candles = exchange.fetch_ohlcv(self.symbol, self.interval, since=since_timestamp, limit=1000)

                if not candles:
                    break

                all_candles.extend(candles)

                if len(candles) > 1:
                    step = candles[-1][0] - candles[-2][0]
                else:
                    step = 86400000 if self.interval == '1d' else 60000

                last_candle_time = candles[-1][0]
                since_timestamp = last_candle_time + step

                current_pipeline_date = datetime.fromtimestamp(last_candle_time / 1000, tz=timezone.utc)
                logger.info(f"Fetched {len(all_candles)} total candles | Latest bar: {current_pipeline_date.strftime('%Y-%m-%d')}")

                time.sleep(exchange.rateLimit / 1000)

            if not all_candles:
                raise ValueError(f"Exchange returned no data for {self.symbol}.")

            df = pd.DataFrame(all_candles, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['Date'], unit='ms', utc=True)
            df.set_index('Date', inplace=True)

            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()

            df = df[self.required_cols]
            df = df.dropna()

            return df

        except Exception as e:
            logger.error(f"Error downloading data for {self.ticker}: {e}", exc_info=True)
            return pd.DataFrame()

    def save_to_parquet(self, df: pd.DataFrame) -> None:
        """Saves dataframe to snappy-compressed parquet file.

            Args:
                df: Market dataframe to write to disk.
        """
        if df.empty:
            logger.warning("Dataframe is empty. Skipping file write operation.")
            return
        try:
            os.makedirs(config.DATA_DIR, exist_ok=True)

            df.to_parquet(config.DATA_FILE_PATH, compression="snappy")

            logger.info(f"Market data successfully saved to {config.DATA_FILE_PATH}")
            logger.info(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns.")
        except Exception as e:
            logger.error(f"Failed to write parquet file to disk: {e}", exc_info=True)
            raise e


if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)

    logger.info("Running DataLoader in standalone mode...")

    loader = DataLoader()
    df_raw = loader.download_crypto_data()

    if not df_raw.empty:
        print("\nDownloaded raw dataframe tail:")
        print(df_raw.tail(3))
        loader.save_to_parquet(df_raw)
    else:
        logger.error("Test download failed.")