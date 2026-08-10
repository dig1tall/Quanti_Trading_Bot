"""Module for validating market historical data and ensuring temporal grid continuity."""

import os
import logging
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates market historical OHLCV data for continuity, NaNs, and price anomalies."""

    def __init__(self):
        """Initializes frequency mapping and target column configurations based on settings."""
        interval = config.DATA_LOAD_PARAMS['interval']
        if interval == '1m':
            self.freq = 'min'
        elif interval == '1d':
            self.freq = 'D'
        else:
            self.freq = None
            logger.warning(f"Timeframe '{interval}' has no explicit pandas frequency mapping. Grid validation is limited.")

        self.price_cols = ["Open", "High", "Low", "Close"]

    def validate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates dataframe for timestamp missing gaps, NaNs, and zero or negative prices.

            Args:
                df: Raw input dataframe containing OHLCV market data.

            Returns:
                pd.DataFrame: Validated and cleaned dataframe with a continuous timestamp index.
        """
        if df.empty:
            logger.error("Empty dataframe provided. Skipping validation.")
            return df

        logger.info(f"Starting market data validation. Initial shape: {df.shape}")
        df_fixed = df.copy()

        # sort dataframe by datetime index
        df_fixed = df_fixed.sort_index()

        # verify continuous frequency grid and handle small missing gaps
        if self.freq and isinstance(df_fixed.index, pd.DatetimeIndex):
            start_time = df_fixed.index.min()
            end_time = df_fixed.index.max()


            expected_index = pd.date_range(start=start_time, end=end_time, freq=self.freq, tz=df_fixed.index.tz)
            missing_steps = len(expected_index) - len(df_fixed)

            if missing_steps > 0:
                if missing_steps <= 3:
                    logger.warning(
                        f"Found {missing_steps} missing bars in time series. Auto-healing grid using reindex and forward fill... "
                    )
                    df_fixed = df_fixed.reindex(expected_index)
                    df_fixed = df_fixed.ffill()
                else:
                    # Если пропущено много данных, это критично для фичей (например, скользящих средних)
                    logger.error(f"Critical gap in historical data: {missing_steps} missing bars detected.")
                    raise ValueError(f"Corrupted history: missing {missing_steps} bars for timeframe {self.freq}.")
            else:
                logger.info("Time series grid is continuous without missing steps.")

        # drop any remaining missing values
        nan_counts = df_fixed.isna().sum().sum()
        if nan_counts > 0:
            logger.warning(f"Found {nan_counts} NaN values in dataset. Dropping invalid rows.")
            df_fixed = df_fixed.dropna()

        # check for invalid non-positive prices
        active_price_cols = [col for col in self.price_cols if col in df_fixed.columns]
        if active_price_cols:
            negative_prices = (df_fixed[active_price_cols] <= 0).sum().sum()
            if negative_prices > 0:
                logger.error(
                    f"Detected {negative_prices} non-positive or zero price entries."
                )
                raise ValueError(
                    f"Found {negative_prices} non-positive or zero price values."
                )

        # track volume anomalies for market activity logging
        zero_volumes = (df_fixed["Volume"] == 0).sum() if "Volume" in df_fixed.columns else 0
        if zero_volumes > 0:
            logger.info(f"Recorded {zero_volumes} candles with zero trading volume.")

        logger.info(f"Data validation passed successfully. Final dataset shape: {df_fixed.shape}")
        return df_fixed


if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)

    logger.info("Running DataValidator module test...")

    file_path = config.DATA_FILE_PATH
    if os.path.exists(file_path):
        base_df = pd.read_parquet(file_path)

        validator = DataValidator()
        cleaned_df = validator.validate_dataset(base_df)

        print("\nValidated dataframe index tail:")
        print(cleaned_df.tail(3))
    else:
        logger.error(f"Default data file not found: {file_path}")