"""Module for managing data pipeline execution, feature scaling, and target labeling."""

import logging
import numpy as np

from src import config
from src.config import DATA_LOAD_PARAMS, SCALING_PARAMS, FEATURE_PARAMS
from src.data_loader import DataLoader
from src.check_data import DataValidator
from src.features import FeatureExtractor
from src.data_preprocessing import Scaler

logger = logging.getLogger(__name__)


class DataEngine:
    """Manages raw data ingestion, feature generation, dataset splits, and scaling."""

    def __init__(self):
        """Initializes data engine components and configurations."""

        self.ticker = DATA_LOAD_PARAMS['ticker']
        self.interval = DATA_LOAD_PARAMS['interval']
        self.period = DATA_LOAD_PARAMS['period']
        self.method = SCALING_PARAMS['method']

        self.loader = DataLoader(ticker=self.ticker, interval=self.interval, period=self.period)
        self.validator = DataValidator()
        self.extractor = FeatureExtractor()
        self.scaler = Scaler(method=self.method)


    def run_pipeline(self) -> None:
        """Executes full data preparation pipeline and saves train, val, and test subsets."""

        logger.info(f"Starting data pipeline execution for {self.ticker} ({self.interval})...")

        df = self.loader.download_crypto_data()
        self.loader.save_to_parquet(df)
        df = self.validator.validate_dataset(df)
        df_features = self.extractor.extract_features(df)

        forward_horizon = FEATURE_PARAMS.get('forward_horizon', 1)
        logger.info(f"Calculating raw target returns for {forward_horizon}d horizon...")

        if 'logret_1' not in df_features.columns:
            raise KeyError("Required feature column 'logret_1' not found in dataset.")

        if forward_horizon == 1:
            df_features['raw_target'] = df_features['logret_1'].shift(-1)
        else:
            df_features['raw_target'] = df_features['logret_1'].shift(-1).rolling(window=forward_horizon).sum().shift(-(forward_horizon-1))

        df_features = df_features.dropna(subset=['raw_target']).copy()

        flat_th = FEATURE_PARAMS.get('flat_threshold', 0.005)
        logger.info(f"Applying ternary target label conditions (threshold: +/-{flat_th*100}%")

        # map raw forward returns into 3 categorical target classes
        conditions = [
            (df_features['raw_target'] < -flat_th),                  # Class 0: Short
            (df_features['raw_target'].abs() <= flat_th),            # Class 1: Flat
            (df_features['raw_target'] > flat_th)                    # Class 2: Long
        ]
        choices = [0, 1, 2]
        df_features['target'] = np.select(conditions, choices, default=1)
        df_features.drop(columns=['raw_target'], inplace=True)


        logger.info("Splitting dataset into train, validation, and test subsets...")
        from src.config import TRAINING_PARAMS
        train_split = TRAINING_PARAMS.get('train_split', 0.8)

        # slice datasets chronologically to preserve time sequence
        split_idx = int(len(df_features) * train_split)
        df_train = df_features.iloc[:split_idx].copy()
        df_train = df_train.iloc[:-1]       # drop last row due to target shift
        df_val = df_features.iloc[split_idx:].copy()

        # log target class distribution across splits
        for name, dataset in [("Train", df_train), ("Val", df_val)]:
            counts = dataset['target'].value_counts(normalize=True).sort_index()
            logger.info(
                f" class distribution {name}: "
                f"Short(0): {counts.get(0,0):.2%}, "
                f"Flat(1): {counts.get(1,0):.2%}, "
                f"Long(2): {counts.get(2,0):.2%}"
            )

        columns_to_exclude = ['Date', 'date', 'target']
        cols_to_scale = [col for col in df_train.columns if col not in columns_to_exclude]
        self.scaler.fit(df_train, cols_to_scale)
        self.scaler.save()
        df_train_scaled = self.scaler.transform(df_train)
        df_val_scaled = self.scaler.transform(df_val)
        df_train_scaled.to_parquet(config.TRAIN_FEATURES_PATH)
        df_val_scaled.to_parquet(config.VAL_FEATURES_PATH)
        logger.info("Data pipeline completed successfully.")