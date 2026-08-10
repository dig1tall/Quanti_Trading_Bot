"""Module for managing PyTorch Dataset and DataLoader instances for temporal sliding window sequences."""

import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from src import config

logger = logging.getLogger(__name__)

class CryptoDataset(Dataset):
    """PyTorch Dataset for daily cryptocurrency candles, converting data to tensors upon initialization."""

    def __init__(self, file_path: str, sequence_length: int = 60):
        """Initializes features and target matrices from parquet storage into memory."""

        self.sequence_length = sequence_length

        logger.info(f"Loading PyTorch dataset features from: {file_path}")
        df = pd.read_parquet(file_path)

        if 'target' not in df.columns:
            raise KeyError(f"Critical error: 'target' column not found in dataset file: {file_path}!")

        self.feature_names = [col for col in df.columns if col not in ['target', 'Date', 'date']]

        # retain feature column names for diagnostic and explainability tasks
        self.X = torch.tensor(df[self.feature_names].values, dtype=torch.float32)
        self.targets = torch.tensor(df['target'].values, dtype=torch.long)

        logger.info(
            f"Dataset initialized:\n"
            f" -> daily rows {len(df)}\n"
            f" -> sliding windows: {self.__len__()}\n"
            f" -> features per bar: {self.X.shape[1]}"
        )

        unique, counts = np.unique(df['target'].values, return_counts=True)
        class_dist = {int(k): int(v) for k, v in zip(unique, counts)}
        logger.info(f" -> Class distribution (0:Short, 1:Flat, 2:Long): {class_dist}")

    def __len__(self) -> int:
        """Calculates total count of valid sequence windows."""

        return max(0, len(self.X) - self.sequence_length + 1)

    def __getitem__(self, idx: int):
        """Retrieves a sliding sequence window and corresponding target label."""

        X_window = self.X[idx : idx + self.sequence_length]

        target_idx = idx + self.sequence_length - 1

        assert target_idx < len(self.targets), f"Target index {target_idx} exceeds matrix length {len(self.targets)}"

        y_target = self.targets[target_idx]

        return X_window, y_target


def get_separated_data_loaders(sequence_length: int = None, batch_size: int = None):
    """Creates isolated DataLoaders for train and validation dataset splits.

        Args:
            sequence_length: Optional sequence window length override.
            batch_size: Optional batch size override.

        Returns:
            tuple: (train_loader, val_loader, test_loader)
    """

    seq_len = sequence_length if sequence_length is not None else config.MODEL_PARAMS.get('sequence_length', 60)
    b_size = batch_size if batch_size is not None else config.TRAINING_PARAMS.get('batch_size', 32)

    train_dataset = CryptoDataset(config.TRAIN_FEATURES_PATH, seq_len)
    val_dataset = CryptoDataset(config.VAL_FEATURES_PATH, seq_len)

    logger.info(f"DataLoaders created: Train = {len(train_dataset)} windows, Validation = {len(val_dataset)} windows.")

    num_workers = config.TRAINING_PARAMS.get('num_workers', 0)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=b_size,
        shuffle=False,
        drop_last=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=b_size,
        shuffle=False,
        drop_last=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )

    return train_loader, val_loader


def get_backtest_loader(sequence_length: int = None, batch_size: int = None):
    """Creates a sequential DataLoader dedicated for inference and strategy backtesting."""

    seq_len = sequence_length if sequence_length is not None else config.MODEL_PARAMS.get('sequence_length', 60)
    b_size = batch_size if batch_size is not None else config.TRAINING_PARAMS.get('batch_size', 32)

    val_dataset = CryptoDataset(config.VAL_FEATURES_PATH, seq_len)
    num_workers = config.TRAINING_PARAMS.get('num_workers', 0)
    pin_memory = torch.cuda.is_available()

    logger.info(f"Initializing backtest inference DataLoader with {len(val_dataset)} windows.")

    val_loader = DataLoader(
        val_dataset,
        batch_size=b_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )
    return val_loader