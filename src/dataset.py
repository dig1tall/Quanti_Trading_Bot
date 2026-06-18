import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.config import MODEL_PARAMS, TRAINING_PARAMS, FEATURES_FILE_PATH

logger = logging.getLogger(__name__)

class CryptoDataset(Dataset):
    """
    Кастомный датасет PyTorch для работы со скользящими окнами временных рядов.
    Нарезает плоскую таблицу на трехмерные тензоры X и вектор ответов y.
    """
    def __init__(self, file_path: str, sequence_length: int = 30):
        self.sequence_length = sequence_length

        logger.info(f"Загрузка данных для PyTorch Датасета из: {file_path}")
        self.df = pd.read_parquet(file_path)

        if 'Date' in self.df.columns:
            self.df.set_index('Date', inplace=True)
        elif 'date' in self.df.columns:
            self.df.set_index('date', inplace=True)

        self.df = self.df.sort_index()
        if 'target_forward' not in self.df.columns:
            raise KeyError("Критическая ошибка: колонка 'target_forward' не найдена в датасете!")

        self.targets = self.df['target_forward'].values.astype(np.float32)

        # Исключаем лишнее, оставляем только чистые X фичи
        drop_cols = ['target_forward', 'Daily_Return', 'Date', 'date', 'Datetime']
        feature_cols = [col for col in self.df.columns if col not in drop_cols]

        self.X = self.df[feature_cols].to_numpy().astype(np.float32)
        self.y = self.df['target_forward'].to_numpy().astype(np.float32)

        if 'Daily_Return' in self.df.columns:
            self.returns = self.df['Daily_Return'].values.astype(np.float32)
        else:
            raise KeyError("Критическая ошибка: колонка 'Daily_Return' не найдена в датасете!")

        logger.info(f"Датасет успешно инициализирован. Доступно строк: {len(self.df)}")

    def __len__(self) -> int:
        return len(self.df) - self.sequence_length

    def __getitem__(self, idx: int):
        # ФИКС: Берем self.X вместо несуществующего self.features
        X = self.X[idx : idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length - 1]

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor([target], dtype=torch.float32)

        return X_tensor, y_tensor

def get_data_loaders(file_path: str, sequence_length: int, batch_size: int, train_split: float = 0.8):
    full_dataset = CryptoDataset(file_path, sequence_length)
    total_samples = len(full_dataset)

    train_size = int(total_samples * train_split)
    val_size = total_samples - train_size

    logger.info(f"Хронологическое разделение: Train = {train_size} окон, Validation = {val_size} окон.")

    train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, total_samples))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader