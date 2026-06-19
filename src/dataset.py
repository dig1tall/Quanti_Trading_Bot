import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.config import MODEL_PARAMS, TRAINING_PARAMS, FEATURES_FILE_PATH

logger = logging.getLogger(__name__)

class CryptoDataset(Dataset):
    def __init__(self, file_path: str, sequence_length: int = 30, forward_horizon: int = 5):
        self.sequence_length = sequence_length
        self.forward_horizon = forward_horizon

        logger.info(f"Загрузка данных для PyTorch Датасета из: {file_path}")
        self.df = pd.read_parquet(file_path)

        if 'Date' in self.df.columns:
            self.df.set_index('Date', inplace=True)
        elif 'date' in self.df.columns:
            self.df.set_index('date', inplace=True)
        self.df = self.df.sort_index()

        # Гарантируем, что Daily_Return на месте
        if 'Daily_Return' not in self.df.columns:
            raise KeyError("Критическая ошибка: колонка 'Daily_Return' не найдена в датасете!")

        # 1. Вытаскиваем фичи (чистая история, без заглядывания вперед)
        # Колонку Daily_Return можно оставить как фичу, она ведь историческая
        self.X = self.df.values.astype(np.float32)

        # 2. Считаем таргет вручную прямо из массива доходностей.
        # Для индекса i нам нужно среднее доходностей на шагах от (i + 1) до (i + forward_horizon)
        returns = self.df['Daily_Return'].values
        self.targets = np.zeros_like(returns, dtype=np.float32)

        for i in range(len(returns) - forward_horizon):
            # Будущее окно доходностей строго ПОСЛЕ текущего шага i
            self.targets[i] = np.mean(returns[i + 1 : i + 1 + forward_horizon])

        logger.info(f"Датасет успешно инициализирован. Доступно строк: {len(self.df)}")

    def __len__(self) -> int:
        # Урезаем длину, чтобы окно фич + горизонт предсказания не вылетали за массив
        return len(self.df) - self.sequence_length - self.forward_horizon

    def __getitem__(self, idx: int):
        # Окно фич: от idx до idx + sequence_length - 1 (например, 30 свечей)
        X_window = self.X[idx : idx + self.sequence_length]

        # Таргет: привязан строго к ПОСЛЕДНЕЙ свече в этом окне.
        # Так как в __init__ мы посчитали target[i] как будущее для шага i,
        # то для последней свечи окна (индекс idx + self.sequence_length - 1)
        # значение targets[...] будет содержать среднее за СЛЕДУЮЩИЕ forward_horizon шагов.
        target_idx = idx + self.sequence_length - 1
        target_val = self.targets[target_idx]

        X_tensor = torch.tensor(X_window, dtype=torch.float32)
        y_tensor = torch.tensor([target_val], dtype=torch.float32)

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