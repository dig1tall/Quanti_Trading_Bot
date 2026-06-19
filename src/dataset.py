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

        if 'Daily_Return' not in self.df.columns:
            raise KeyError("Критическая ошибка: колонка 'Daily_Return' не найдена в датасете!")

        # 1. Вытаскиваем чистые исторические фичи
        self.X = self.df.values.astype(np.float32)

        # 2. Считаем сырые значения будущего изменения цены
        returns = self.df['Daily_Return'].values
        raw_targets = np.zeros_like(returns, dtype=np.float32)

        for i in range(len(returns) - forward_horizon):
            raw_targets[i] = np.mean(returns[i + 1 : i + 1 + forward_horizon])

        # Вычисляем квантили на основе всей выборки (отсекаем шум)
        # 35% самых сильных падений — Шорт, 35% самых сильных ростов — Лонг, остальное — АФК
        lower_threshold = np.quantile(raw_targets, 0.35)
        upper_threshold = np.quantile(raw_targets, 0.65)

        # Создаем массив меток классов (по умолчанию 1 — АФК/Боковик)
        self.targets = np.ones_like(raw_targets, dtype=np.int64)

        self.targets[raw_targets > upper_threshold] = 2  # Класс 2: Лонг
        self.targets[raw_targets < lower_threshold] = 0  # Класс 0: Шорт

        # Считаем баланс классов для логов
        unique, counts = np.unique(self.targets[:-forward_horizon], return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Распределение классов в датасете: {class_dist}")
        logger.info(f"Датасет успешно инициализирован. Доступно строк: {len(self.df)}")

    def __len__(self) -> int:
        return len(self.df) - self.sequence_length - self.forward_horizon

    def __getitem__(self, idx: int):
        X_window = self.X[idx : idx + self.sequence_length]

        # Берём класс, соответствующий последней свече в окне
        target_idx = idx + self.sequence_length - 1
        target_val = self.targets[target_idx]

        X_tensor = torch.tensor(X_window, dtype=torch.float32)
        # Для CrossEntropy таргет должен быть скаляром типа LongTensor
        y_tensor = torch.tensor(target_val, dtype=torch.long)

        return X_tensor, y_tensor

def get_separated_data_loaders(sequence_length: int, batch_size: int):
    """Используется в train.py для честного раздельного обучения"""
    import os
    from src import config

    train_path = os.path.join(config.DATA_DIR, "train_features.parquet")
    val_path = os.path.join(config.DATA_DIR, "val_features.parquet")

    train_dataset = CryptoDataset(train_path, sequence_length)
    val_dataset = CryptoDataset(val_path, sequence_length)

    logger.info(f"Изолированные датасеты для обучения: Train = {len(train_dataset)} окон, Validation = {len(val_dataset)} окон.")

    # shuffle=True только для обучения! Валидация идет строго хронологически
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader

def get_backtest_loader(sequence_length: int, batch_size: int):
    """Используется в backtest.py для инференса на валидационных данных"""
    import os
    from src import config

    val_path = os.path.join(config.DATA_DIR, "val_features.parquet")
    val_dataset = CryptoDataset(val_path, sequence_length)

    logger.info(f"Загрузка датасета для бэктестинга: {len(val_dataset)} окон.")

    # Для бэктеста shuffle строго False
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return val_loader