import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

class CryptoDataset(Dataset):
    """
    Высокоэффективный PyTorch Dataset для интрадей трейдинга.
    Максимально облегчен: принимает уже готовые, отскейленные фичи
    и размеченные таргеты из DataEngine. Занимается только нарезкой 3D-окон.
    """
    def __init__(self, file_path: str, sequence_length: int = 120):
        self.sequence_length = sequence_length

        logger.info(f"Загрузка данных для PyTorch Датасета из: {file_path}")
        self.df = pd.read_parquet(file_path)

        # Проверяем наличие колонки таргета, которую теперь генерирует DataEngine
        if 'target' not in self.df.columns:
            raise KeyError(f"Критическая ошибка: колонка 'target' не найдена в файле {file_path}!")

        # Отделяем матрицу признаков (все колонки, кроме таргета)
        feature_cols = [col for col in self.df.columns if col != 'target']

        # Переводим в numpy массивы для максимальной скорости __getitem__
        self.X = self.df[feature_cols].values.astype(np.float32)
        self.targets = self.df['target'].values.astype(np.int64)

        logger.info(
            f"Датасет успешно инициализирован.\n"
            f" -> Доступно строк: {len(self.df)}\n"
            f" -> Количество фич в векторе: {self.X.shape[1]}"
        )

        # Выводим баланс классов, чтобы контролировать перекосы
        unique, counts = np.unique(self.targets, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f" -> Распределение классов в файле: {class_dist}")

    def __len__(self) -> int:
        # Учитываем, что нам нужно seq_len свечей для окна фич,
        # и ОДНА следующая свеча, чтобы забрать у нее таргет.
        return len(self.df) - self.sequence_length

    def __getitem__(self, idx: int):
        X_window = self.X[idx : idx + self.sequence_length]

        # ИСПРАВЛЕНО: Таргет берем с той же строки, на которой закончилось окно фич,
        # потому что DataEngine УЖЕ записал в эту строку будущее изменение цены.
        target_idx = idx + self.sequence_length - 1
        target_val = self.targets[target_idx]

        X_tensor = torch.tensor(X_window, dtype=torch.float32)
        y_tensor = torch.tensor(target_val, dtype=torch.long)

        return X_tensor, y_tensor


def get_separated_data_loaders(sequence_length: int, batch_size: int):
    """Используется в train.py для честного раздельного обучения."""
    import os
    from src import config

    train_path = config.TRAIN_FEATURES_PATH
    val_path = config.VAL_FEATURES_PATH

    train_dataset = CryptoDataset(train_path, sequence_length)
    val_dataset = CryptoDataset(val_path, sequence_length)

    logger.info(f"Изолированные датасеты: Train = {len(train_dataset)} окон, Validation = {len(val_dataset)} окон.")

    # shuffle=True строго только для обучения, чтобы разбивать корреляцию последовательных батчей
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # Валидация идет строго хронологически без перемешивания для честного бэктеста
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader


def get_backtest_loader(sequence_length: int, batch_size: int):
    """Используется в backtest.py для инференса на валидационных данных."""
    import os
    from src import config

    val_path = os.path.join(config.DATA_DIR, "val_features.parquet")
    val_dataset = CryptoDataset(val_path, sequence_length)

    logger.info(f"Загрузка датасета для бэктестинга: {len(val_dataset)} окон.")

    # Для бэктеста shuffle строго False, drop_last=False (нельзя терять финальные свечи)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return val_loader