import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Подтягиваем параметры по умолчанию из конфига
from src.config import MODEL_PARAMS, TRAINING_PARAMS

logger = logging.getLogger(__name__)

class CryptoDataset(Dataset):
    """
    Высокоэффективный PyTorch Dataset для дневных свечей (1d).
    Данные конвертируются в тензоры ОДИН раз при инициализации для максимальной скорости.
    """
    def __init__(self, file_path: str, sequence_length: int = 60):
        self.sequence_length = sequence_length

        logger.info(f"Загрузка данных для PyTorch Датасета из: {file_path}")
        df = pd.read_parquet(file_path)

        if 'target' not in df.columns:
            raise KeyError(f"Критическая ошибка: колонка 'target' не найдена в файле {file_path}!")

        # Сохраняем имена фич для будущей диагностики (SHAP / Feature Importance)
        self.feature_names = [col for col in df.columns if col not in ['target', 'Date', 'date']]

        # Перевод всего массива в PyTorch тензоры в RAM
        self.X = torch.tensor(df[self.feature_names].values, dtype=torch.float32)
        self.targets = torch.tensor(df['target'].values, dtype=torch.long)

        logger.info(
            f"Датасет успешно инициализирован.\n"
            f" -> Доступно строк (дней): {len(df)}\n"
            f" -> Итого скользящих окон: {self.__len__()}\n"
            f" -> Количество фич в векторе: {self.X.shape[1]}"
        )

        # Контроль распределения классов (0, 1, 2)
        unique, counts = np.unique(df['target'].values, return_counts=True)
        class_dist = {int(k): int(v) for k, v in zip(unique, counts)}
        logger.info(f" -> Распределение классов в файле (0:Short, 1:Flat, 2:Long): {class_dist}")

    def __len__(self) -> int:
        # Явно возвращаем количество возможных скользящих окон длины sequence_length
        return max(0, len(self.X) - self.sequence_length + 1)

    def __getitem__(self, idx: int):
        # Быстрый срез тензора прямо в памяти
        X_window = self.X[idx : idx + self.sequence_length]

        # Таргет берем на последней свече окна (DataEngine уже сместил его на 1d вперед)
        target_idx = idx + self.sequence_length - 1

        # Жесткий ассерт для защиты от look-ahead багов и выхода за границы
        assert target_idx < len(self.targets), f"Индекс таргета {target_idx} вышел за пределы массива {len(self.targets)}"

        y_target = self.targets[target_idx]

        return X_window, y_target


def get_separated_data_loaders(sequence_length: int = None, batch_size: int = None):
    """Используется в train.py для раздельного обучения и валидации."""
    from src import config

    # Если параметры не переданы явно, вытягиваем их из глобального конфига
    seq_len = sequence_length if sequence_length is not None else config.MODEL_PARAMS.get('sequence_length', 60)
    b_size = batch_size if batch_size is not None else config.TRAINING_PARAMS.get('batch_size', 32)

    train_dataset = CryptoDataset(config.TRAIN_FEATURES_PATH, seq_len)
    val_dataset = CryptoDataset(config.VAL_FEATURES_PATH, seq_len)

    logger.info(f"Изолированные датасеты: Train = {len(train_dataset)} окон, Validation = {len(val_dataset)} окон.")

    num_workers = config.TRAINING_PARAMS.get('num_workers', 0)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    # [ИСПРАВЛЕНО] shuffle=False для сохранения хронологической структуры рынка на дневках
    train_loader = DataLoader(
        train_dataset,
        batch_size=b_size,
        shuffle=False,
        drop_last=True, # Оставляем True, чтобы убрать неполный финальный батч, ломающий шаг градиента
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
    """Используется в backtest.py для инференса на валидационных данных."""
    from src import config

    seq_len = sequence_length if sequence_length is not None else config.MODEL_PARAMS.get('sequence_length', 60)
    b_size = batch_size if batch_size is not None else config.TRAINING_PARAMS.get('batch_size', 32)

    val_dataset = CryptoDataset(config.VAL_FEATURES_PATH, seq_len)
    num_workers = config.TRAINING_PARAMS.get('num_workers', 0)
    pin_memory = torch.cuda.is_available()

    logger.info(f"Загрузка датасета для бэктестинга: {len(val_dataset)} окон.")

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