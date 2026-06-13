import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src import config
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
        # Загрузка отскейленного parquet-файла, созданного DataEngine
        self.df = pd.read_parquet(file_path)

        # Если Date осталась в колонках, принудительная переноска её в индекс
        if 'Date' in self.df.columns:
            self.df.set_index('Date', inplace=True)
        elif 'date' in self.df.columns:
            self.df.set_index('date', inplace=True)

        # Сортировка строго по времени (индексу), чтобы не нарушить хронологию
        self.df = self.df.sort_index()

        # Выделение фич (все колонки) в виде numpy-массива для быстрого доступа
        self.features = self.df.values.astype(np.float32)

        # Нам нужно знать цену закрытия для расчета реального таргета.
        if 'Daily_Return' in self.df.columns:
            self.returns = self.df['Daily_Return'].values.astype(np.float32)
        else:
            raise KeyError("Критическая ошибка: колонка 'Daily_Return' не найдена в датасете!")

        logger.info(f"Датасет успешно инициализирован. Доступно строк: {len(self.df)}")

    def __len__(self) -> int:
        # Мы не можем брать окна, которые выходят за границу датасета (+1 свеча на таргет)
        return len(self.df) - self.sequence_length

    def __getitem__(self, idx: int):
        """
        Возвращает один элемент выборки по индексу.
        X: [sequence_length, num_features] -> история рынка
        y: [1] -> процент изменения цены на следующей свече
        """
        # 1. Нарезаем окно фич (например, с idx по idx+30)
        X = self.features[idx : idx + self.sequence_length]

        # 2. Считаем таргет
        target = self.returns[idx + self.sequence_length]

        # Превращаем всё в тензоры PyTorch
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor([target], dtype=torch.float32)

        return X_tensor, y_tensor


def get_data_loaders(file_path: str, sequence_length: int, batch_size: int, train_split: float = 0.8):
    """
    Загружает данные и делит их на Train и Validation строго ХРОНОЛОГИЧЕСКИ.
    Возвращает готовые DataLoader'ы для PyTorch.
    """
    # Инициализируем полный датасет
    full_dataset = CryptoDataset(file_path, sequence_length)
    total_samples = len(full_dataset)

    # Считаем точку раскола (никакого random_split! Только срез по времени)
    train_size = int(total_samples * train_split)
    val_size = total_samples - train_size

    logger.info(f"Хронологическое разделение: Train = {train_size} окон, Validation = {val_size} окон.")

    # ====Используем Subsets для создания подвыборок без перемешивания индексов
    train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, total_samples))

    # Создаем даталоадеры. Shuffle=False для временных рядов — это хорошая практика на валидации,
    # но на Train мы ставим Shuffle=True, чтобы модель не зазубривала глобальные циклы, а учила локальные паттерны окон.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader


# --- АВТОНОМНЫЙ ТЕСТ КОНВЕЙЕРА ДАННЫХ ---
if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)

    logger.info("=== Запуск автономного теста dataset.py ===")

    SEQ_LEN = MODEL_PARAMS['sequence_length']
    BATCH_SIZE = TRAINING_PARAMS['batch_size']
    TRAIN_SPLIT = TRAINING_PARAMS['train_split']

    try:
        # Проверяем, существует ли файл после работы DataEngine
        import os
        if not os.path.exists(FEATURES_FILE_PATH):
            logger.error(f"Файл {FEATURES_FILE_PATH} не найден! Сначала запусти data_engine.py")
            exit(1)

        # Пытаемся собрать даталоадеры
        train_loader, val_loader = get_data_loaders(
            file_path=FEATURES_FILE_PATH,
            sequence_length=SEQ_LEN,
            batch_size=BATCH_SIZE,
            train_split=TRAIN_SPLIT
        )

        # Вытаскиваем первый батч на проверку
        X_batch, y_batch = next(iter(train_loader))

        logger.info("--- Результаты проверки конвейера ---")
        logger.info(f"Размерность батча X (Features): {list(X_batch.shape)} -> [Батч, Окно, Фичи]")
        logger.info(f"Размерность батча y (Target):   {list(y_batch.shape)} -> [Батч, 1]")
        logger.info(f"Пример первого таргета в батче: {y_batch[0].item():.6f}")
        logger.info("Успех! Данные полностью готовы для передачи в QuantiGRU.")

    except Exception as e:
        logger.error(f"Ошибка при тестировании датасета: {e}", exc_info=True)