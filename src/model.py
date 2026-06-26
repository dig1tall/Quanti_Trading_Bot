import logging
import torch
import torch.nn as nn
import pandas as pd
import os

from src.config import MODEL_PARAMS, TRAINING_PARAMS, setup_logging

# Инициализируем логгер для текущего файла
logger = logging.getLogger(__name__)


class QuantiGRU(nn.Module):
    """
    Архитектура QuantiGRU для ТРИНАРНОЙ классификации дневных трендов (1d).
    Принимает 3D-тензор: [Batch_Size, Sequence_Length, Input_Size]
    Возвращает сырые логиты для 3 классов (Short, Flat, Long).
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size=3, dropout_rate=0.0):
        super(QuantiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size) # Теперь на выходе 3 логита

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# --- АВТОНОМНЫЙ ТЕСТ АРХИТЕКТУРЫ ---
if __name__ == "__main__":
    setup_logging()

    logger.info("=== Запуск автономного теста архитектуры QuantiGRU (Режим: 2 класса) ===")

    # Автоматически определяем INPUT_SIZE по актуальному файлу тренировочных фич
    try:
        from src import config
        df_temp = pd.read_parquet(config.TRAIN_FEATURES_PATH)

        # Исключаем служебные колонки, все остальное — фичи (должен быть 21 признак)
        feature_cols = [col for col in df_temp.columns if col not in ['target', 'Date', 'date']]
        INPUT_SIZE = len(feature_cols)
        logger.info(f"Динамически определено количество признаков из датасета: {INPUT_SIZE}")
    except Exception as e:
        INPUT_SIZE = 21  # Наш новый базис ортогональных фич под 1d
        logger.warning(f"Файл фич не найден ({e}). Используем дефолтный фолбэк: {INPUT_SIZE}")

    HIDDEN_SIZE = MODEL_PARAMS.get('hidden_size', 64)
    NUM_LAYERS = MODEL_PARAMS.get('num_layers', 2)
    OUTPUT_SIZE = MODEL_PARAMS.get('output_size', 2)  # Теперь ожидаем 2 класса
    DROPOUT_RATE = MODEL_PARAMS.get('dropout_rate', 0.2)

    try:
        # Инициализация модели
        model = QuantiGRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            output_size=OUTPUT_SIZE,
            dropout_rate=DROPOUT_RATE
        )
        logger.info(f"Модель успешно собрана: hidden_size={HIDDEN_SIZE}, num_layers={NUM_LAYERS}, dropout={DROPOUT_RATE}")

        # Проверяем доступность видеокарты
        device = torch.device(TRAINING_PARAMS.get('device', 'cpu') if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logger.info(f"Модель переведена на устройство: {device}")

        # Генерируем тестовый батч
        seq_len = MODEL_PARAMS.get('sequence_length', 30) # Например, берем историю за месяц (30 дней)
        test_input = torch.randn(32, seq_len, INPUT_SIZE).to(device)
        logger.info(f"Сгенерирован тестовый 3D-тензор батча: {list(test_input.shape)}")

        # Прямой проход (Forward pass)
        test_output = model(test_input)

        logger.info("--- Результаты проверки размерностей ---")
        logger.info(f"Вход (Размер батча): {list(test_input.shape)}")
        logger.info(f"Выход модели (Логиты):   {list(test_output.shape)} -> Успешно проверено на {OUTPUT_SIZE} класса!")

        if test_output.shape[1] != 2:
            logger.error(f"Размерность выхода {test_output.shape[1]} не равна 2! Проверь config.py")
        else:
            logger.info("Автономный тест QuantiGRU успешно завершен!")

    except Exception as e:
        logger.error(f"Критическая ошибка во время теста модели: {e}", exc_info=True)