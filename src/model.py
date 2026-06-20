import logging
import torch
import torch.nn as nn
import pandas as pd
import os

# Импортируем параметры и функцию настройки логирования из конфигурационного файла
from src.config import MODEL_PARAMS, TRAINING_PARAMS, FEATURES_FILE_PATH, setup_logging

# Инициализируем логгер для текущего файла
logger = logging.getLogger(__name__)


class QuantiGRU(nn.Module):
    """
    Архитектура QuantiGRU для трехклассовой интрадей-классификации.
    Принимает 3D-тензор: [Batch_Size, Sequence_Length, Input_Size]
    Возвращает сырые логиты для 3 классов (Short, Flat, Long).
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size=3, dropout_rate=0.0):
        super(QuantiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Слой GRU для обработки временных рядов свечей
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )

        # Слой регуляризации Dropout для предотвращения переобучения на рыночном шуме
        self.dropout = nn.Dropout(dropout_rate)

        # Выходной линейный слой. Жестко фиксируем на 3 класса (логиты)
        # Внимание: Softmax здесь не нужен, так как он встроен в nn.CrossEntropyLoss
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Инициализируем скрытые состояния нулями
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Прямой проход через GRU
        out, _ = self.gru(x, h0)

        # Берем выход только с ПОСЛЕДНЕГО временного шага последовательности [batch, hidden_size]
        out = self.dropout(out[:, -1, :])

        # Возвращаем сырые логиты для расчета лосса или инференса
        return self.fc(out)


# --- АВТОНОМНЫЙ ТЕСТ АРХИТЕКТУРЫ ---
if __name__ == "__main__":
    setup_logging()

    logger.info("=== Запуск автономного теста архитектуры QuantiGRU ===")

    # Автоматически определяем INPUT_SIZE по актуальному файлу фич, чтобы исключить хардкод
    try:
        from src import config
        # Берем путь к тренировочным фичам, так как базовый FEATURES_FILE_PATH теперь промежуточный
        df_temp = pd.read_parquet(config.TRAIN_FEATURES_PATH)

        # Исключаем служебную колонку таргета, все остальное — фичи
        feature_cols = [col for col in df_temp.columns if col not in ['target', 'Date', 'date']]
        INPUT_SIZE = len(feature_cols)
        logger.info(f"Динамически определено количество признаков из датасета: {INPUT_SIZE}")
    except Exception as e:
        INPUT_SIZE = 15  # Наш жесткий базис
        logger.warning(f"Файл фич не найден ({e}). Используем дефолтный фолбэк: {INPUT_SIZE}")

    HIDDEN_SIZE = MODEL_PARAMS['hidden_size']
    NUM_LAYERS = MODEL_PARAMS['num_layers']
    OUTPUT_SIZE = MODEL_PARAMS['output_size']  # Должно быть равно 3
    DROPOUT_RATE = MODEL_PARAMS['dropout_rate']

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
        device = torch.device(TRAINING_PARAMS['device'] if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logger.info(f"Модель переведена на устройство: {device}")

        # Генерируем фейковый батч для проверки пропускной способности тензоров
        seq_len = MODEL_PARAMS['sequence_length']
        test_input = torch.randn(32, seq_len, INPUT_SIZE).to(device)
        logger.info(f"Сгенерирован тестовый 3D-тензор батча: {list(test_input.shape)}")

        # Прямой проход (Forward pass)
        test_output = model(test_input)

        logger.info("--- Результаты проверки размерностей ---")
        logger.info(f"Вход (Размер батча): {list(test_input.shape)}")
        logger.info(f"Выход модели (Логиты):   {list(test_output.shape)} -> Успешно проверено на 3 класса!")
        logger.info("Автономный тест QuantiGRU успешно завершен!")

    except Exception as e:
        logger.error(f"Критическая ошибка во время теста модели: {e}", exc_info=True)