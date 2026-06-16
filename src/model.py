import logging
import torch
import torch.nn as nn
# Импортируем параметры и функцию настройки логирования из твоего конфигурационного файла
from src.config import MODEL_PARAMS, TRAINING_PARAMS, setup_logging

# Инициализируем логер для текущего файла
logger = logging.getLogger(__name__)

class QuantiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(QuantiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Слой GRU для обработки последовательности свечей
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # Линейный слой для финального прогноза
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Инициализируем начальную память нулями
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Прогоняем данные через GRU
        out, _ = self.gru(x, h0)

        # Берем выход только последней свечи из окна (индекс -1)
        out = out[:, -1, :]

        # Получаем финальный прогноз
        prediction = self.fc(out)

        return prediction

# Проверяем работу архитектуры через логи
if __name__ == "__main__":
    # Настраиваем глобальный логер из конфига
    setup_logging()

    logger.info("Запуск автономного теста архитектуры QuantiGRU...")

    # Вытаскиваем настройки из конфига
    INPUT_SIZE = 17
    HIDDEN_SIZE = MODEL_PARAMS['hidden_size']
    NUM_LAYERS = MODEL_PARAMS['num_layers']
    OUTPUT_SIZE = MODEL_PARAMS['output_size']

    try:
        # Создаем модель
        model = QuantiGRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
        logger.info(f"Модель успешно инициализирована: hidden_size={HIDDEN_SIZE}, num_layers={NUM_LAYERS}")

        # Проверяем устройство (cuda/cpu)
        device = torch.device(TRAINING_PARAMS['device'] if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logger.info(f"Модель переведена на устройство: {device}")

        # Тестовый тензор [батч, окно, фичи] отправляем на выбранное устройство
        seq_len = MODEL_PARAMS['sequence_length']
        test_input = torch.randn(32, seq_len, INPUT_SIZE).to(device)
        logger.debug(f"Сгенерирован тестовый тензор размерности: {test_input.shape}")

        # Прямой проход (forward pass)
        test_output = model(test_input)

        logger.info("--- Результаты теста ---")
        logger.info(f"Входной тензор (батч): {list(test_input.shape)}")
        logger.info(f"Выходной тензор модели: {list(test_output.shape)}")
        logger.info("Автономный тест модели успешно завершен без ошибок!")

    except Exception as e:
        logger.error(f"Ошибка во время прогона теста модели: {e}", exc_info=True)