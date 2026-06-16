import os
import logging
import torch
import numpy as np
import pandas as pd

from src.config import MODEL_PARAMS, TRAINING_PARAMS, FEATURES_FILE_PATH, PROJECT_ROOT
from src.dataset import get_data_loaders
from src.model import QuantiGRU

logger = logging.getLogger(__name__)

def run_backtest():
    logger.info("=== Запуск модуля бэктестинга Quanti ===")

    device = torch.device(TRAINING_PARAMS['device'] if torch.cuda.is_available() else 'cpu')

    # 1. Загружаем лоадеры (нам нужен только val_loader)
    _, val_loader = get_data_loaders(
        file_path=FEATURES_FILE_PATH,
        sequence_length=MODEL_PARAMS['sequence_length'],
        batch_size=TRAINING_PARAMS['batch_size'],
        train_split=TRAINING_PARAMS['train_split']
    )

    # Автоматически определяем размер входа
    X_sample, _ = next(iter(val_loader))
    input_size = X_sample.shape[2]

    # 2. Инициализируем модель и загружаем сохраненные веса
    model = QuantiGRU(
        input_size=input_size,
        hidden_size=MODEL_PARAMS['hidden_size'],
        num_layers=MODEL_PARAMS['num_layers'],
        output_size=MODEL_PARAMS['output_size']
    ).to(device)

    model_path = os.path.join(PROJECT_ROOT, "models", "best_quanti_model.pth")
    if not os.path.exists(model_path):
        logger.error(f"Файл весов {model_path} не найден! Сначала обучи модель через train.py")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Веса лучшей модели успешно загружены. Запуск симуляции...")

    all_preds = []
    all_targets = []

    # 3. Собираем прогнозы модели по всей валидационной выборке
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)

            all_preds.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(y_batch.numpy().flatten())

    preds = np.array(all_preds)
    targets = np.array(all_targets) # Реальные Daily_Return следующих свечей

    # 4. Магия бэктеста: симулируем простейшую торговую стратегию
    # Если прогноз > 0, мы в позиции (Long = 1), если < 0, мы вне рынка (Cash = 0)
    signals = np.where(preds > 0, 1, 0)

    # Доходность нашей стратегии: если мы в позиции, получаем реальную доходность дня, если нет — 0.
    strategy_returns = signals * targets

    # Считаем кумулятивную (накопленную) доходность через сложные проценты
    # Формула: Prod(1 + return) - 1
    cum_strategy = np.prod(1.0 + strategy_returns) - 1.0
    cum_market = np.prod(1.0 + targets) - 1.0 # Стратегия "Купи и держи"

    # Считаем точность знака (угадал ли бот просто направление движения: вверх/вниз)
    direction_correct = np.sum(np.sign(preds) == np.sign(targets)) / len(targets)

    logger.info("--- Результаты бэктеста на валидационном периоде ---")
    logger.info(f"Количество дней для теста: {len(targets)}")
    logger.info(f"Точность предсказания направления (Accuracy): {direction_correct * 100:.2f}%")
    logger.info(f"Доходность стратегии 'Купи и держи' (Рынок): {cum_market * 100:.2f}%")
    logger.info(f"Доходность торгового бота Quanti:          {cum_strategy * 100:.2f}%")

    if cum_strategy > cum_market:
        logger.info(" Результат: Бот переиграл маркет-тренд!")
    else:
        logger.warning(" Результат: Бот уступил пассивному удержанию актива.")

if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)
    run_backtest()