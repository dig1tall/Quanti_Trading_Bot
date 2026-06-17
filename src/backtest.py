import os
import logging
import torch
import numpy as np
import pandas as pd
import vectorbt as vbt
from datetime import datetime

from src.config import MODEL_PARAMS, TRAINING_PARAMS, FEATURES_FILE_PATH, PROJECT_ROOT
from src.dataset import get_data_loaders
from src.model import QuantiGRU

logger = logging.getLogger(__name__)

def run_backtest():
    logger.info("=== Запуск продвинутого бэктестинга Quanti через VectorBT ===")

    device = torch.device(TRAINING_PARAMS['device'] if torch.cuda.is_available() else 'cpu')

    # 1. Загружаем лоадеры
    _, val_loader = get_data_loaders(
        file_path=FEATURES_FILE_PATH,
        sequence_length=MODEL_PARAMS['sequence_length'],
        batch_size=TRAINING_PARAMS['batch_size'],
        train_split=TRAINING_PARAMS['train_split']
    )

    # Определяем размер входа
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

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    logger.info("Веса лучшей модели успешно загружены.")

    all_preds = []
    all_targets = []

    # 3. Собираем прогнозы модели
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)

            all_preds.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(y_batch.numpy().flatten())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # 4. ПОДГОТОВКА ДАННЫХ ДЛЯ VECTORBT
    # Нам нужны реальные цены закрытия для валидационного куска. Загружаем сырой файл.
    df = pd.read_parquet(FEATURES_FILE_PATH)

    # Вычисляем точную длину валидационной выборки, учитывая сдвиг на sequence_length
    # get_data_loaders отрезает train_split, остаток идет в validation.
    total_samples = len(df) - MODEL_PARAMS['sequence_length']
    val_size = len(targets) # Берем точно по размеру собранных таргетов

    # Вырезаем цены закрытия, которые соответствуют валидационным прогнозам
    # Они находятся в самом конце датасета
    val_close = df['Close'].iloc[-val_size:].reset_index(drop=True)

    # 5. ГЕНЕРАЦИЯ СИГНАЛОВ (Включая Шорты!)
    # Векторизованная логика:
    # Если прогноз > 0 -> 1 (Long)
    # Если прогноз < 0 -> -1 (Short)
    signals = np.where(preds > 0, 1, -1)

    # Превращаем в Pandas Series, чтобы VectorBT подтянул правильные индексы
    signals_series = pd.Series(signals, name='Quanti_Signals')

    # 6. МАГИЯ VECTORBT: ЗАПУСК СИМУЛЯЦИИ ПОРТФЕЛЯ
    logger.info("Запуск движка симуляции VectorBT Portfolio...")

    entries = signals_series == 1
    short_entries = signals_series == -1

    # Задаем комиссию биржи (например, 0.06% за сделку — стандарт для Binance Futures Taker)
    fee_rate = 0.0006

    portfolio = vbt.Portfolio.from_signals(
        close=val_close,
        entries=entries,                # Сигналы на открытие LONG
        exits=short_entries,            # Выход из LONG совпадает со входом в SHORT
        short_entries=short_entries,    # Сигналы на открытие SHORT
        short_exits=entries,            # Выход из SHORT совпадает со входом в LONG
        init_cash=10000.0,              # Стартовый депозит в $
        fees=fee_rate,                  # Учитываем комиссии
        freq='1D'                       # Частота данных
    )

    # 7. РАСЧЕТ И ВЫВОД МЕТРИК
    direction_correct = np.sum(np.sign(preds) == np.sign(targets)) / len(targets)

    logger.info("--- МЕТРИКИ VECTORBT (ВАЛИДАЦИОННЫЙ ПЕРИОД) ---")
    logger.info(f"Количество свечей в тесте:        {len(targets)}")
    logger.info(f"Accuracy направления знака:       {direction_correct * 100:.2f}%")

    # Печатаем весь дашборд VectorBT целиком без риска словить KeyError
    logger.info(f"\n{portfolio.stats().to_string()}")

    # 8. СОХРАНЕНИЕ ГРАФИКА (Опционально)
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Формируем имя файла: дата_время (например: 2026-06-17_20-35)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_filename = f"backtest_{current_time}.html"
    report_path = os.path.join(reports_dir, report_filename)

    # Генерируем и сохраняем интерактивный Plotly график
    fig = portfolio.plot()
    fig.write_html(report_path)

    logger.info(f"Интерактивный график бэктеста успешно сохранен в: reports/{report_filename}")

if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)
    run_backtest()