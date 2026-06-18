import os
import logging
import torch
import numpy as np
import pandas as pd
import vectorbt as vbt
from datetime import datetime

from src.config import MODEL_PARAMS, TRAINING_PARAMS, FEATURES_FILE_PATH, PROJECT_ROOT, DATA_FILE_PATH, BACKTEST_PARAMS
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

    # 4. ПОДГОТОВКА ДАННЫХ ДЛЯ VECTORBT (Синхронизация по датам)
    # Загружаем базовый файл с сырыми ценами и обязательно ставим дату в индекс, как в датасете
    df_raw = pd.read_parquet(DATA_FILE_PATH)
    for date_col in ['Date', 'date']:
        if date_col in df_raw.columns:
            df_raw.set_index(date_col, inplace=True)
    df_raw = df_raw.sort_index()

    # Загружаем итоговый файл фич, чтобы узнать точные даты валидационного окна
    df_features_file = pd.read_parquet(FEATURES_FILE_PATH)
    for date_col in ['Date', 'date']:
        if date_col in df_features_file.columns:
            df_features_file.set_index(date_col, inplace=True)
    df_features_file = df_features_file.sort_index()

    # Вычисляем точный срез индексов дат, которые попали в валидацию
    # CryptoDataset отрезает sequence_length окон с начала выборки
    total_samples = len(df_features_file) - MODEL_PARAMS['sequence_length']
    train_size = int(total_samples * TRAINING_PARAMS['train_split'])

    # Индексы валидации начинаются после train_size + sequence_length
    val_start_idx = train_size + MODEL_PARAMS['sequence_length']
    val_dates = df_features_file.index[val_start_idx : val_start_idx + len(targets)]

    # Вытаскиваем цены закрытия строго по этим датам
    val_close = df_raw.loc[val_dates, 'Close'].reset_index(drop=True)
    logger.info(f"Цены для VectorBT успешно синхронизированы по датам. Период: с {val_dates[0]} по {val_dates[-1]}")

    # 5. ГЕНЕРАЦИЯ СИГНАЛОВ (Включая Шорты!)
    # Задаем порог уверенности. Подбирается экспериментально (например, 0.002 = 0.2%)
    threshold = BACKTEST_PARAMS['threshold']

    signals = np.zeros_like(preds)
    signals[preds > threshold] = 1
    signals[preds < -threshold] = -1

    # Превращаем в Pandas Series
    signals_series = pd.Series(signals, name='Quanti_Signals')

    # 6. МАГИЯ VECTORBT: ЗАПУСК СИМУЛЯЦИИ ПОРТФЕЛЯ
    logger.info("Запуск движка симуляции VectorBT Portfolio со стоп-лоссами...")

    entries = signals_series == 1
    exits = signals_series == -1
    short_entries = signals_series == -1
    short_exits = signals_series == 1

    fee_rate = BACKTEST_PARAMS['fee_rate']
    init_cash = BACKTEST_PARAMS['init_cash']
    freq = BACKTEST_PARAMS['freq']

    sl_val = BACKTEST_PARAMS.get('stop_loss', None)
    tp_val = BACKTEST_PARAMS.get('take_profit', None)

    portfolio = vbt.Portfolio.from_signals(
        close=val_close,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=init_cash,
        fees=fee_rate,
        freq=freq,
        sl_stop=sl_val,         # Процент стоп-лосса (например, 0.02)
        tp_stop=tp_val          # Процент тейк-профита (например, 0.04)
    )

    # 7. РАСЧЕТ И ВЫВОД МЕТРИК
    direction_correct = np.sum(np.sign(preds) == np.sign(targets)) / len(targets)

    logger.info("--- МЕТРИКИ VECTORBT (ВАЛИДАЦИОННЫЙ ПЕРИОД) ---")
    logger.info(f"Количество свечей в тесте:        {len(targets)}")
    logger.info(f"Accuracy направления знака:       {direction_correct * 100:.2f}%")

    # Печатаем весь дашборд VectorBT целиком без риска словить KeyError
    logger.info(f"\n{portfolio.stats().to_string()}")

    # 8. СОХРАНЕНИЕ ГРАФИКА
    if BACKTEST_PARAMS.get('save_plots', False):
        reports_dir = os.path.join(PROJECT_ROOT, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        # Формируем имя файла: дата_время (например: 2026-06-18_01-47)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        report_filename = f"backtest_{current_time}.html"
        report_path = os.path.join(reports_dir, report_filename)

        # Генерируем и сохраняем интерактивный Plotly график
        fig = portfolio.plot()
        fig.write_html(report_path)

        logger.info(f"Интерактивный график бэктеста успешно сохранен в: reports/{report_filename}")
    else:
        logger.info("Сохранение графиков отключено в конфиге (save_plots=False).")

if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)
    run_backtest()