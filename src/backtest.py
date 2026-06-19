import os
import logging
import torch
import numpy as np
import pandas as pd
import vectorbt as vbt
from datetime import datetime

from src import config
from src.config import MODEL_PARAMS, TRAINING_PARAMS, PROJECT_ROOT, DATA_FILE_PATH, BACKTEST_PARAMS
from src.dataset import get_backtest_loader
from src.model import QuantiGRU

logger = logging.getLogger(__name__)


def run_backtest():
    logger.info("=== Запуск продвинутого бэктестинга Quanti через VectorBT ===")

    device = torch.device(TRAINING_PARAMS['device'] if torch.cuda.is_available() else 'cpu')

    # 1. Загружаем лоадер (только валидационный датасет)
    val_loader = get_backtest_loader(
        sequence_length=MODEL_PARAMS['sequence_length'],
        batch_size=TRAINING_PARAMS['batch_size']
    )

    # Автоматически определяем размер входного вектора фич
    X_sample, _ = next(iter(val_loader))
    input_size = X_sample.shape[2]

    # 2. Инициализируем архитектуру и загружаем сохраненные веса модели
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

    all_pred_classes = []
    all_targets = []

    # 3. Инференс модели с применением АФК-фильтра по порогу уверенности
    logger.info("Запуск инференса модели для генерации сигналов...")
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)  # Сырые логиты [batch_size, 3]

            probs = torch.softmax(logits, dim=1)  # Считаем вероятности [batch_size, 3]
            max_probs, preds_classes = torch.max(probs, dim=1)

            preds_classes = preds_classes.cpu().numpy()
            max_probs = max_probs.cpu().numpy()

            # Если уверенность в предсказании Long/Short ниже порога — принудительно переводим в класс 1 (АФК)
            confidence_threshold = BACKTEST_PARAMS.get('threshold', 0.40)
            for i in range(len(preds_classes)):
                if max_probs[i] < confidence_threshold:
                    preds_classes[i] = 1

            all_pred_classes.extend(preds_classes)
            all_targets.extend(y_batch.numpy().flatten())

    final_classes = np.array(all_pred_classes)
    targets = np.array(all_targets)

    # 4. СИНХРОНИЗАЦИЯ ЦЕН ПО ИНДЕКСАМ ВАЛИДАЦИИ
    df_raw = pd.read_parquet(DATA_FILE_PATH)
    for date_col in ['Date', 'date']:
        if date_col in df_raw.columns:
            df_raw.set_index(date_col, inplace=True)
    df_raw = df_raw.sort_index()

    val_path = os.path.join(config.DATA_DIR, "val_features.parquet")
    df_features_file = pd.read_parquet(val_path)
    for date_col in ['Date', 'date']:
        if date_col in df_features_file.columns:
            df_features_file.set_index(date_col, inplace=True)
    df_features_file = df_features_file.sort_index()

    # Срез строго по количеству предсказаний с конца датасета
    val_dates = df_features_file.index[-len(targets):]
    val_close = df_raw.loc[val_dates, 'Close']
    logger.info(f"Цены для VectorBT успешно синхронизированы. Период: с {val_dates[0]} по {val_dates[-1]}")

    # 5. КОНВЕРТАЦИЯ КЛАССОВ В СИГНАЛЫ (2 -> Long, 0 -> Short)
    signals = np.zeros_like(final_classes, dtype=np.float32)
    signals[final_classes == 2] = 1.0
    signals[final_classes == 0] = -1.0

    signals_series = pd.Series(signals, name='Quanti_Signals', index=val_close.index)
    logger.info(f"Форма массива сигналов для VectorBT: {signals_series.shape}")
    logger.info(f"Проверка размерностей: Цены {val_close.shape} | Прогнозы {signals.shape}")

    # 6. СИМУЛЯЦИЯ ПОРТФЕЛЯ (Логика удержания позиций, Time-Stop)
    logger.info("Запуск движка симуляции VectorBT Portfolio со стоп-лоссами и Time-Stop...")

    raw_entries = (signals_series == 1).to_numpy()
    raw_exits = (signals_series == -1).to_numpy()

    time_stop_val = BACKTEST_PARAMS.get('time_stop', 60)

    entries = np.zeros_like(raw_entries, dtype=bool)
    exits = np.zeros_like(raw_exits, dtype=bool)
    short_entries = np.zeros_like(raw_entries, dtype=bool)
    short_exits = np.zeros_like(raw_exits, dtype=bool)

    in_long = False
    in_short = False
    bars_since_entry = 0

    for i in range(len(signals_series)):
        # Лонг-позиции
        if in_long:
            bars_since_entry += 1
            if bars_since_entry >= time_stop_val or raw_exits[i]:
                exits[i] = True
                in_long = False
                bars_since_entry = 0
        elif raw_entries[i] and not in_short:
            entries[i] = True
            in_long = True
            bars_since_entry = 0

        # Шорт-позиции
        if in_short:
            bars_since_entry += 1
            if bars_since_entry >= time_stop_val or raw_entries[i]:
                short_exits[i] = True
                in_short = False
                bars_since_entry = 0
        elif raw_exits[i] and not in_long:
            short_entries[i] = True
            in_short = True
            bars_since_entry = 0

    fee_rate = BACKTEST_PARAMS.get('fee_rate', 0.0006)
    init_cash = BACKTEST_PARAMS.get('init_cash', 10000.0)
    freq = BACKTEST_PARAMS.get('freq', '1m')

    sl_val = BACKTEST_PARAMS.get('stop_loss', None)
    tp_val = BACKTEST_PARAMS.get('take_profit', None)

    # Все булевы маски строго оборачиваются в Series с DatetimeIndex
    portfolio = vbt.Portfolio.from_signals(
        close=val_close,
        entries=pd.Series(entries, index=val_close.index),
        exits=pd.Series(exits, index=val_close.index),
        short_entries=pd.Series(short_entries, index=val_close.index),
        short_exits=pd.Series(short_exits, index=val_close.index),
        init_cash=init_cash,
        fees=fee_rate,
        freq=freq,
        sl_stop=sl_val,
        tp_stop=tp_val
    )

    # 7. РАСЧЕТ И ВЫВОД МЕТРИК КЛАССИФИКАЦИИ И ТОРГОВЛИ
    accuracy = np.sum(final_classes == targets) / len(targets)

    logger.info("--- МЕТРИКИ VECTORBT (ВАЛИДАЦИОННЫЙ ПЕРИОД) ---")
    logger.info(f"Количество свечей в тесте:        {len(targets)}")
    logger.info(f"Итоговый Accuracy моделей на тесте: {accuracy * 100:.2f}%")

    logger.info(f"\n{portfolio.stats().to_string()}")

    # 8. СОХРАНЕНИЕ ГРАФИКА РЕЗУЛЬТАТОВ
    if BACKTEST_PARAMS.get('save_plots', False):
        reports_dir = os.path.join(PROJECT_ROOT, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        report_filename = f"backtest_{current_time}.html"
        report_path = os.path.join(reports_dir, report_filename)

        fig = portfolio.plot()
        fig.write_html(report_path)

        logger.info(f"Интерактивный график бэктеста успешно сохранен в: reports/{report_filename}")
    else:
        logger.info("Сохранение графиков отключено в конфиге (save_plots=False).")


if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)
    run_backtest()