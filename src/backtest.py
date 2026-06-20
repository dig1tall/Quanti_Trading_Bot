import os
import logging
import torch
import numpy as np
import pandas as pd
import vectorbt as vbt
from datetime import datetime

from src import config
from src.config import MODEL_PARAMS, TRAINING_PARAMS, PROJECT_ROOT, BACKTEST_PARAMS
from src.dataset import get_backtest_loader
from src.model import QuantiGRU

logger = logging.getLogger(__name__)


def run_backtest():
    logger.info("=== Запуск промышленного бэктестинга Quanti Ver 1.1.0 через VectorBT ===")

    device = torch.device(TRAINING_PARAMS['device'] if torch.cuda.is_available() else 'cpu')

    # 1. Загружаем лоадер (только валидационный датасет)
    val_loader = get_backtest_loader(
        sequence_length=MODEL_PARAMS['sequence_length'],
        batch_size=TRAINING_PARAMS['batch_size']
    )

    # 2. Загрузка расширенного чекпоинта модели
    model_path = os.path.join(PROJECT_ROOT, "models", "best_quanti_model.pth")
    if not os.path.exists(model_path):
        logger.error(f"Критическая ошибка: файл весов {model_path} не найден! Сначала обучи сеть.")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Извлекаем сохранённые фичи и проверяем консистентность
    saved_features = checkpoint.get('feature_names', [])
    input_size = len(saved_features) if saved_features else val_loader.dataset.X.shape[1]

    logger.info(f"Загружен чекпоинт лучшей эпохи {checkpoint.get('epoch', 'unknown')}.")
    logger.info(f"Проверка фич модели: обнаружено {input_size} признаков.")

    # Инициализируем архитектуру и накатываем веса
    model = QuantiGRU(
        input_size=input_size,
        hidden_size=MODEL_PARAMS['hidden_size'],
        num_layers=MODEL_PARAMS['num_layers'],
        output_size=MODEL_PARAMS['output_size']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Веса нейросети успешно развёрнуты на устройстве.")

    all_raw_probs = []
    all_targets = []

    # 3. ФАЗА ИНФЕРЕНСА: Собираем чистые Softmax-вероятности
    logger.info("Запуск инференса модели для извлечения Softmax-вероятностей...")
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)

            probs = torch.softmax(logits, dim=1)
            all_raw_probs.append(probs.cpu().numpy())
            all_targets.extend(y_batch.numpy().flatten())

    raw_probs_matrix = np.vstack(all_raw_probs)  # Матрица размерностью [N_windows, 3]
    targets = np.array(all_targets)

    # 4. РЕАЛИЗАЦИЯ WEIGHTED TIME-DECAY VOTING И CONFIDENCE FILTER
    logger.info("Применение алгоритма сглаживания Weighted Time-Decay и Confidence Filter...")

    voting_window = BACKTEST_PARAMS.get('voting_window', 3)       # Окно накопления истории (минуты)
    decay_lambda = BACKTEST_PARAMS.get('decay_lambda', 0.2)        # Скорость затухания старых прогнозов
    confidence_threshold = BACKTEST_PARAMS.get('threshold', 0.46)  # Порог уверенности

    # Рассчитываем веса затухания для окна истории в явном виде
    if voting_window > 1:
        decay_weights = np.exp(-decay_lambda * np.arange(voting_window)[::-1])
        decay_weights /= np.sum(decay_weights) # Нормализуем в сумму = 1
    else:
        decay_weights = np.array([1.0])

    final_classes = np.ones(len(raw_probs_matrix), dtype=int) # По дефолту всё забиваем во Flat (1)

    # Двигаемся скользящим окном по вероятностям для агрегации истории
    for idx in range(len(raw_probs_matrix)):
        if idx < voting_window or voting_window <= 1:
            # Если истории ещё недостаточно или окно отключено, берём текущую точку «как есть»
            prob_vector = raw_probs_matrix[idx]
        else:
            # Вырезаем окно прошлых вероятностей [voting_window, 3]
            window_probs = raw_probs_matrix[idx - voting_window + 1 : idx + 1]
            # Взвешенное суммирование по оси времени
            prob_vector = np.dot(decay_weights, window_probs)

        # Confidence Filter: проверяем максимальную силу сигнала
        max_prob_class = np.argmax(prob_vector)
        if prob_vector[max_prob_class] >= confidence_threshold:
            final_classes[idx] = max_prob_class
        else:
            final_classes[idx] = 1 # Откат во Flat при неуверенности

    # 5. ХРОНОЛОГИЧЕСКАЯ СИНХРОНИЗАЦИЯ ЦЕН
    val_path = config.VAL_FEATURES_PATH
    df_features_file = pd.read_parquet(val_path)

    for date_col in ['Date', 'date']:
        if date_col in df_features_file.columns:
            df_features_file.set_index(date_col, inplace=True)
    df_features_file = df_features_file.sort_index()

    val_dates = df_features_file.index[-len(targets):]

    # Подгружаем ОРИГИНАЛЬНЫЕ сырые цены
    raw_data_path = os.path.join(config.DATA_DIR, "BTC-USDT_1m.parquet")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Критическая ошибка: сырой файл цен не найден по пути {raw_data_path}!")

    df_raw = pd.read_parquet(raw_data_path)

    for date_col in ['Date', 'date']:
        if date_col in df_raw.columns:
            df_raw.set_index(date_col, inplace=True)
    df_raw = df_raw.sort_index()

    try:
        val_close = df_raw.loc[val_dates, 'Close'].copy()
    except KeyError:
        logger.warning("Прямой матчинг индексов не удался. Применяем фолбэк по хронологическому срезу с конца.")
        val_close = df_raw['Close'].iloc[-len(targets):].copy()

    if val_close.isna().any():
        logger.warning(f"В синхронизированных ценах найдено {val_close.isna().sum()} NaN. Исправляем через ffill.")
        val_close = val_close.ffill().bfill()

    logger.info(f"Цены успешно синхронизированы из сырого кэша. Период: с {val_close.index[0]} по {val_close.index[-1]}")
    logger.info(f"Диапазон цен BTC на тесте: {val_close.min():.2f}$ - {val_close.max():.2f}$")

    # 6. ТРАНСФОРМАЦИЯ В СИГНАЛЫ ТОРГОВЛИ
    signals = np.zeros_like(final_classes, dtype=np.float32)
    signals[final_classes == 2] = 1.0  # Long
    signals[final_classes == 0] = -1.0 # Short

    signals_series = pd.Series(signals, name='Quanti_Signals', index=val_close.index)

    # 7. ДВИЖОК СИМУЛЯЦИИ ПОРТФЕЛЯ (ИСПРАВЛЕНО: Фильтр удержания min_hold_bars)
    logger.info("Конструирование масок торговых ордеров с фильтром минимального удержания...")
    raw_entries = (signals_series == 1).to_numpy()
    raw_exits = (signals_series == -1).to_numpy()

    time_stop_val = BACKTEST_PARAMS.get('time_stop', 120)
    min_hold_bars = BACKTEST_PARAMS.get('min_hold_bars', 20)  # Берем из лимитов удержания

    entries = np.zeros_like(raw_entries, dtype=bool)
    exits = np.zeros_like(raw_exits, dtype=bool)
    short_entries = np.zeros_like(raw_entries, dtype=bool)
    short_exits = np.zeros_like(raw_exits, dtype=bool)

    in_long = False
    in_short = False
    bars_since_entry = 0

    for i in range(len(signals_series)):
        # Обработка активного лонга
        if in_long:
            bars_since_entry += 1
            if (bars_since_entry >= min_hold_bars and raw_exits[i]) or bars_since_entry >= time_stop_val:
                exits[i] = True
                in_long = False
                bars_since_entry = 0
        # Обработка активного шорта
        elif in_short:
            bars_since_entry += 1
            if (bars_since_entry >= min_hold_bars and raw_entries[i]) or bars_since_entry >= time_stop_val:
                short_exits[i] = True
                in_short = False
                bars_since_entry = 0
        # Вход в позицию (только если мы свободны от сделок)
        else:
            if raw_entries[i]:
                entries[i] = True
                in_long = True
                bars_since_entry = 0
            elif raw_exits[i]:
                short_entries[i] = True
                in_short = True
                bars_since_entry = 0

    fee_rate = BACKTEST_PARAMS.get('fee_rate', 0.0006)
    slippage_rate = BACKTEST_PARAMS.get('slippage', 0.0002) # Добавили извлечение проскальзывания
    init_cash = BACKTEST_PARAMS.get('init_cash', 10000.0)
    freq = BACKTEST_PARAMS.get('freq', '1m')

    sl_val = BACKTEST_PARAMS.get('stop_loss', None)
    tp_val = BACKTEST_PARAMS.get('take_profit', None)

    # Инициализация VectorBT портфеля с учетом комиссий И проскальзывания
    portfolio = vbt.Portfolio.from_signals(
        close=val_close,
        entries=pd.Series(entries, index=val_close.index).astype(bool),
        exits=pd.Series(exits, index=val_close.index).astype(bool),
        short_entries=pd.Series(short_entries, index=val_close.index).astype(bool),
        short_exits=pd.Series(short_exits, index=val_close.index).astype(bool),
        init_cash=init_cash,
        fees=fee_rate,
        slippage=slippage_rate,  # ДОБАВЛЕНО УЧИТЫВАНИЕ ПРОСКАЛЬЗЫВАНИЯ
        freq=freq,
        sl_stop=sl_val,
        tp_stop=tp_val
    )

    # 8. РАСЧЕТ И ВЫВОД РЕЗУЛЬТАТОВ КВАНТ-ТЕСТА
    accuracy = np.sum(final_classes == targets) / len(targets)

    logger.info("=== ИТОГОВЫЕ МЕТРИКИ СИМУЛЯЦИИ VECTORBT ===")
    logger.info(f"Обработано свечей: {len(targets)}")
    logger.info(f"Точность сглаженного инференса (Accuracy): {accuracy * 100:.2f}%")
    logger.info(f"Количество сгенерированных сигналов Long: {np.sum(final_classes == 2)}")
    logger.info(f"Количество сгенерированных сигналов Short: {np.sum(final_classes == 0)}")
    logger.info(f"Количество отфильтрованных Flat: {np.sum(final_classes == 1)}")

    logger.info(f"\n{portfolio.stats().to_string()}")

    # 9. ЭКСПОРТ ГРАФИКОВ
    if BACKTEST_PARAMS.get('save_plots', False):
        reports_dir = os.path.join(PROJECT_ROOT, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        report_filename = f"backtest_{current_time}.html"
        report_path = os.path.join(reports_dir, report_filename)

        fig = portfolio.plot()
        fig.write_html(report_path)
        logger.info(f"Интерактивный дашборд бэктеста успешно сохранён в: reports/{report_filename}")


if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)
    run_backtest()