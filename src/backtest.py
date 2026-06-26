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
    logger.info("=== Запуск промышленного бэктестинга Quanti Ver 2.0.0 (Режим: 1d, ТРИНАРНЫЙ) ===")

    device = torch.device(TRAINING_PARAMS.get('device', 'cpu') if torch.cuda.is_available() else 'cpu')

    # 1. Загружаем лоадер
    val_loader = get_backtest_loader()

    # 2. Загрузка чекпоинта модели
    model_path = os.path.join(PROJECT_ROOT, "models", "best_quanti_model.pth")
    if not os.path.exists(model_path):
        logger.error(f"Критическая ошибка: файл весов {model_path} не найден! Сначала обучи сеть через train.py.")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    saved_features = checkpoint.get('feature_names', [])
    input_size = len(saved_features) if saved_features else val_loader.dataset.X.shape[1]

    logger.info(f"Успешно загружен чекпоинт лучшей эпохи {checkpoint.get('epoch', 'unknown')} (Macro F1: {checkpoint.get('val_macro_f1', 0):.4f}).")

    # Инициализируем архитектуру (output_size=3 подтянется из config автоматически)
    model = QuantiGRU(
        input_size=input_size,
        hidden_size=MODEL_PARAMS['hidden_size'],
        num_layers=MODEL_PARAMS['num_layers'],
        output_size=MODEL_PARAMS['output_size'],
        dropout_rate=MODEL_PARAMS['dropout_rate']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Веса тринарной нейросети развёрнуты.")

    all_raw_probs = []
    all_targets = []

    # 3. ФАЗА ИНФЕРЕНСА: Собираем вероятности классов (размерность output_size=3)
    logger.info("Запуск инференса модели на валидационной выборке...")
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)

            TEMPERATURE = 0.8  # Чем меньше, тем увереннее (полярнее) будут предсказания
            probs = torch.softmax(logits / TEMPERATURE, dim=1)
            all_raw_probs.append(probs.cpu().numpy())
            all_targets.extend(y_batch.numpy().flatten())

    raw_probs_matrix = np.vstack(all_raw_probs)  # [N_windows, 3]
    targets = np.array(all_targets)

    # 4. ИНТЕРПРЕТАЦИЯ СИГНАЛОВ (КЛАССИЧЕСКИЙ АБСОЛЮТНЫЙ ПОРОГ + ГИСТЕРЕЗИС)
    logger.info("Преобразование выходов сети в торговые сигналы (Фильтрация по абсолютному порогу)...")

    # Достаем наш классический порог (например, 0.55). Если забыл прописать, дефолт 0.50
    threshold_long = BACKTEST_PARAMS.get('threshold_long', 0.55)
    threshold_short = BACKTEST_PARAMS.get('threshold_short', 0.45)

    signals = np.zeros(len(raw_probs_matrix), dtype=np.float32)
    current_signal = 0.0 # Начинаем вне рынка (Flat)

    for idx in range(len(raw_probs_matrix)):
        prob_vector = raw_probs_matrix[idx]
        max_prob_class = np.argmax(prob_vector)
        max_prob = prob_vector[max_prob_class]

        # Определяем нужный порог динамически
        current_threshold = threshold_long if max_prob_class == 2 else threshold_short

        # Базовое желание модели на основе максимальной вероятности
        if max_prob_class == 2:
            target_signal = 1.0  # Хочет в Long
        elif max_prob_class == 0:
            target_signal = -1.0 # Хочет в Short
        else:
            target_signal = 0.0  # Хочет во Flat

        # --- ЛОГИКА АБСОЛЮТНОГО ПОРОГА С УДЕРЖАНИЕМ ТРЕНДА ---
        if current_signal == 0.0:
            if max_prob >= current_threshold:
                current_signal = target_signal
        else:
            # Логика выхода (смена позиции)
            if target_signal != current_signal:
                # Для выхода требуем чуть большую уверенность, чтобы не закрываться раньше времени
                if max_prob >= current_threshold:
                    current_signal = target_signal

        signals[idx] = current_signal

    logger.info(f"Фильтрация завершена. Сигналы стабилизированы.")

    # 5. ХРОНОЛОГИЧЕСКАЯ СИНХРОНИЗАЦИЯ ЦЕН
    val_path = config.VAL_FEATURES_PATH
    df_features_file = pd.read_parquet(val_path)

    for date_col in ['Date', 'date']:
        if date_col in df_features_file.columns:
            df_features_file.set_index(date_col, inplace=True)
    df_features_file = df_features_file.sort_index()

    val_dates = df_features_file.index[-len(targets):]

    raw_data_path = os.path.join(config.DATA_DIR, "BTC-USDT_1d.parquet")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Критическая ошибка: сырой файл дневных цен не найден по пути {raw_data_path}!")

    df_raw = pd.read_parquet(raw_data_path)

    for date_col in ['Date', 'date']:
        if date_col in df_raw.columns:
            df_raw.set_index(date_col, inplace=True)
    df_raw = df_raw.sort_index()

    try:
        val_close = df_raw.loc[val_dates, 'Close'].copy()
    except KeyError:
        logger.warning("Прямой матчинг индексов не удался. Применяем хронологический срез.")
        val_close = df_raw['Close'].iloc[-len(targets):].copy()

    if val_close.isna().any():
        val_close = val_close.ffill().bfill()

    logger.info(f"Цены успешно синхронизированы. Окон на тесте: {len(val_close)} | с {val_close.index[0]} по {val_close.index[-1]}")

    signals_series = pd.Series(signals, name='Quanti_Signals', index=val_close.index)

    # 6. КОНСТРУИРОВАНИЕ МАСОК ОРДЕРОВ ДЛЯ VECTORBT
    entries = (signals_series == 1.0)
    # Выходим из лонга, если модель явно говорит выйти в кэш (0) или перевернуться в шорт (-1)
    exits = (signals_series == 0.0) | (signals_series == -1.0)

    short_entries = (signals_series == -1.0)
    # Выходим из шорта, если модель уходит в кэш (0) или переворачивается в лонг (1)
    short_exits = (signals_series == 0.0) | (signals_series == 1.0)

    # 7. ДВИЖОК СИМУЛЯЦИИ ПОРТФЕЛЯ
    portfolio = vbt.Portfolio.from_signals(
        close=val_close,
        high=df_raw.loc[val_dates, 'High'],
        low=df_raw.loc[val_dates, 'Low'],
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=BACKTEST_PARAMS['init_cash'],
        fees=BACKTEST_PARAMS['fee_rate'],
        slippage=BACKTEST_PARAMS['slippage'],
        freq=BACKTEST_PARAMS['freq'],

        # === МЕНЕДЖМЕНТ РИСКОВ ===
        sl_stop=BACKTEST_PARAMS.get('stop_loss', None),
        tp_stop=BACKTEST_PARAMS.get('take_profit', None),
        upon_stop_exit=1,
        accumulate=False
    )

    # 8. РАСЧЕТ И ВЫВОД РЕЗУЛЬТАТОВ КВАНТ-ТЕСТА
    # Метрика Accuracy считается по всем направлениям (0, 1, 2)
    pred_classes = np.zeros(len(signals))
    pred_classes[signals == 1.0] = 2
    pred_classes[signals == 0.0] = 1
    pred_classes[signals == -1.0] = 0

    total_accuracy = np.mean(pred_classes == targets) * 100

    logger.info("=== ИТОГОВЫЕ МЕТРИКИ СИМУЛЯЦИИ VECTORBT (1D ТРИНАРНЫЙ МАКРО) ===")
    logger.info(f"Всего окон в валидации: {len(targets)}")
    logger.info(f"Общая точность модели (Total Accuracy): {total_accuracy:.2f}%")
    logger.info(f"Количество дней в Long: {np.sum(signals == 1.0)}")
    logger.info(f"Количество дней в Short: {np.sum(signals == -1.0)}")
    logger.info(f"Количество дней удержания кэша (Flat): {np.sum(signals == 0.0)}")

    logger.info(f"\n{portfolio.stats().to_string()}")

    # 9. ЭКСПОРТ ГРАФИКОВ
    if BACKTEST_PARAMS.get('save_plots', False):
        reports_dir = os.path.join(PROJECT_ROOT, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        report_filename = f"backtest_3cl_1d_{current_time}.html"
        report_path = os.path.join(reports_dir, report_filename)

        fig = portfolio.plot()
        fig.write_html(report_path)
        logger.info(f"Интерактивный дашборд бэктеста сохранён в: reports/{report_filename}")

if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)
    run_backtest()