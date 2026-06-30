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
    logger.info("=== Запуск промышленного бэктестинга Quanti Ver 2.0.0 (Мягкий Выход) ===")
    device = torch.device(TRAINING_PARAMS.get('device', 'cpu') if torch.cuda.is_available() else 'cpu')

    val_loader = get_backtest_loader()
    model_path = os.path.join(PROJECT_ROOT, "models", "best_quanti_model.pth")
    if not os.path.exists(model_path):
        logger.error(f"Критическая ошибка: файл весов {model_path} не найден!")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_features = checkpoint.get('feature_names', [])
    input_size = len(saved_features) if saved_features else val_loader.dataset.X.shape[1]

    model = QuantiGRU(
        input_size=input_size,
        hidden_size=MODEL_PARAMS['hidden_size'],
        num_layers=MODEL_PARAMS['num_layers'],
        output_size=MODEL_PARAMS['output_size'],
        dropout_rate=MODEL_PARAMS['dropout_rate']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_raw_probs = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch.to(device))
            TEMPERATURE = 0.8
            probs = torch.softmax(logits / TEMPERATURE, dim=1)
            all_raw_probs.append(probs.cpu().numpy())
            all_targets.extend(y_batch.numpy().flatten())

    raw_probs_matrix = np.vstack(all_raw_probs)
    targets = np.array(all_targets)

    # 4. ИНТЕРПРЕТАЦИЯ СИГНАЛОВ (МЯГКИЙ ВЫХОД)
    entry_threshold_long = BACKTEST_PARAMS.get('threshold', 0.52)
    entry_threshold_short = entry_threshold_long - 0.10
    exit_threshold = BACKTEST_PARAMS.get('soft_exit_threshold', 0.33)

    logger.info(f"Рабочие пороги ВХОДА: Long >= {entry_threshold_long:.2f}, Short >= {entry_threshold_short:.2f}")
    logger.info(f"Порог мягкого ВЫХОДА в кэш: < {exit_threshold:.2f}")

    signals = np.zeros(len(raw_probs_matrix), dtype=np.float32)
    current_signal = 0.0

    for idx in range(len(raw_probs_matrix)):
        prob_vector = raw_probs_matrix[idx]
        p_short = prob_vector[0]
        p_flat = prob_vector[1]
        p_long = prob_vector[2]

        if current_signal == 0.0:
            if p_long >= entry_threshold_long and p_long > p_short:
                current_signal = 1.0
            elif p_short >= entry_threshold_short and p_short > p_long:
                current_signal = -1.0
        elif current_signal == 1.0:
            if p_long < exit_threshold or p_short > p_long or p_flat > p_long:
                current_signal = 0.0
        elif current_signal == -1.0:
            if p_short < exit_threshold or p_long > p_short or p_flat > p_short:
                current_signal = 0.0

        signals[idx] = current_signal

    # 5. ХРОНОЛОГИЧЕСКАЯ СИНХРОНИЗАЦИЯ ЦЕН
    val_path = config.VAL_FEATURES_PATH
    df_features_file = pd.read_parquet(val_path)
    for date_col in ['Date', 'date']:
        if date_col in df_features_file.columns:
            df_features_file.set_index(date_col, inplace=True)
    df_features_file = df_features_file.sort_index()
    val_dates = df_features_file.index[-len(targets):]

    df_raw = pd.read_parquet(os.path.join(config.DATA_DIR, "BTC-USDT_1d.parquet"))
    for date_col in ['Date', 'date']:
        if date_col in df_raw.columns:
            df_raw.set_index(date_col, inplace=True)
    df_raw = df_raw.sort_index()

    val_close = df_raw.loc[val_dates, 'Close'].copy()
    signals_series = pd.Series(signals, name='Quanti_Signals', index=val_close.index)

    # 6. КОНСТРУИРОВАНИЕ МАСОК ОРДЕРОВ
    entries = (signals_series == 1.0)
    exits = (signals_series == 0.0)
    short_entries = (signals_series == -1.0)
    short_exits = (signals_series == 0.0)

    # 7. ДВИЖОК СИМУЛЯЦИИ ПОРТФЕЛЯ
    portfolio = vbt.Portfolio.from_signals(
        close=val_close,
        high=df_raw.loc[val_dates, 'High'],
        low=df_raw.loc[val_dates, 'Low'],
        entries=entries, exits=exits,
        short_entries=short_entries, short_exits=short_exits,
        init_cash=BACKTEST_PARAMS['init_cash'],
        fees=BACKTEST_PARAMS['fee_rate'],
        slippage=BACKTEST_PARAMS['slippage'],
        freq=BACKTEST_PARAMS['freq'],
        sl_stop=BACKTEST_PARAMS.get('stop_loss', None),
        tp_stop=BACKTEST_PARAMS.get('take_profit', None),
        upon_stop_exit=1, accumulate=False
    )

    # 8. РАСЧЕТ МЕТРИК
    pred_classes = np.zeros(len(signals))
    pred_classes[signals == 1.0] = 2
    pred_classes[signals == 0.0] = 1
    pred_classes[signals == -1.0] = 0
    total_accuracy = np.mean(pred_classes == targets) * 100

    logger.info("=== ИТОГОВЫЕ МЕТРИКИ СИМУЛЯЦИИ VECTORBT ===")
    logger.info(f"Общая точность модели (Total Accuracy): {total_accuracy:.2f}%")
    logger.info(f"\n{portfolio.stats().to_string()}")

    if BACKTEST_PARAMS.get('save_plots', False):
        reports_dir = os.path.join(PROJECT_ROOT, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, f"backtest_3cl_1d_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.html")
        portfolio.plot().write_html(report_path)

if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)
    run_backtest()