import os
import logging
import numpy as np
import pandas as pd
import vectorbt as vbt
import torch
from datetime import datetime

from src import config
from src.config import MODEL_PARAMS, TRAINING_PARAMS, PROJECT_ROOT, BACKTEST_PARAMS
from src.dataset import get_backtest_loader
from src.model import QuantiGRU

logger = logging.getLogger(__name__)

def run_optimization_search(val_data_path=None):
    logger.info("=== Запуск глобальной оптимизации параметров Quanti (Мягкий Выход) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Инференс модели (считаем вероятности один раз)
    val_loader = get_backtest_loader()
    model_path = os.path.join(PROJECT_ROOT, "models", "best_quanti_model.pth")
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
    with torch.no_grad():
        for X_batch, _ in val_loader:
            logits = model(X_batch.to(device))
            TEMPERATURE = 0.8
            probs = torch.softmax(logits / TEMPERATURE, dim=1)
            all_raw_probs.append(probs.cpu().numpy())
    raw_probs_matrix = np.vstack(all_raw_probs)

    # 2. Синхронизация цен
    target_val_path = val_data_path if val_data_path is not None else config.VAL_FEATURES_PATH
    df_features_file = pd.read_parquet(target_val_path)
    val_dates = df_features_file.index[-len(raw_probs_matrix):]
    df_raw = pd.read_parquet(os.path.join(config.DATA_DIR, "BTC-USDT_1d.parquet"))

    val_close = df_raw.loc[val_dates, 'Close'].copy()
    val_high = df_raw.loc[val_dates, 'High'].copy()
    val_low = df_raw.loc[val_dates, 'Low'].copy()

    # 3. СЕТКА ПАРАМЕТРОВ ДЛЯ ПЕРЕБОРА (GRID С УЧЕТОМ МЯГКОГО ВЫХОДА)
    thresholds_long = np.arange(0.51, 0.62, 0.01)
    stop_losses = np.arange(0.02, 0.15, 0.01)
    take_profits = np.arange(0.06, 0.30, 0.01)

    logger.info(f"Размерность сетки: Thresholds={len(thresholds_long)}, SL={len(stop_losses)}, TP={len(take_profits)}")

    entries_list, exits_list = [], []
    short_entries_list, short_exits_list = [], []
    param_tuples = []

    exit_threshold = 0.35

    for th_long in thresholds_long:
        th_short = th_long - 0.10
        signals = np.zeros(len(raw_probs_matrix), dtype=np.float32)
        current_signal = 0.0

        for idx in range(len(raw_probs_matrix)):
            prob_vector = raw_probs_matrix[idx]
            p_short = prob_vector[0]
            p_flat = prob_vector[1]
            p_long = prob_vector[2]

            if current_signal == 0.0:
                if p_long >= th_long and p_long > p_short:
                    current_signal = 1.0
                elif p_short >= th_short and p_short > p_long:
                    current_signal = -1.0
            elif current_signal == 1.0:
                if p_long < exit_threshold or p_short > p_long or p_flat > p_long:
                    current_signal = 0.0
            elif current_signal == -1.0:
                if p_short < exit_threshold or p_long > p_short or p_flat > p_short:
                    current_signal = 0.0

            signals[idx] = current_signal

        signals_series = pd.Series(signals, index=val_close.index)

        ent = (signals_series == 1.0)
        ex = (signals_series == 0.0)
        se = (signals_series == -1.0)
        sx = (signals_series == 0.0)

        for sl in stop_losses:
            for tp in take_profits:
                entries_list.append(ent)
                exits_list.append(ex)
                short_entries_list.append(se)
                short_exits_list.append(sx)
                param_tuples.append((th_long, sl, tp))

    m_index = pd.MultiIndex.from_tuples(param_tuples, names=['threshold', 'stop_loss', 'take_profit'])
    m_entries = pd.concat(entries_list, axis=1, keys=m_index)
    m_exits = pd.concat(exits_list, axis=1, keys=m_index)
    m_short_entries = pd.concat(short_entries_list, axis=1, keys=m_index)
    m_short_exits = pd.concat(short_exits_list, axis=1, keys=m_index)

    logger.info("Запуск параллельного векторного бэктеста...")
    sl_array = m_index.get_level_values('stop_loss').values
    tp_array = m_index.get_level_values('take_profit').values

    portfolio = vbt.Portfolio.from_signals(
        close=val_close, high=val_high, low=val_low,
        entries=m_entries, exits=m_exits,
        short_entries=m_short_entries, short_exits=m_short_exits,
        init_cash=BACKTEST_PARAMS['init_cash'],
        fees=BACKTEST_PARAMS['fee_rate'], slippage=BACKTEST_PARAMS['slippage'],
        freq=BACKTEST_PARAMS['freq'], sl_stop=sl_array, tp_stop=tp_array,
        upon_stop_exit=1, accumulate=False
    )

    stats_df = pd.DataFrame({
        'Total Return [%]': portfolio.total_return() * 100,
        'Sharpe Ratio': portfolio.sharpe_ratio(),
        'Max Drawdown [%]': portfolio.max_drawdown() * 100,
        'Total Trades': portfolio.trades.count()
    }).sort_values(by='Total Return [%]', ascending=False)

    print("\n" + "="*70)
    print("   ТОП-10 ЛУЧШИХ КОМБИНАЦИЙ ПАРАМЕТРОВ ПО TOTAL RETURN")
    print("="*70)
    print(stats_df.head(10).to_string())
    print("="*70)

    best_config = stats_df.index[0]

    best_params_dict = {
        'threshold': float(best_config[0]),
        'stop_loss': float(best_config[1]),
        'take_profit': float(best_config[2])
    }
    return best_params_dict

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_optimization_search()