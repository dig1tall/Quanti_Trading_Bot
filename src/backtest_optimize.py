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

def run_optimization():
    logger.info("=== Запуск глобальной оптимизации параметров Quanti ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Инференс модели (считаем вероятности один раз с учетом TEMPERATURE)
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
            # Применяем TEMPERATURE = 0.8 как в одиночном бэктесте
            TEMPERATURE = 0.8
            probs = torch.softmax(logits / TEMPERATURE, dim=1)
            all_raw_probs.append(probs.cpu().numpy())
    raw_probs_matrix = np.vstack(all_raw_probs)

    # 2. Синхронизация цен (один в один как в backtest.py)
    df_features_file = pd.read_parquet(config.VAL_FEATURES_PATH)
    val_dates = df_features_file.index[-len(raw_probs_matrix):]
    df_raw = pd.read_parquet(os.path.join(config.DATA_DIR, "BTC-USDT_1d.parquet"))

    val_close = df_raw.loc[val_dates, 'Close'].copy()
    val_high = df_raw.loc[val_dates, 'High'].copy()
    val_low = df_raw.loc[val_dates, 'Low'].copy()

    # ==========================================
    # 3. СЕТКА ПАРАМЕТРОВ ДЛЯ ПЕРЕБОРА (GRID)
    # ==========================================
    # Перебираем базовый порог для лонга. Шорт будет автоматически на 0.10 меньше
    thresholds_long = np.arange(0.51, 0.62, 0.02)  # [0.51, 0.53, 0.55, ... 0.61]
    stop_losses = np.arange(0.02, 0.09, 0.02)     # [0.02, 0.04, 0.06, 0.08]
    take_profits = np.arange(0.06, 0.25, 0.04)    # [0.06, 0.10, 0.14, 0.18, 0.22]

    logger.info(f"Размерность сетки: Thresholds={len(thresholds_long)}, SL={len(stop_losses)}, TP={len(take_profits)}")
    logger.info("Генерация многомерных матриц сигналов с раздельными порогами...")

    entries_list, exits_list = [], []
    short_entries_list, short_exits_list = [], []
    param_tuples = []

    for th_long in thresholds_long:
        # Воспроизводим разницу порогов из твоего backtest.py (0.55 - 0.45 = 0.10)
        th_short = th_long - 0.10

        signals = np.zeros(len(raw_probs_matrix), dtype=np.float32)
        current_signal = 0.0

        for idx in range(len(raw_probs_matrix)):
            prob_vector = raw_probs_matrix[idx]
            max_prob_class = np.argmax(prob_vector)
            max_prob = prob_vector[max_prob_class]

            # Выбираем порог динамически, как в backtest.py
            current_threshold = th_long if max_prob_class == 2 else th_short

            if max_prob_class == 2:
                target_signal = 1.0
            elif max_prob_class == 0:
                target_signal = -1.0
            else:
                target_signal = 0.0

            # Логика удержания тренда
            if current_signal == 0.0:
                if max_prob >= current_threshold:
                    current_signal = target_signal
            else:
                if target_signal != current_signal:
                    if max_prob >= current_threshold:
                        current_signal = target_signal

            signals[idx] = current_signal

        signals_series = pd.Series(signals, index=val_close.index)

        ent = (signals_series == 1.0)
        ex = (signals_series == 0.0) | (signals_series == -1.0)

        se = (signals_series == -1.0)
        sx = (signals_series == 0.0) | (signals_series == 1.0)

        for sl in stop_losses:
            for tp in take_profits:
                entries_list.append(ent)
                exits_list.append(ex)
                short_entries_list.append(se)
                short_exits_list.append(sx)
                # В индекс пишем th_long, чтобы понимать, какой лонг-порог победил
                param_tuples.append((th_long, sl, tp))

    # Схлопываем в DataFrame с MultiIndex для VectorBT
    m_index = pd.MultiIndex.from_tuples(param_tuples, names=['threshold', 'stop_loss', 'take_profit'])

    m_entries = pd.concat(entries_list, axis=1, keys=m_index)
    m_exits = pd.concat(exits_list, axis=1, keys=m_index)
    m_short_entries = pd.concat(short_entries_list, axis=1, keys=m_index)
    m_short_exits = pd.concat(short_exits_list, axis=1, keys=m_index)

    logger.info("Запуск параллельного векторного бэктеста...")

    # Чистые плоские массивы параметров для VectorBT
    sl_array = m_index.get_level_values('stop_loss').values
    tp_array = m_index.get_level_values('take_profit').values

    # 4. ВЕКТОРНЫЙ МУЛЬТИ-БЭКТЕСТ
    portfolio = vbt.Portfolio.from_signals(
        close=val_close,
        high=val_high,
        low=val_low,
        entries=m_entries,
        exits=m_exits,
        short_entries=m_short_entries,
        short_exits=m_short_exits,
        init_cash=BACKTEST_PARAMS['init_cash'],
        fees=BACKTEST_PARAMS['fee_rate'],
        slippage=BACKTEST_PARAMS['slippage'],
        freq=BACKTEST_PARAMS['freq'],
        sl_stop=sl_array,
        tp_stop=tp_array,
        upon_stop_exit=1,
        accumulate=False
    )

    # 5. АНАЛИЗ РЕЗУЛЬТАТОВ
    total_returns = portfolio.total_return()
    sharpe_ratios = portfolio.sharpe_ratio()
    max_drawdowns = portfolio.max_drawdown()

    stats_df = pd.DataFrame({
        'Total Return [%]': total_returns * 100,
        'Sharpe Ratio': sharpe_ratios,
        'Max Drawdown [%]': max_drawdowns * 100,
        'Total Trades': portfolio.trades.count()
    })

    stats_df = stats_df.sort_values(by='Total Return [%]', ascending=False)

    print("\n" + "="*70)
    print("   ТОП-10 ЛУЧШИХ КОМБИНАЦИЙ ПАРАМЕТРОВ ПО TOTAL RETURN")
    print("="*70)
    print(stats_df.head(10).to_string())
    print("="*70)

    best_config = stats_df.index[0]
    print(f"\n🏆 РЕКОМЕНДУЕМЫЕ НАСТРОЙКИ:")
    print(f" -> 'threshold': {best_config[0]:.2f}")
    print(f" -> 'stop_loss': {best_config[1]:.2f}")
    print(f" -> 'take_profit': {best_config[2]:.2f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_optimization()