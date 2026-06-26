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

    # 1. Инференс модели (считаем вероятности один раз, чтобы не пересчитывать в циклах)
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
            all_raw_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    raw_probs_matrix = np.vstack(all_raw_probs)

    # 2. Синхронизация цен
    df_features_file = pd.read_parquet(config.VAL_FEATURES_PATH)
    val_dates = df_features_file.index[-len(raw_probs_matrix):]
    df_raw = pd.read_parquet(os.path.join(config.DATA_DIR, "BTC-USDT_1d.parquet"))

    val_close = df_raw.loc[val_dates, 'Close'].copy()
    val_high = df_raw.loc[val_dates, 'High'].copy()
    val_low = df_raw.loc[val_dates, 'Low'].copy()

    # ==========================================
    # 3. СЕТКА ПАРАМЕТРОВ ДЛЯ ПЕРЕБОРА (GRID)
    # ==========================================
    thresholds = np.arange(0.51, 0.62, 0.02)      # [0.51, 0.53, 0.55, 0.57, 0.59, 0.61]
    stop_losses = np.arange(0.02, 0.09, 0.02)     # [0.02, 0.04, 0.06, 0.08] (2% - 8%)
    take_profits = np.arange(0.06, 0.25, 0.04)    # [0.06, 0.10, 0.14, 0.18, 0.22] (6% - 22%)

    logger.info(f"Размерность сетки: Thresholds={len(thresholds)}, SL={len(stop_losses)}, TP={len(take_profits)}")
    logger.info("Генерация многомерных матриц сигналов...")

    # Массивы под мульти-индексы VectorBT
    entries_list, exits_list = [], []
    short_entries_list, short_exits_list = [], []
    param_tuples = []

    # Тот самый вложенный цикл, но мы используем его только для генерации масок!
    for th in thresholds:
        # Считаем сигналы под конкретный threshold
        signals = np.zeros(len(raw_probs_matrix))
        for idx in range(len(raw_probs_matrix)):
            prob_vector = raw_probs_matrix[idx]
            max_prob_class = np.argmax(prob_vector)
            if prob_vector[max_prob_class] >= th:
                signals[idx] = 1.0 if max_prob_class == 1 else -1.0

        signals_series = pd.Series(signals, index=val_close.index)

        ent = (signals_series == 1.0)
        ex = (signals_series <= 0.0)
        se = (signals_series == -1.0)
        sx = (signals_series >= 0.0)

        for sl in stop_losses:
            for tp in take_profits:
                entries_list.append(ent)
                exits_list.append(ex)
                short_entries_list.append(se)
                short_exits_list.append(sx)
                param_tuples.append((th, sl, tp))

    # Схлопываем в DataFrame с MultiIndex для VectorBT
    m_index = pd.MultiIndex.from_tuples(param_tuples, names=['threshold', 'stop_loss', 'take_profit'])

    m_entries = pd.concat(entries_list, axis=1, keys=m_index)
    m_exits = pd.concat(exits_list, axis=1, keys=m_index)
    m_short_entries = pd.concat(short_entries_list, axis=1, keys=m_index)
    m_short_exits = pd.concat(short_exits_list, axis=1, keys=m_index)

    logger.info("Запуск параллельного векторного бэктеста через Numba движок...")

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
        sl_stop=m_index.get_level_values('stop_loss').values,  # Векторизованный SL
        tp_stop=m_index.get_level_values('take_profit').values, # Векторизованный TP
        upon_stop_exit=1,
        accumulate=False
    )

    # 5. АНАЛИЗ РЕЗУЛЬТАТОВ
    total_returns = portfolio.total_return()
    sharpe_ratios = portfolio.sharpe_ratio()
    max_drawdowns = portfolio.max_drawdown()

    # Собираем всё в одну красивую таблицу
    stats_df = pd.DataFrame({
        'Total Return [%]': total_returns * 100,
        'Sharpe Ratio': sharpe_ratios,
        'Max Drawdown [%]': max_drawdowns * 100,
        'Total Trades': portfolio.trades.count()  # Используем валидный для базовой версии метод
    })

    # Сортируем по доходности (или по Шарпу)
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