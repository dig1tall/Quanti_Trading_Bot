import os
import logging

DATA_LOAD_PARAMS = {
    'ticker': "BTC-USDT",
    'interval': "1d",
    'period': "6y"
}

# --- НАСТРОЙКИ МАСШТАБИРОВАНИЯ ---
SCALING_PARAMS = {
    'method': 'robust'  # Допустимые: 'standard', 'minmax', 'robust'
}

FEATURE_PARAMS = {
    'target_column': 'Close',
    'forward_horizon': 1,

    # === НАСТРОЙКИ ТРИНАРНОЙ РАЗМЕТКИ ===
    'flat_threshold': 0.005,

    'vol_fast_period': 14,
    'vol_slow_period': 60,
    'ema_fast_period': 12,
    'ema_slow_period': 26,
    'bb_period': 20,
    'bb_std_dev': 2.0,
    'parkinson_window': 14
}

MODEL_PARAMS = {
    'architecture': 'GRU',
    'sequence_length': 20,
    'hidden_size': 16,
    'num_layers': 1,
    'output_size': 3,         # Строго 3 класса (0: Short, 1: Flat, 2: Long)
    'dropout_rate': 0.2,
    'label_smoothing': 0.05
}

# --- НАСТРОЙКИ ОБУЧЕНИЯ ---
TRAINING_PARAMS = {
    'batch_size': 32,
    'epochs': 60,
    'learning_rate': 3e-4,
    'train_split': 0.8,
    'device': 'cuda',
    'patience': 8,
    'lr': 3e-4,
    'weight_decay': 0.015,
    'num_workers': 0,
    'min_delta': 0.001
}

# --- НАСТРОЙКИ БЭКТЕСТОВ 1D ---
BACKTEST_PARAMS = {
    'init_cash': 10000.0,
    'fee_rate': 0.0006,
    'slippage': 0.0005,
    'freq': '1d',

    # Базовые параметры (будут перезаписаны оптимизатором, если RUN_OPTIMIZATION=True)
    'threshold': 0.52,
    'soft_exit_threshold': 0.33,
    'stop_loss': 0.12,
    'take_profit': 0.07,
    'save_plots': False
}

# --- ПУТИ К ФАЙЛАМ ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DATA_FILE_NAME = f"{DATA_LOAD_PARAMS['ticker']}_{DATA_LOAD_PARAMS['interval']}.parquet"
DATA_FILE_PATH = os.path.join(DATA_DIR, DATA_FILE_NAME)

CLEAN_TICKER = DATA_LOAD_PARAMS['ticker'].replace("-", "_")
INTERVAL = DATA_LOAD_PARAMS['interval']

FEATURES_FILE_NAME = f"{CLEAN_TICKER}_{DATA_LOAD_PARAMS['interval']}_features.parquet"
FEATURES_FILE_PATH = os.path.join(DATA_DIR, FEATURES_FILE_NAME)

TRAIN_FEATURES_PATH = os.path.join(DATA_DIR, f"{CLEAN_TICKER}_{INTERVAL}_train.parquet")
VAL_FEATURES_PATH = os.path.join(DATA_DIR, f"{CLEAN_TICKER}_{INTERVAL}_val.parquet")

# --- НАСТРОЙКИ ЛОГИРОВАНИЯ ---
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(level=logging.INFO):
    """Настройка глобального конфигуратора логирования"""
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler()
        ]
    )

def update_backtest_params(best_params: dict):
    """Динамически обновляет конфигурацию бэктеста в памяти результатами оптимизации"""
    global BACKTEST_PARAMS
    logger = logging.getLogger(__name__)
    logger.info("[Config] Динамическая перезапись параметров бэктеста результатами оптимизации:")

    for param_name, new_value in best_params.items():
        if param_name in BACKTEST_PARAMS or param_name == 'soft_exit_threshold':
            old_value = BACKTEST_PARAMS.get(param_name, "None")
            BACKTEST_PARAMS[param_name] = new_value
            logger.info(f"   -> {param_name}: {old_value} ===> {new_value}")