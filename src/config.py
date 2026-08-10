"""Configuration settings for data engine, model parameters, and backtesting."""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
API_PARAMS = {
    'api_key': os.getenv('BYBIT_DEMO_API_KEY', ''),
    'secret': os.getenv('BYBIT_DEMO_API_SECRET', ''),
    'enable_demo': True  # True включает тестовую сеть (Sandbox)
}

# market data ingestion settings
DATA_LOAD_PARAMS = {
    'ticker': "BTC-USDT",
    'interval': "1d",
    'period': "6y"
}

# feature scaling method configuration
SCALING_PARAMS = {
    'method': 'robust'  # options: 'standard', 'minmax', 'robust'
}

# feature engineering and labeling parameters
FEATURE_PARAMS = {
    'target_column': 'Close',
    'forward_horizon': 1,
    'flat_threshold': 0.01,
    'vol_fast_period': 14,
    'vol_slow_period': 60,
    'ema_fast_period': 12,
    'ema_slow_period': 26,
    'bb_period': 20,
    'bb_std_dev': 2.0,
    'parkinson_window': 14
}

# neural network architecture hyperparameters
MODEL_PARAMS = {
    'architecture': 'GRU',
    'sequence_length': 20,
    'hidden_size': 16,
    'num_layers': 1,
    'output_size': 3,
    'dropout_rate': 0.2,
    'label_smoothing': 0.05
}

# model training loop settings
TRAINING_PARAMS = {
    'batch_size': 32,
    'epochs': 60,
    'learning_rate': 1e-3,
    'train_split': 0.8,
    'device': 'cuda',
    'patience': 15,
    'weight_decay': 0.015,
    'num_workers': 0,
    'min_delta': 0.001
}

# default backtest execution settings
BACKTEST_PARAMS = {
    'init_cash': 10000.0,
    'fee_rate': 0.0006,
    'slippage': 0.0005,
    'freq': '1d',
    'threshold_long': 0.515,
    'threshold_short': 0.39,
    'soft_exit_threshold': 0.33,
    'stop_loss': 0.105,
    'take_profit': 0.09,
    'save_plots': True
}

# system path resolution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

DATA_FILE_NAME = f"{DATA_LOAD_PARAMS['ticker']}_{DATA_LOAD_PARAMS['interval']}.parquet"
DATA_FILE_PATH = os.path.join(DATA_DIR, DATA_FILE_NAME)

CLEAN_TICKER = DATA_LOAD_PARAMS['ticker'].replace("-", "_")
INTERVAL = DATA_LOAD_PARAMS['interval']

FEATURES_FILE_NAME = f"{CLEAN_TICKER}_{DATA_LOAD_PARAMS['interval']}_features.parquet"
FEATURES_FILE_PATH = os.path.join(DATA_DIR, FEATURES_FILE_NAME)

TRAIN_FEATURES_PATH = os.path.join(DATA_DIR, f"{CLEAN_TICKER}_{INTERVAL}_train.parquet")
VAL_FEATURES_PATH = os.path.join(DATA_DIR, f"{CLEAN_TICKER}_{INTERVAL}_val.parquet")

# logging format setup
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(level=logging.INFO):
    """Configures system-wide logging handler and format."""
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler()
        ]
    )

def update_backtest_params(best_params: dict):
    """Updates backtest configuration parameters with optimization search output."""
    global BACKTEST_PARAMS
    logger = logging.getLogger(__name__)
    logger.info("Overwriting backtest configuration with search results...")

    for param_name, new_value in best_params.items():
        if param_name in BACKTEST_PARAMS:
            old_value = BACKTEST_PARAMS.get(param_name, "None")
            BACKTEST_PARAMS[param_name] = new_value
            logger.info(f"   -> {param_name}: {old_value} ===> {new_value}")