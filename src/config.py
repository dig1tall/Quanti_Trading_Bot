import os
import logging

# --- НАСТРОЙКИ ЗАГРУЗКИ ДАННЫХ ---
DATA_LOAD_PARAMS = {
    'ticker': "ETH-USDT",  # ticker (Тикер) — это краткое уникальное название актива на бирже.
    'interval': "1m",     # interval (Таймфрейм) — это размер одной свечи (одной строчки в таблице).
    'period': "60d"
}

# --- НАСТРОЙКИ ГЕНЕРАЦИИ ПРИЗНАКОВ ДЛЯ 1m таймфрейма ---
FEATURE_PARAMS = {
    'target_column': 'Close',

    'forward_horizon': 5,      # прогноз на 5 минут

    'ret_horizons': [1, 5, 15, 30, 60],

    'vol_fast_period': 30,     # 30 минут
    'vol_slow_period': 240,    # 4 часа

    'ema_fast_period': 20,     # 20 минут
    'ema_slow_period': 120,    # 2 часа

    'sma_period': 60,          # 1 час

    'rsi_period': 30,

    'macd_signal_period': 15,

    'bb_period': 60,
    'bb_std_dev': 2.5,

    'adx_period': 30,

    'obv_rolling_window': 30,
    'chaikin_rolling_window': 60
}

# --- НАСТРОЙКИ МАСШТАБИРОВАНИЯ ---
SCALING_PARAMS = {
    'method': 'robust'  # Допустимые: 'standard', 'minmax', 'robust'
}

# --- НАСТРОЙКИ НЕЙРОСЕТИ ---
MODEL_PARAMS = {
    'architecture': 'GRU',
    'sequence_length': 120,  # Сколько свечей смотрим назад (память модели)
    'hidden_size': 128,       # Мощность памяти скрытого слоя
    'num_layers': 2,        # Количество слоев GRU
    'output_size': 3,        # Предсказываем 1 число
    'dropout_rate': 0.25     # Доля случайно отключаемых нейронов во время тренировки
}

# --- НАСТРОЙКИ ОБУЧЕНИЯ ---
TRAINING_PARAMS = {
    'batch_size': 128,
    'epochs': 80,
    'learning_rate': 0.0003,
    'train_split': 0.8,     # 80% данных на учебу, 20% на валидацию
    'device': 'cuda',        # Режим видеокарты
    'patience': 15           # Ожидание Early Stopping
}

# --- НАСТРОЙКИ БЭКТЕСТОВ 1D (Автоматически синхронизированы с DATA_LOAD_PARAMS) ---
BACKTEST_PARAMS = {
    'init_cash': 10000.0,

    'threshold': 0.55,

    'fee_rate': 0.0006,

    'stop_loss': 0.003,
    'take_profit': 0.006,

    'time_stop': 15,

    'freq': '1m',

    'save_plots': False
}

# --- ПУТИ К ФАЙЛАМ ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Базовый файл с сырыми ценами
DATA_FILE_NAME = f"{DATA_LOAD_PARAMS['ticker']}_{DATA_LOAD_PARAMS['interval']}.parquet"
DATA_FILE_PATH = os.path.join(DATA_DIR, DATA_FILE_NAME)

# Итоговый файл со сгенерированными фичами и скейлингом
CLEAN_TICKER = DATA_LOAD_PARAMS['ticker'].replace('-', '_')
FEATURES_FILE_NAME = f"{CLEAN_TICKER}_{DATA_LOAD_PARAMS['interval']}_features.parquet"
FEATURES_FILE_PATH = os.path.join(DATA_DIR, FEATURES_FILE_NAME)

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
            logging.StreamHandler() # Вывод в консоль
        ]
    )