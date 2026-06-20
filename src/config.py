import os
import logging

# --- НАСТРОЙКИ ЗАГРУЗКИ ДАННЫХ ---
DATA_LOAD_PARAMS = {
    'ticker': "BTC-USDT",  # ticker (Тикер) — это краткое уникальное название актива на бирже.
    'interval': "1m",     # interval (Таймфрейм) — это размер одной свечи (одной строчки в таблице).
    'period': "60d"
}

# --- НАСТРОЙКИ ГЕНЕРАЦИИ ПРИЗНАКОВ ДЛЯ 1m ТАЙМФРЕЙМА ---
FEATURE_PARAMS = {
    'target_column': 'Close',

    # Горизонт таргета (в минутах)
    'forward_horizon': 5,

    # Логарифмические доходности (мульти-горизонты лагов)
    'ret_horizons': [1, 3, 5, 15, 30],

    # Окна волатильности (быстрая / медленная)
    'vol_fast_period': 20,
    'vol_slow_period': 120,

    # Окна для скользящих средних тренда (EMA_spread)
    'ema_fast_period': 15,
    'ema_slow_period': 90,

    # Параметры Полос Боллинджера (BB_Position)
    'bb_period': 30,
    'bb_std_dev': 2.2,

    # Окна отклонения цены (Price Z-score)
    'zscore_fast_period': 30,
    'zscore_slow_period': 120,

    # Окно базового профиля объемов (volume_zscore)
    'volume_window': 60,

    # Окно сглаживания внутрисвечевой паники (parkinson_vol)
    'parkinson_window': 20
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
    'output_size': 3,        # Предсказываем 3 числа
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
CLEAN_TICKER = DATA_LOAD_PARAMS['ticker'].replace("-", "_") # BTC-USDT -> BTC_USDT
INTERVAL = DATA_LOAD_PARAMS['interval']

FEATURES_FILE_NAME = f"{CLEAN_TICKER}_{DATA_LOAD_PARAMS['interval']}_features.parquet"
FEATURES_FILE_PATH = os.path.join(DATA_DIR, FEATURES_FILE_NAME)

# --- ДОБАВЛЯЕМ ДИНАМИЧЕСКИЕ ПУТИ ДЛЯ TRAIN / VAL ---
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
            logging.StreamHandler() # Вывод в консоль
        ]
    )