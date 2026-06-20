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
    'sequence_length': 120,   # Окно в 2 часа истории
    'hidden_size': 96,        # Золотая середина по мощности скрытого слоя
    'num_layers': 2,          # 2 слоя для включения внутреннего dropout
    'output_size': 3,         # Строго 3 класса
    'dropout_rate': 0.25,     # Регуляризация слоев
    'label_smoothing': 0.05   # Защита от излишней уверенности на шумном рынке
}

# --- НАСТРОЙКИ ОБУЧЕНИЯ ---
TRAINING_PARAMS = {
    'batch_size': 128,        # Размер батча (золотая середина для стабильных градиентов на GPU)
    'epochs': 60,             # Максимальное количество эпох (с запасом, Early Stopping остановит раньше)
    'learning_rate': 5e-4,    # Стартовая скорость обучения (0.0005 — аккуратный шаг для AdamW)
    'train_split': 0.8,       # Хронологическое разделение: 80% на обучение, 20% на валидацию (без перемешивания!)
    'device': 'cuda',         # Обучение строго на видеокарте с использованием AMP (ускорение в 1.5–2 раза)
    'patience': 12,            # Early Stopping: ждем максимум 10 эпох застоя Val Loss, прежде чем завершить процесс
    'num_workers': 0
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