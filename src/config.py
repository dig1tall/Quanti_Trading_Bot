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
    # Дневное изменение цены в диапазоне [-0.005, 0.005] (т.е. +-0.5%) считается флэтом
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
    'output_size': 3,         # === ИЗМЕНЕНО: Строго 3 класса (0: Short, 1: Flat, 2: Long) ===
    'dropout_rate': 0.2,
    'label_smoothing': 0.05
}

# --- НАСТРОЙКИ ОБУЧЕНИЯ ---
TRAINING_PARAMS = {
    'batch_size': 32,        # Размер батча (золотая середина для стабильных градиентов на GPU)
    'epochs': 60,             # Максимальное количество эпох (с запасом, Early Stopping остановит раньше)
    'learning_rate': 3e-4,    # Стартовая скорость обучения (0.0005 — аккуратный шаг для AdamW)
    'train_split': 0.8,       # Хронологическое разделение: 80% на обучение, 20% на валидацию (без перемешивания!)
    'device': 'cuda',         # Обучение строго на видеокарте с использованием AMP (ускорение в 1.5–2 раза)
    'patience': 8,            # Early Stopping: ждем максимум 10 эпох застоя Val Loss, прежде чем завершить процесс
    'lr': 3e-4,
    'weight_decay': 0.015,
    'num_workers': 0,
    'min_delta': 0.001
}

# --- НАСТРОЙКИ БЭКТЕСТОВ 1D ---
BACKTEST_PARAMS = {
    'init_cash': 10000.0,
    'fee_rate': 0.001,
    'slippage': 0.0005,
    'freq': '1d',

    # === ИЗМЕНЕНО: Жесткий фильтр дребезга ===
    # Перевес топ-1 класса над топ-2 должен быть минимум 22%
    # Это отсечет ситуации вроде 40/30/30 и заставит модель сидеть в Flat
    'threshold': 0.55,

    # === РИСК-МЕНЕДЖМЕНТ ===
    'stop_loss': 0.025,   # Поджимаем стоп с 3.5% до 2.5% (быстрее выходим из ошибок)
    'take_profit': 0.06,  # Тейк на 6% (забираем локальные импульсы)
    'save_plots': True
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