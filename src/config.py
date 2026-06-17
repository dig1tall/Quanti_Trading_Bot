import os
import logging

# --- НАСТРОЙКИ ЗАГРУЗКИ ДАННЫХ ---
DATA_LOAD_PARAMS = {
    'ticker': "BTC-USD",# ticker (Тикер) — это краткое уникальное название актива на бирже.
    'interval': "1d",   # interval (Таймфрейм) — это размер одной свечи (одной строчки в таблице).
    'period': "5y"
}

# --- НАСТРОЙКИ ГЕНЕРАЦИИ ПРИЗНАКОВ (Feature Engineering) ---
FEATURE_PARAMS = {
    'target_column': 'Close',
    'ema_fast_period': 12,
    'ema_slow_period': 26,
    'sma_period': 20,
    'rsi_period': 14,
    'macd_signal_period': 9,
    'bb_period': 20,
    'bb_std_dev': 2,
    'adx_period': 14,             # Период для Trend_Strength (упрощенный ADX)
    'obv_rolling_window': 14,     # Окно сглаживания для безопасного OBV (rolling вместо cumsum)
    'chaikin_rolling_window': 20  # Окно накопления для безопасного Осциллятора Чайкина
}

# --- НАСТРОЙКИ МАСШТАБИРОВАНИЯ ---
SCALING_PARAMS = {
    'method': 'robust'  # Допустимые: 'standard', 'minmax', 'robust'
}

# --- НАСТРОЙКИ НЕЙРОСЕТИ ---
MODEL_PARAMS = {
    'architecture': 'GRU',
    'sequence_length': 30,  # Сколько свечей смотрим назад (память модели)
    'hidden_size': 64,       # Мощность памяти скрытого слоя
    'num_layers': 1,        # Количество слоев GRU
    'output_size': 1,        # Предсказываем 1 число
    'dropout_rate': 0.4     # Доля случайно отключаемых нейронов во время тренировки
}

# --- НАСТРОЙКИ ОБУЧЕНИЯ ---
TRAINING_PARAMS = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'train_split': 0.8,     # 80% данных на учебу, 20% на валидацию
    'device': 'cuda',        # Режим видеокарты
    'patience': 7           # Ожидание Early Stopping
}
# --- НАСТРОЙКИ БЭКТЕСТОВ ---
BACKTEST_PARAMS = {
    'threshold': 0.0015,
    'fee_rate': 0.0006
}
# --- ПУТИ К ФАЙЛАМ ---
# __file__ — это путь к src/config.py
# Первый dirname дает папку src/
# Второй dirname выводит нас в корень Quanti/
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
            # В будущем сюда можно легко добавить вывод в файл:
            # logging.FileHandler(os.path.join(PROJECT_ROOT, "quanti.log"))
        ]
    )