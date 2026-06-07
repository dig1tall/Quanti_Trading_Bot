import os

# --- НАСТРОЙКИ ЗАГРУЗКИ ДАННЫХ ---
DATA_LOAD_PARAMS = {
    'ticker': "BTC-USD",
    'interval': "1d",
    'period': "2y"
}

# --- ПУТИ К ФАЙЛАМ ---
# __file__ — это путь к src/config.py
# Первый dirname дает папку src/
# Второй dirname выводит нас в корень Quanti/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_FILE_NAME = f"{DATA_LOAD_PARAMS['ticker']}_{DATA_LOAD_PARAMS['interval']}.parquet"
DATA_FILE_PATH = os.path.join(DATA_DIR, DATA_FILE_NAME)


# Настройки генерации признаков (Feature Engineering)
FEATURE_PARAMS = {
    'ema_fast_period': 12,
    'ema_slow_period': 26,
    'sma_period': 20,
    'rsi_period': 14,
    'target_column': 'Close'
}