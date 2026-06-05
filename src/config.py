import os

# --- НАСТРОЙКИ ЗАГРУЗКИ ДАННЫХ ---
TICKER = "BTC-USD"
INTERVAL = "1d"
PERIOD = "2y"

# --- ПУТИ К ФАЙЛАМ ---
# __file__ — это путь к src/config.py
# Первый dirname дает папку src/
# Второй dirname выводит нас в корень TraidER/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_FILE_NAME = f"{TICKER}_{INTERVAL}.parquet"
DATA_FILE_PATH = os.path.join(DATA_DIR, DATA_FILE_NAME)