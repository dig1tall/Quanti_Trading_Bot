import os
import logging

# --- НАСТРОЙКИ ЗАГРУЗКИ ДАННЫХ ---
DATA_LOAD_PARAMS = {
    'ticker': "BTC-USD",  # ticker (Тикер) — это краткое уникальное название актива на бирже.
    'interval': "1m",     # interval (Таймфрейм) — это размер одной свечи (одной строчки в таблице).
    'period': "7d"
}

# --- НАСТРОЙКИ ГЕНЕРАЦИИ ПРИЗНАКОВ 1D (Feature Engineering) ---
"""FEATURE_PARAMS = {
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
}"""
# --- НАСТРОЙКИ ГЕНЕРАЦИИ ПРИЗНАКОВ ДЛЯ 1m таймфрейма ---
FEATURE_PARAMS = {
    'target_column': 'Close',
    'forward_horizon': 5,
    'ema_fast_period': 60,       # Fast тренд: 1 час (60 минут)
    'ema_slow_period': 180,      # Slow тренд: 3 часа (180 минут)
    'sma_period': 120,           # Базовая скользящая: 2 часа
    'rsi_period': 30,            # Сглаженный RSI под минуты
    'macd_signal_period': 45,
    'bb_period': 120,            # Боллинджер на 2 часа
    'bb_std_dev': 2,
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
    'sequence_length': 45,  # Сколько свечей смотрим назад (память модели)
    'hidden_size': 64,       # Мощность памяти скрытого слоя
    'num_layers': 2,        # Количество слоев GRU
    'output_size': 1,        # Предсказываем 1 число
    'dropout_rate': 0.3     # Доля случайно отключаемых нейронов во время тренировки
}

# --- НАСТРОЙКИ ОБУЧЕНИЯ ---
"""TRAINING_PARAMS = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'train_split': 0.8,     # 80% данных на учебу, 20% на валидацию
    'device': 'cuda',        # Режим видеокарты
    'patience': 7           # Ожидание Early Stopping
}"""
#1m
TRAINING_PARAMS = {
    'batch_size': 64,
    'epochs': 50,
    'learning_rate': 0.001,
    'train_split': 0.7,     # 80% данных на учебу, 20% на валидацию
    'device': 'cuda',        # Режим видеокарты
    'patience': 10           # Ожидание Early Stopping
}
# --- НАСТРОЙКИ БЭКТЕСТОВ 1D (Автоматически синхронизированы с DATA_LOAD_PARAMS) ---
"""BACKTEST_PARAMS = {
    'init_cash': 10000.0,      # Стартовый депозит в долларах
    'threshold': 0.005,        # Порог уверенности (для изменения цен/доходностей)
    'fee_rate': 0.0006,        # Комиссия за сделку (0.06% — стандарт Binance Futures)
    'freq': DATA_LOAD_PARAMS['interval'].upper().replace('H', 'h'),
    'save_plots': False,         # Флаг: сохранять ли интерактивный HTML-отчет Plotly
    'stop_loss': 0.02,         # 2% от цены входа (0.02) — жесткий выход при убытке
    'take_profit': 0.04,       # 4% от цены входа (0.04) — фиксация прибыли
}"""
# 1m
BACKTEST_PARAMS = {
    'init_cash': 10000.0,
    'threshold': 0.0015,       # Снижаем порог уверенности модели для входа (0.1%)
    'fee_rate': 0.0006,       # 0.06% комиссия (Binance Futures)
    'stop_loss': 0.005,       # Короткий стоп-лосс: 0.4%
    'take_profit': 0.015,     # Тейк-профит: 0.8% (соотношение риск/прибыль 1:2)
    'time_stop': 60,
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