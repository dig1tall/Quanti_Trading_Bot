import os
import pandas as pd
from src import config

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Проверяет переданный DataFrame на пропуски, аномалии и непрерывность дат.
    Универсальна для любого тикера и таймфрейма.
    """
    if df.empty:
        print("[Ошибка Валидации] Передан пустой датасет.")
        return

    print(f"\n--- Запуск валидации данных (Размерность матрицы: {df.shape}) ---")

    # 1. Проверка на пустые значения (NaN)
    nan_counts = df.isna().sum().sum()

    # 2. Проверка на логические аномалии (отрицательные цены)
    # Берем только те колонки, которые гарантированно должны быть положительными цифрами
    price_cols = [col for col in ["Open", "High", "Low", "Close"] if col in df.columns]
    negative_prices = (df[price_cols] <= 0).sum().sum() if price_cols else 0

    # 3. Проверка на нулевые объемы
    zero_volumes = (df["Volume"] <= 0).sum() if "Volume" in df.columns else 0

    # 4. Проверка непрерывности календарной сетки дат
    df_sorted = df.sort_index()

    # Пытаемся определить частоту из конфига (например, '1d' для yfinance — это 'D' в pandas)
    freq = 'D' if config.DATA_LOAD_PARAMS['interval'] == '1d' else None

    missing_days = 0
    if freq and isinstance(df_sorted.index, pd.DatetimeIndex):
        expected_range = pd.date_range(
            start=df_sorted.index.min(),
            end=df_sorted.index.max(),
            freq=freq
        )
        missing_days = len(expected_range) - len(df_sorted)

    # Вывод отчета в консоль
    print(f"Всего строк в памяти: {len(df_sorted)}")
    if isinstance(df_sorted.index, pd.DatetimeIndex):
        print(f"Временной интервал: с {df_sorted.index.min().date()} по {df_sorted.index.max().date()}")
    print(f"Пропущенных значений (NaN): {nan_counts}")
    print(f"Отрицательных/нулевых цен: {negative_prices}")
    print(f"Строк с нулевым объемом торгов: {zero_volumes}")
    if freq:
        print(f"Пропущенных шагов в сетке временного ряда: {missing_days}")

    # Итоговый вердикт
    if nan_counts == 0 and negative_prices == 0 and zero_volumes == 0 and missing_days == 0:
        print("[Вердикт] Данные идеальны. Ошибок и пропусков не обнаружено.")
    else:
        print("[Внимание] В данных обнаружены аномалии! Рекомендуется проверить сырой источник.")


# Этот блок сработает ТОЛЬКО если ты запустишь этот файл напрямую в PyCharm
if __name__ == "__main__":
    print("Запуск валидации в автономном режиме для базового файла...")
    # Здесь мы берем дефолтный файл из конфига для быстрой проверки
    file_path = config.DATA_FILE_PATH
    if os.path.exists(file_path):
        base_df = pd.read_parquet(file_path)
        validate_dataset(base_df)
    else:
        print(f"Файл по умолчанию не найден: {file_path}")