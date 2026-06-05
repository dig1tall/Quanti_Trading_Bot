import os
import pandas as pd
from src import config


def validate_dataset() -> None:
    # 1. Берем точный путь к файлу напрямую из конфига
    file_path = config.DATA_FILE_PATH

    if not os.path.exists(file_path):
        print(f" Ошибка: Файл не найден по пути: {file_path}")
        return

    print(f"--- Анализ датасета: {config.DATA_FILE_NAME} ---")

    # 2. Безопасная загрузка
    df = pd.read_parquet(file_path)

    # 3. Проверка на пустые значения (NaN)
    nan_counts = df.isna().sum().sum()

    # 4. Проверка на логические аномалии (отрицательные цены)
    negative_prices = (df[["Open", "High", "Low", "Close"]] <= 0).sum().sum()

    # 5. Проверка на нулевые объемы
    zero_volumes = (df["Volume"] <= 0).sum()

    # 6. Проверка непрерывности календарной сетки дат
    df = df.sort_index()
    expected_range = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq="D"
    )
    missing_days = len(expected_range) - len(df)

    # Вывод отчета в консоль
    print(f"Всего торговых дней в базе: {len(df)}")
    print(
        f"Временной интервал: с {df.index.min().date()} по {df.index.max().date()}"
    )
    print(f"Пропущенных значений (NaN): {nan_counts}")
    print(f"Отрицательных/нулевых цен: {negative_prices}")
    print(f"Дней с нулевым объемом торгов: {zero_volumes}")
    print(f"Пропущенных дней в сетке: {missing_days}")

    if (
        nan_counts == 0
        and negative_prices == 0
        and zero_volumes == 0
        and missing_days == 0
    ):
        print("\n Вердикт: Датасет идеален. Ошибок и пропусков не обнаружено.")
    else:
        print("\n Внимание: В данных обнаружены аномалии!")


if __name__ == "__main__":
    validate_dataset()