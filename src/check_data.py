import os
import pandas as pd


def validate_dataset(file_name: str = "BTC-USD_1d.parquet") -> None:
    # 1. Вычисляем корень проекта динамически
    # __file__ — это путь к текущему скрипту (TraidER/src/check_data.py)
    # Первый dirname возвращает папку src, второй — корень TraidER
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # 2. Собираем точный путь к файлу в папке data
    file_path = os.path.join(project_root, "data", file_name)

    if not os.path.exists(file_path):
        print(f" Ошибка: Файл не найден по пути: {file_path}")
        return

    print(f"--- Анализ датасета: {file_name} ---")

    # 3. Безопасная загрузка
    df = pd.read_parquet(file_path)

    # 4. Проверка на пустые значения (NaN)
    nan_counts = df.isna().sum().sum()

    # 5. Проверка на логические аномалии (отрицательные цены)
    negative_prices = (df[["Open", "High", "Low", "Close"]] <= 0).sum().sum()

    # 6. Проверка на нулевые объемы
    zero_volumes = (df["Volume"] <= 0).sum()

    # 7. Проверка непрерывности календарной сетки дат
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