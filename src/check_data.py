import os
import logging
import pandas as pd

from src import config

# Инициализация логгера для текущего модуля
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Класс для валидации целостности рыночных данных.
    Проверяет DataFrame на пропуски, логические аномалии и непрерывность временной сетки.
    """
    def __init__(self):
        # Определение частоты для pandas (для '1d' -> 'D'). Если интервал другой (н-р, '1h'),
        # запишется None, и проверка непрерывности сетки будет временно пропущена.
        self.freq       = 'D' if config.DATA_LOAD_PARAMS['interval'] == '1d' else None
        self.price_cols = ["Open", "High", "Low", "Close"]

    def validate_dataset(self, df: pd.DataFrame) -> None:
        """
        Проверяет переданный DataFrame на пропуски, аномалии и непрерывность дат.
        Универсальна для любого тикера и таймфрейма.
        """
        if df.empty:
            logger.error("Передан пустой датасет.")
            return

        logger.info(f"--- Запуск валидации данных (Размерность матрицы: {df.shape}) ---")

        # 1. Проверка на пустые значения (NaN)
        nan_counts = df.isna().sum().sum()

        # 2. Проверка на логические аномалии (отрицательные цены)
        active_price_cols = [col for col in self.price_cols if col in df.columns]
        negative_prices = (df[active_price_cols] <= 0).sum().sum() if active_price_cols else 0

        # 3. Проверка на нулевые объемы
        zero_volumes = (df["Volume"] <= 0).sum() if "Volume" in df.columns else 0

        # 4. Проверка непрерывности календарной сетки дат
        df_sorted = df.sort_index()

        missing_days = 0
        if self.freq and isinstance(df_sorted.index, pd.DatetimeIndex):
            expected_range = pd.date_range(
                start=df_sorted.index.min(),
                end=df_sorted.index.max(),
                freq=self.freq
            )
            missing_days = len(expected_range) - len(df_sorted)

        # Вывод отчета через логгер
        logger.info(f"Всего строк в памяти: {len(df_sorted)}")
        if isinstance(df_sorted.index, pd.DatetimeIndex):
            logger.info(f"Временной интервал: с {df_sorted.index.min().date()} по {df_sorted.index.max().date()}")
        logger.info(f"Пропущенных значений (NaN): {nan_counts}")
        logger.info(f"Отрицательных/нулевых цен: {negative_prices}")
        logger.info(f"Строк с нулевым объемом торгов: {zero_volumes}")
        if self.freq:
            logger.info(f"Пропущенных шагов в сетке временного ряда: {missing_days}")

        # Итоговый вердикт
        if nan_counts == 0 and negative_prices == 0 and zero_volumes == 0 and missing_days == 0:
            logger.info("[Вердикт] Данные идеальны. Ошибок и пропусков не обнаружено.")
        else:
            logger.warning("[Внимание] В данных обнаружены аномалии! Рекомендуется проверить сырой источник.")


# --- АВТОНОМНЫЙ ТЕСТ МОДУЛЯ ---
if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)

    logger.info("Запуск валидации в автономном режиме для базового файла...")
    file_path = config.DATA_FILE_PATH
    if os.path.exists(file_path):
        base_df = pd.read_parquet(file_path)

        # Создание экземпляра валидатора и тестируем
        validator = DataValidator()
        validator.validate_dataset(base_df)
    else:
        logger.error(f"Файл по умолчанию не найден: {file_path}")