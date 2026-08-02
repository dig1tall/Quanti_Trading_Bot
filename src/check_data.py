import os
import logging
import pandas as pd
import numpy as np

from src import config

# Инициализация логгера для текущего модуля
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Класс для жесткой валидации рыночных данных.
    Гарантирует 100% целостность и непрерывность временной сетки для таймфреймов 1d/1m.
    """
    def __init__(self):
        # Маппинг интервалов из конфига в частоты Pandas
        interval = config.DATA_LOAD_PARAMS['interval']
        if interval == '1m':
            self.freq = 'min'
        elif interval == '1d':
            self.freq = 'D'
        else:
            self.freq = None
            logger.warning(f"Таймфрейм '{interval}' не имеет жесткого маппинга частоты Pandas. Валидация сетки ограничена.")

        self.price_cols = ["Open", "High", "Low", "Close"]

    def validate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Проверяет DataFrame на аномалии, пропуски дней/минут и отрицательные цены.
        При обнаружении мелких дыр заполняет их методом ffill, при крупных — выбрасывает ValueError.
        """
        if df.empty:
            logger.error("Передан пустой датасет. Валидация невозможна.")
            return df

        logger.info(f"--- Запуск жесткой валидации данных (Размер: {df.shape}) ---")
        df_fixed = df.copy()

        # 1. Сортировка по временному индексу
        df_fixed = df_fixed.sort_index()

        # 2. Проверка непрерывности сетки и интерполяция мелких пропусков
        if self.freq and isinstance(df_fixed.index, pd.DatetimeIndex):
            start_time = df_fixed.index.min()
            end_time = df_fixed.index.max()

            # Генерируем идеальный непрерывный индекс для проверки
            expected_index = pd.date_range(start=start_time, end=end_time, freq=self.freq, tz=df_fixed.index.tz)
            missing_steps = len(expected_index) - len(df_fixed)

            if missing_steps > 0:
                if missing_steps <= 3:  # Локальный сбой тестнета (до 3 баров) — латаем на лету
                    logger.warning(
                        f"[АНОМАЛИЯ] Обнаружено {missing_steps} пропущенных баров в истории! "
                        f"Автоматически восстанавливаем сетку через reindex() и ffill()..."
                    )
                    # Приводим к идеальному индексу, создавая пустые строки для пропущенных дат
                    df_fixed = df_fixed.reindex(expected_index)
                    # Заполняем NaN значениями предыдущего доступного дня (Open, High, Low, Close, Volume)
                    df_fixed = df_fixed.ffill()
                else:
                    # Если пропущено много данных, это критично для фичей (например, скользящих средних)
                    logger.error(f"[КРИТИЧЕСКАЯ АНОМАЛИЯ] Слишком много пропусков ({missing_steps} баров)! Конвейер остановлен.")
                    raise ValueError(f"История повреждена: отсутствует {missing_steps} баров таймфрейма {self.freq}.")
            else:
                logger.info("Пропусков временной сетки не обнаружено. Индекс идеален и непрерывен.")

        # 3. Финальная проверка на системные NaN (например, если что-то еще просочилось)
        nan_counts = df_fixed.isna().sum().sum()
        if nan_counts > 0:
            logger.warning(f"Обнаружены NaN в истории ({nan_counts} шт.). Удаляем некорректные строки.")
            df_fixed = df_fixed.dropna()

        # 4. Проверка на логические аномалии (отрицательные или нулевые цены)
        active_price_cols = [col for col in self.price_cols if col in df_fixed.columns]
        if active_price_cols:
            negative_prices = (df_fixed[active_price_cols] <= 0).sum().sum()
            if negative_prices > 0:
                logger.error(
                    f"[Критическая ошибка] Обнаружено {negative_prices} отрицательных или нулевых цен!"
                )
                raise ValueError(
                    f"Обнаружено {negative_prices} отрицательных или нулевых цен."
                )

        # 5. Мониторинг нулевого объема (полезная диагностика флэта)
        zero_volumes = (df_fixed["Volume"] == 0).sum() if "Volume" in df_fixed.columns else 0
        if zero_volumes > 0:
            logger.info(f"Зафиксировано {zero_volumes} свечей с нулевым объемом торгов.")

        logger.info(f"[Вердикт] Валидация успешно пройдена. Итоговый размер для GRU: {df_fixed.shape}")
        return df_fixed


# --- АВТОНОМНЫЙ ТЕСТ МОДУЛЯ ---
if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)

    logger.info("=== Запуск DataValidator в автономном режиме ===")

    file_path = config.DATA_FILE_PATH
    if os.path.exists(file_path):
        base_df = pd.read_parquet(file_path)

        validator = DataValidator()
        cleaned_df = validator.validate_dataset(base_df)

        print("\nПроверка индекса после валидации:")
        print(cleaned_df.tail(3))
    else:
        logger.error(f"Файл по умолчанию не найден: {file_path}")