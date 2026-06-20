import os
import logging
import pandas as pd
import numpy as np

from src import config

# Инициализация логгера для текущего модуля
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Класс для валидации и жесткого исправления рыночных данных под минутные таймфреймы.
    Гарантирует непрерывность временной сетки без утечек данных (Data Leakage).
    """
    def __init__(self):
        # Маппинг интервалов из конфига в понятные для Pandas частоты (freq)
        interval = config.DATA_LOAD_PARAMS['interval']
        if interval == '1m':
            self.freq = '1min'
        elif interval == '1d':
            self.freq = 'D'
        else:
            self.freq = None
            logger.warning(f"Таймфрейм '{interval}' не имеет жесткого маппинга частоты Pandas. Валидация сетки ограничена.")

        self.price_cols = ["Open", "High", "Low", "Close"]

    def validate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Проверяет DataFrame на аномалии, находит временные дыры,
        заполняет их методом Forward Fill и обнуляет пустые объемы.
        Возвращает очищенный и непрерывный DataFrame.
        """
        if df.empty:
            logger.error("Передан пустой датасет. Исправление невозможно.")
            return df

        logger.info(f"--- Запуск валидации и исправления данных (Сырой размер: {df.shape}) ---")
        df_fixed = df.copy()

        # 1. Сортировка по временному индексу
        df_fixed = df_fixed.sort_index()

        # 2. Проверка и исправление непрерывности сетки (критично для 1m таймфреймов)
        if self.freq and isinstance(df_fixed.index, pd.DatetimeIndex):
            start_time = df_fixed.index.min()
            end_time = df_fixed.index.max()

            # Генерируем идеальный непрерывный индекс без пропусков минут
            expected_index = pd.date_range(start=start_time, end=end_time, freq=self.freq, tz=df_fixed.index.tz)

            missing_steps = len(expected_index) - len(df_fixed)

            if missing_steps > 0:
                logger.warning(f"[Аномалия] Обнаружено {missing_steps} пропущенных минутных свечей в истории!")
                # Реиндексируем датасет, вставляя NaN в места пропусков
                df_fixed = df_fixed.reindex(expected_index)

                # Заполняем цены методом ffill (предыдущим значением) — это исключает Data Leakage
                # Важно: сначала заполняем Close, так как это базис цены актива
                df_fixed['Close'] = df_fixed['Close'].ffill()

                # Для искусственных свечей все цены (O, H, L) приравниваем к Close прошлого бара
                for col in ["Open", "High", "Low"]:
                    df_fixed[col] = df_fixed[col].fillna(df_fixed['Close'])

                # Объём искусственных свечей строго равен 0 (торгов-то не было)
                df_fixed['Volume'] = df_fixed['Volume'].fillna(0.0)

                logger.info(f" Сетка успешно восстановлена. Новый размер датасета: {df_fixed.shape}")
            else:
                logger.info("Пропусков временной сетки не обнаружено. Индекс непрерывен.")

        # 3. Финальная проверка на системные NaN (если пропуски были в самом начале датасета)
        nan_counts = df_fixed.isna().sum().sum()
        if nan_counts > 0:
            logger.warning(f"Обнаружены NaN в начале истории ({nan_counts} шт.). Удаляем начальный неполный сегмент.")
            df_fixed = df_fixed.dropna()

        # 4. Проверка на логические аномалии (отрицательные цены)
        active_price_cols = [col for col in self.price_cols if col in df_fixed.columns]
        negative_prices = (df_fixed[active_price_cols] <= 0).sum().sum() if active_price_cols else 0
        if negative_prices > 0:
            logger.error(f"[Критическая ошибка] Обнаружено {negative_prices} отрицательных или нулевых цен!")
            # Заменяем аномальные нули/минусы на ffill
            df_fixed[active_price_cols] = df_fixed[active_price_cols].replace(0, np.nan)
            df_fixed[active_price_cols] = df_fixed[active_price_cols].loc[df_fixed[active_price_cols] < 0] = np.nan
            df_fixed = df_fixed.ffill().bfill()

        # 5. Мониторинг микро-флэтов (Volume == 0) — для информации на 1m
        zero_volumes = (df_fixed["Volume"] == 0).sum() if "Volume" in df_fixed.columns else 0
        if zero_volumes > 0:
            logger.info(f"Зафиксировано {zero_volumes} минут с нулевым объемом торгов (нормально для низкого спреда).")

        logger.info(f"[Вердикт] Валидация завершена. Данные приведены к идеальному виду для GRU. Итоговый размер: {df_fixed.shape}")
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
        # Теперь метод возвращает исправленный датафрейм
        cleaned_df = validator.validate_and_fix_dataset(base_df)

        print("\nПроверка индекса после валидации:")
        print(cleaned_df.tail(3))
    else:
        logger.error(f"Файл по умолчанию не найден: {file_path}")