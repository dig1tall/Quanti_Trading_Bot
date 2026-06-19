import logging
import os
import numpy as np
import pandas as pd

# Импорт словарей конфигурации по их зонам ответственности
from src import config
from src.config import SCALING_PARAMS

# Инициализация логгера для модуля масштабирования фич
logger = logging.getLogger(__name__)


class Scaler:
    """
    Класс для ручного масштабирования признаков.
    Поддерживает методы: 'standard', 'minmax', 'robust'.
    Запоминает параметры на Train-выборке и применяется к Test/Real-time.
    """
    def __init__(self, method: str = SCALING_PARAMS['method']):
        if method not in ['standard', 'minmax', 'robust']:
            raise ValueError("Допустимые методы: 'standard', 'minmax', 'robust'")
        self.method = method
        self.params = {}  # Здесь хранятся коэффициенты для каждой колонки

    def fit(self, df: pd.DataFrame, columns: list) -> None:
        """Вычисляет и сохраняет статистические параметры для указанных числовых колонок."""
        logger.info(f"Инициализация Scaler (метод: {self.method}) на обучающей выборке...")

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Колонка {col} не найдена в DataFrame при fit. Пропускаем.")
                continue

            # Защита от нечисловых типов данных
            if not np.issubdtype(df[col].dtype, np.number):
                logger.warning(f"Колонка {col} не является числовой. Масштабирование невозможно.")
                continue

            if self.method == 'minmax':
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                # ЗАЩИТА: если max == min (константный признак), ставим разницу 1e-8 вместо 0
                denom = (max_val - min_val) if max_val != min_val else 1e-8
                self.params[col] = {
                    'min': min_val,
                    'denom': denom
                }
            elif self.method == 'standard':
                mean_val = float(df[col].mean())
                std_val = float(df[col].std())
                # ЗАЩИТА: если std == 0, ставим 1e-8
                if std_val == 0 or np.isnan(std_val):
                    std_val = 1e-8
                self.params[col] = {
                    'mean': mean_val,
                    'std': std_val
                }
            elif self.method == 'robust':
                median_val = float(df[col].median())
                q75 = df[col].quantile(0.75)
                q25 = df[col].quantile(0.25)
                iqr_val = float(q75 - q25)

                # === ВОТ ТУТ ПРАВКА И ЗАЩИТА ОТ ДЕЛЕНИЯ НА НОЛЬ ===
                # Если IQR равен 0 (признак не менялся на 50% минутных свечей),
                # принудительно выставляем минимальное значение 1e-8, чтобы не было NaN/Inf
                if iqr_val == 0 or np.isnan(iqr_val):
                    iqr_val = 1e-8

                self.params[col] = {
                    'median': median_val,
                    'iqr': iqr_val
                }

        logger.info(f"Параметры масштабирования успешно сохранены для {len(self.params)} признаков.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет сохраненные параметры масштабирования к указанным колонкам."""
        if not self.params:
            raise ValueError("Scaler еще не обучен! Сначала вызови метод fit().")

        # Создание копии, чтобы не портить исходный датафрейм в памяти
        df_scaled = df.copy()

        for col, col_params in self.params.items():
            if col not in df_scaled.columns:
                continue

            # Явное приведение колонки к float64 перед записью дробей
            df_scaled[col] = df_scaled[col].astype(float)

            if self.method == 'standard':
                mean = col_params['mean']
                std = col_params['std']
                df_scaled.loc[:, col] = (df_scaled[col].to_numpy() - mean) / std

            elif self.method == 'minmax':
                min_val = col_params['min']
                denom = col_params['denom']
                df_scaled.loc[:, col] = (df_scaled[col].to_numpy() - min_val) / denom

            elif self.method == 'robust':
                median = col_params['median']
                iqr = col_params['iqr']

                # Дополнительная локальная проверка при трансформации
                if iqr == 0:
                    iqr = 1e-8

                df_scaled.loc[:, col] = (df_scaled[col].to_numpy() - median) / iqr

        return df_scaled

    def fit_transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Удобный метод-комбайн для одновременного обучения и восстановления."""
        self.fit(df, columns)
        return self.transform(df)


# --- АВТОНОМНЫЙ ТЕСТ МОДУЛЯ ---
if __name__ == "__main__":
    from src.config import setup_logging

    # Инициализация логгера через конфигурацию
    setup_logging(level=logging.INFO)

    logger.info("=== Запуск Scaler в автономном режиме на реальных данных ===")

    # Готовый путь к файлу с фичами прямо из конфига путей
    file_path = config.FEATURES_FILE_PATH

    if os.path.exists(file_path):
        base_df = pd.read_parquet(file_path)
        logger.info(f"Успешно загружен файл для теста: {file_path} (Размерность: {base_df.shape})")

        base_cols = ["Open", "High", "Low", "Close", "Volume"]
        cols_to_scale = [col for col in base_df.columns if col not in base_cols]

        scaler = Scaler()
        scaled_df = scaler.fit_transform(base_df, cols_to_scale)

        logger.info(f"Масштабирование методом '{scaler.method}' успешно завершено!")
        print("\nПревью отмасштабированных признаков (последние 3 строки):")
        print(scaled_df[cols_to_scale].tail(3))
    else:
        logger.error(f"Файл с признаками не найден по пути: {file_path}")