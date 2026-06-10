import logging
import os
import numpy as np
import pandas as pd

# Импортируем словари конфигурации по их зонам ответственности
from src import config
from src.config import SCALING_PARAMS

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
        self.params = {}  # Здесь храним коэффициенты для каждой колонки

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
                self.params[col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
            elif self.method == 'standard':
                self.params[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
            elif self.method == 'robust':
                self.params[col] = {
                    'median': float(df[col].median()),
                    'iqr': float(df[col].quantile(0.75) - df[col].quantile(0.25))
                }

        logger.info(f"Параметры масштабирования успешно сохранены для {len(self.params)} признаков.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет сохраненные параметры масштабирования к датафрейму."""
        if not self.params:
            raise RuntimeError("Scaler еще не обучен! Сначала вызови метод .fit()")

        # Копируем датафрейм, чтобы не портить исходные данные
        df_scaled = df.copy()

        for col, stats in self.params.items():
            if col not in df_scaled.columns:
                continue

            # Используем .loc[:, col] для избежания фрагментации памяти и SettingWithCopyWarning
            if self.method == 'minmax':
                col_min = stats['min']
                col_max = stats['max']
                if col_max != col_min:
                    df_scaled.loc[:, col] = (df_scaled[col].to_numpy() - col_min) / (col_max - col_min)
                else:
                    df_scaled.loc[:, col] = 0.0

            elif self.method == 'standard':
                mean = stats['mean']
                std = stats['std']
                if std != 0.0:
                    df_scaled.loc[:, col] = (df_scaled[col].to_numpy() - mean) / std
                else:
                    df_scaled.loc[:, col] = 0.0

            elif self.method == 'robust':
                median = stats['median']
                iqr = stats['iqr']
                if iqr != 0.0:
                    df_scaled.loc[:, col] = (df_scaled[col].to_numpy() - median) / iqr
                else:
                    df_scaled.loc[:, col] = 0.0

        return df_scaled

    def fit_transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Совмещает вычисление параметров и transformação."""
        self.fit(df, columns)
        return self.transform(df)


# --- АВТОНОМНЫЙ ТЕСТ МОДУЛЯ ---
if __name__ == "__main__":
    from src.config import setup_logging

    # Настраиваем логирование через наш единый логгер из конфига
    setup_logging(level=logging.INFO)

    logger.info("=== Запуск Scaler в автономном режиме на реальных данных ===")

    # Берем готовый путь к файлу с фичами прямо из конфига путей
    file_path = config.FEATURES_FILE_PATH

    if os.path.exists(file_path):
        # Читаем реальный сгенерированный датасет
        base_df = pd.read_parquet(file_path)
        logger.info(f"Успешно загружен файл для теста: {file_path} (Размерность: {base_df.shape})")

        # Автоматически собираем фичи для масштабирования (все, кроме базовых колонок OHLCV)
        base_cols = ["Open", "High", "Low", "Close", "Volume"]
        cols_to_scale = [col for col in base_df.columns if col not in base_cols]

        # Создаем экземпляр, метод подтянется сам из SCALING_PARAMS['method']
        scaler = Scaler()
        scaled_df = scaler.fit_transform(base_df, cols_to_scale)

        logger.info(f"Масштабирование методом '{scaler.method}' успешно завершено!")
        print("\nПревью отмасштабированных признаков (последние 3 строки):")
        print(scaled_df[cols_to_scale].tail(3))
    else:
        logger.error(f"Файл с признаками не найден по пути: {file_path}")
        logger.error("Запусти main.py, чтобы сгенерировать валидный датасет с фичами.")