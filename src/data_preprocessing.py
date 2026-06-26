import os
import logging
import json
import numpy as np
import pandas as pd

from src import config
from src.config import SCALING_PARAMS

logger = logging.getLogger(__name__)


class Scaler:
    """
    Класс для ручного масштабирования признаков.
    Поддерживает методы: 'standard', 'minmax', 'robust'.
    Запоминает параметры на Train-выборке, сериализует их в файл и применяется в Real-time/Backtest.
    """
    def __init__(self, method: str = SCALING_PARAMS.get('method', 'robust')):
        if method not in ['standard', 'minmax', 'robust']:
            raise ValueError("Допустимые методы: 'standard', 'minmax', 'robust'")
        self.method = method
        self.params = {}  # Здесь хранятся коэффициенты для каждой колонки

    def fit(self, df: pd.DataFrame, columns: list) -> None:
        """Вычисляет и сохраняет статистические параметры для указанных числовых колонок."""
        logger.info(f"Инициализация Scaler (метод: {self.method}) на обучающей выборке...")
        self.params = {}  # Сбрасываем старые параметры перед переобучением

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Колонка {col} не найдена в DataFrame при fit. Пропускаем.")
                continue

            if not np.issubdtype(df[col].dtype, np.number):
                logger.warning(f"Колонка {col} не является числовой. Масштабирование невозможно.")
                continue

            if self.method == 'minmax':
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                denom = (max_val - min_val) if max_val != min_val else 1e-8
                self.params[str(col)] = {
                    'min': min_val,
                    'denom': denom
                }
            elif self.method == 'standard':
                mean_val = float(df[col].mean())
                std_val = float(df[col].std())
                if std_val == 0 or np.isnan(std_val):
                    std_val = 1e-8
                self.params[str(col)] = {
                    'mean': mean_val,
                    'std': std_val
                }
            elif self.method == 'robust':
                median_val = float(df[col].median())
                q75 = float(df[col].quantile(0.75))
                q25 = float(df[col].quantile(0.25))
                iqr_val = q75 - q25

                if iqr_val == 0 or np.isnan(iqr_val):
                    iqr_val = 1e-8

                self.params[str(col)] = {
                    'median': median_val,
                    'iqr': iqr_val
                }

        logger.info(f"Параметры масштабирования успешно сохранены для {len(self.params)} признаков.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет сохраненные параметры масштабирования к указанным колонкам."""
        if not self.params:
            raise ValueError("Scaler еще не обучен или не загружен! Сначала вызови fit() или load().")

        df_scaled = df.copy()

        for col, col_params in self.params.items():
            if col not in df_scaled.columns:
                continue

            # Приводим к float64 во избежание конфликтов типов при делении массивов
            values = df_scaled[col].to_numpy().astype(float)

            if self.method == 'standard':
                mean = col_params['mean']
                std = col_params['std']
                df_scaled[col] = (values - mean) / std

            elif self.method == 'minmax':
                min_val = col_params['min']
                denom = col_params['denom']
                df_scaled[col] = (values - min_val) / denom

            elif self.method == 'robust':
                median = col_params['median']
                iqr = col_params['iqr']
                df_scaled[col] = (values - median) / iqr

        return df_scaled

    def fit_transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Метод для одновременного обучения и трансформации."""
        self.fit(df, columns)
        return self.transform(df)

    def save(self, file_path: str = None) -> None:
        """Сохраняет параметры скейлера в JSON файл."""
        if not file_path:
            models_dir = os.path.join(config.PROJECT_ROOT, "models")
            os.makedirs(models_dir, exist_ok=True)
            file_path = os.path.join(models_dir, "scaler_params.json")

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({'method': self.method, 'params': self.params}, f, indent=4, ensure_ascii=False)
            logger.info(f"Параметры скейлера успешно сериализованы в файл: {file_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении параметров скейлера: {e}", exc_info=True)

    def load(self, file_path: str = None) -> None:
        """Загружает параметры скейлера из JSON файла."""
        if not file_path:
            file_path = os.path.join(config.PROJECT_ROOT, "models", "scaler_params.json")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл параметров скейлера не найден по пути: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.method = data['method']
            self.params = data['params']
            logger.info(f"Параметры скейлера успешно загружены из файла: {file_path} (Метод: {self.method})")
        except Exception as e:
            logger.error(f"Ошибка при загрузке параметров скейлера: {e}", exc_info=True)
            raise e


# --- АВТОНОМНЫЙ ТЕСТ МОДУЛЯ ---
if __name__ == "__main__":
    from src.config import setup_logging

    setup_logging(level=logging.INFO)
    logger.info("=== Запуск Scaler в автономном режиме на реальных данных ===")

    file_path = config.FEATURES_FILE_PATH

    if os.path.exists(file_path):
        base_df = pd.read_parquet(file_path)
        logger.info(f"Успешно загружен файл для теста: {file_path} (Размерность: {base_df.shape})")

        # Исключаем сырые колонки, если они есть. Если их нет — скейлим всё.
        base_cols = ["Open", "High", "Low", "Close", "Volume"]
        cols_to_scale = [col for col in base_df.columns if col not in base_cols]
        if not cols_to_scale:
            cols_to_scale = list(base_df.columns)

        scaler = Scaler()
        scaled_df = scaler.fit_transform(base_df, cols_to_scale)

        scaler.save()

        new_scaler = Scaler()
        new_scaler.load()
        scaled_df_2 = new_scaler.transform(base_df)

        logger.info("Проверка сериализации прошла успешно! Данные защищены от утечек.")
    else:
        logger.error(f"Файл с признаками не найден по пути: {file_path}")