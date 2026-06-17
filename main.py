import logging
import random
import numpy as np
import torch

from src.config import setup_logging
from src.data_engine import DataEngine
from src.model_engine import ModelEngine

# Инициализация логгера для main скрипта
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """
    Фиксирует все источники случайности для обеспечения
    100% воспроизводимости результатов ML-пайплайна.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Для мульти-GPU систем

    # Жестко заставляем алгоритмы CUDA быть детерминированными
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Глобальный Random Seed зафиксирован на значении: {seed}")

def main():
    # Инициализация глобальной настройки логирования (вывод в консоль)
    setup_logging(level=logging.INFO)

    logger.info("=== Трейдинг-платформа Quanti ===")

    # Фиксируем сид ПЕРЕД инициализацией и запуском движков
    set_seed(42)

    # 1. Инициализируем компоненты архитектуры
    data_engine = DataEngine()
    model_engine = ModelEngine()

    # Флаги управления пайплайнами (можно переключать в True/False при необходимости)
    RUN_DATA_PIPELINE = True   # Нужно ли скачивать и собирать фичи заново
    RUN_MODEL_TRAINING = True  # Нужно ли запускать цикл обучения нейросети
    RUN_BACKTESTING = True     # Нужно ли проводить финальный бэктест

    # --- ФАЗА 1: Конвейер Данных ---
    if RUN_DATA_PIPELINE:
        logger.info("[ФАЗА 1] Запуск обработки рыночных данных...")
        data_engine.run_pipeline()
    else:
        logger.info("[ФАЗА 1] Пропуск обработки данных (используем кэш Parquet).")

    # --- ФАЗА 2: Обучение Модели ---
    if RUN_MODEL_TRAINING:
        logger.info("[ФАЗА 2] Запуск обучения нейросети QuantiGRU...")
        model_engine.run_training()
    else:
        logger.info("[ФАЗА 2] Пропуск этапа обучения сети.")

    # --- ФАЗА 3: Бэктестинг и Аналитика ---
    if RUN_BACKTESTING:
        logger.info("[ФАЗА 3] Симуляция торговой стратегии на исторических данных...")
        model_engine.run_backtest()

    logger.info("==================================================")
    logger.info("   РАБОТА ВСЕХ СИСТЕМ QUANTIPY ЗАВЕРШЕНА    ")
    logger.info("==================================================")

if __name__ == "__main__":
    main()