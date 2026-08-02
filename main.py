import logging
import random
import numpy as np
import torch

import src.config as config
from src.config import setup_logging
from src.data_engine import DataEngine
from src.model_engine import ModelEngine
from src.execution import run_infinite_loop

try:
    from src.backtest_optimize import run_optimization_search
except ImportError:
    run_optimization_search = None

logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Глобальный Random Seed зафиксирован на значении: {seed}")

def main():
    setup_logging(level=logging.INFO)
    logger.info("=== Трейдинг-платформа Quanti ===")
    set_seed(42)

    data_engine = DataEngine()
    model_engine = ModelEngine()

    # === НАСТРОЙКА ФЛАГОВ ===
    RUN_DATA_PIPELINE = True   # Используем готовый кэш данных
    RUN_MODEL_TRAINING = True  # Используем уже обученную модель
    RUN_OPTIMIZATION = True     # ВКЛЮЧАЕМ оптимизатор под новую логику выхода!
    RUN_BACKTESTING = True      # ВКЛЮЧАЕМ финальный бэктест
    RUN_LIVE = False

    # --- ФАЗА 1 ---
    if RUN_DATA_PIPELINE:
        data_engine.run_pipeline()
    # --- ФАЗА 2 ---
    if RUN_MODEL_TRAINING:
        model_engine.run_training()

    # --- ФАЗА 2.5: Оптимизация Параметров Бэктеста ---
    if RUN_BACKTESTING and RUN_OPTIMIZATION:
        logger.info("[ФАЗА 2.5] Запуск поиска оптимальных параметров стратегии...")
        if run_optimization_search is not None:
            try:
                best_found_params = run_optimization_search(config.VAL_FEATURES_PATH)
                config.update_backtest_params(best_found_params)
                logger.info("[ФАЗА 2.5] Оптимизация завершена. Конфиг в памяти успешно обновлен.")
            except Exception as e:
                logger.error(f"[ФАЗА 2.5] Критическая ошибка: {e}")
        else:
            logger.error("[ФАЗА 2.5] Не удалось импортировать run_optimization_search!")

    # --- ФАЗА 3: Бэктестинг и Аналитика ---
    if RUN_BACKTESTING:
        logger.info("[ФАЗА 3] Симуляция торговой стратегии на исторических данных...")
        model_engine.run_backtest()

    if RUN_LIVE:
        logger.info("[ФАЗА 4] Перевод платформы в режим LIVE постоянного трейдинга...")
        run_infinite_loop()

    logger.info("==================================================")
    logger.info("   РАБОТА ВСЕХ СИСТЕМ QUANTI ЗАВЕРШЕНА    ")
    logger.info("==================================================")

if __name__ == "__main__":
    main()