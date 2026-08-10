"""Main entry point orchestrating data generation, training, optimization, and backtesting pipelines."""

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
    """Sets deterministic global random seeds across random, numpy, and PyTorch backends."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global random seed set to: {seed}")

def main():
    """Orchestrates pipeline flags and executes trading engine lifecycle stages."""

    setup_logging(level=logging.INFO)
    logger.info("Initializing Quanti Trading Platform...")
    set_seed(42)

    data_engine = DataEngine()
    model_engine = ModelEngine()

    RUN_DATA_PIPELINE = True
    RUN_MODEL_TRAINING = True
    RUN_OPTIMIZATION = True
    RUN_BACKTESTING = True
    RUN_LIVE = False

    # 1. data preparation phase
    if RUN_DATA_PIPELINE:
        data_engine.run_pipeline()

    # 2. model training phase
    if RUN_MODEL_TRAINING:
        model_engine.run_training()

    # 2.5. parameter optimization phase
    if RUN_BACKTESTING and RUN_OPTIMIZATION:
        logger.info("Executing strategy hyperparameter optimization search...")
        if run_optimization_search is not None:
            try:
                best_found_params = run_optimization_search(config.VAL_FEATURES_PATH)
                config.update_backtest_params(best_found_params)
                logger.info("Optimization complete. Active backtest configuration updated.")
            except Exception as e:
                logger.error(f"Error during parameter optimization: {e}")
        else:
            logger.error("Unable to import run_optimization_search module.")

    # 3. strategy backtesting phase
    if RUN_BACKTESTING:
        logger.info("Executing historical strategy backtest...")
        model_engine.run_backtest()

    # 4. live execution phase
    if RUN_LIVE:
        logger.info("Switching platform to live execution mode...")
        run_infinite_loop()

    logger.info("Quanti system pipeline execution complete.")

if __name__ == "__main__":
    main()