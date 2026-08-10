"""Module defining the ModelEngine coordinator for managing training and backtesting pipelines."""

import logging
from src.train import train_model
from src.backtest import run_backtest

logger = logging.getLogger(__name__)

class ModelEngine:
    """Coordinator class managing execution of the Quanti ML training and evaluation pipelines."""

    def __init__(self):
        pass

    def run_training(self) -> None:
        """Executes the isolated model training pipeline."""

        logger.info("Initializing model training pipeline phase...")
        train_model()

    def run_backtest(self) -> None:
        """Executes the isolated backtesting and strategy evaluation pipeline."""

        logger.info("Initializing backtesting strategy pipeline phase...")
        run_backtest()


if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)

    logger.info("Running ModelEngine in standalone execution mode...")
    engine = ModelEngine()
    engine.run_training()
    engine.run_backtest()