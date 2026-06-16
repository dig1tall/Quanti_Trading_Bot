import logging
from src.train import train_model
from src.backtest import run_backtest

logger = logging.getLogger(__name__)

class ModelEngine:
    """
    Класс-диспетчер (Engine), управляющий ML-конвейером Quanti.
    Просто связывает воедино готовые модули обучения и бэктестинга.
    """
    def __init__(self):
        pass

    def run_training(self) -> None:
        """Запуск изолированного процесса обучения."""
        logger.info("=== [Engine] Инициализация этапа обучения ===")
        train_model()

    def run_backtest(self) -> None:
        """Запуск изолированного процесса бэктестинга."""
        logger.info("=== [Engine] Инициализация этапа бэктестинга ===")
        run_backtest()


# --- АВТОНОМНЫЙ ТЕСТ ДИСПЕТЧЕРА ---
if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)

    engine = ModelEngine()
    # Для теста можно дёрнуть что-то одно или всё вместе
    engine.run_training()
    engine.run_backtest()