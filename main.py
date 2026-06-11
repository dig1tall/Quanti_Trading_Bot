import logging
from src.config import setup_logging
from src.data_engine import DataEngine

# Инициализация логгера для main скрипта
logger = logging.getLogger(__name__)

def main():
    # Инициализация глобальной настройки логирования (вывод в консоль)
    setup_logging(level=logging.INFO)

    logger.info("=== Трейдинг-платформа Quanti ===")

    # Создание диспетчера конвейера данных и запуск его
    engine = DataEngine()
    engine.run_pipeline()

if __name__ == "__main__":
    main()