import logging
from src.config import setup_logging
from src.data_engine import DataEngine

# Создаем логгер для текущего файла (main)
logger = logging.getLogger(__name__)

def main():
    # Инициализируем глобальные настройки логирования (вывод в консоль)
    setup_logging(level=logging.INFO)

    logger.info("=== Трейдинг-платформа Quanti ===")

    # Создаем движок конвейера данных и запускаем его одной командой
    engine = DataEngine()
    engine.run_pipeline()

if __name__ == "__main__":
    main()