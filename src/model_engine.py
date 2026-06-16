import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src import config
from src.config import MODEL_PARAMS, TRAINING_PARAMS, FEATURES_FILE_PATH, PROJECT_ROOT
from src.dataset import get_data_loaders
from src.model import QuantiGRU

# Инициализация логгера для модуля движка модели
logger = logging.getLogger(__name__)


class ModelEngine:
    """
    Класс-диспетчер (Engine), управляющий жизненным циклом нейросети QuantiGRU.
    Отвечает за запуск обучения (обучение + валидация + Early Stopping) и запуск бэктестинга.
    """
    def __init__(self):
        # Подтягиваем параметры из центрального конфига
        self.sequence_length = MODEL_PARAMS['sequence_length']
        self.hidden_size = MODEL_PARAMS['hidden_size']
        self.num_layers = MODEL_PARAMS['num_layers']
        self.output_size = MODEL_PARAMS['output_size']

        self.batch_size = TRAINING_PARAMS['batch_size']
        self.train_split = TRAINING_PARAMS['train_split']
        self.learning_rate = TRAINING_PARAMS['learning_rate']
        self.epochs = TRAINING_PARAMS['epochs']

        # Определяем девайс для вычислений (CUDA / CPU)
        self.device = torch.device(TRAINING_PARAMS['device'] if torch.cuda.is_available() else 'cpu')

        # Путь для сохранения весов модели
        self.model_dir = os.path.join(PROJECT_ROOT, "models")
        self.model_path = os.path.join(self.model_dir, "best_quanti_model.pth")

        # Переменные для ленивой инициализации лоадеров и модели
        self.train_loader = None
        self.val_loader = None
        self.model = None

    def _prepare_data_and_model(self) -> bool:
        """Внутренний метод для сборки лоадеров и инициализации структуры модели."""
        if not os.path.exists(FEATURES_FILE_PATH):
            logger.error(f"Файл фич не найден по пути: {FEATURES_FILE_PATH}. Запусти сначала DataEngine!")
            return False

        # 1. Загружаем потоки данных
        self.train_loader, self.val_loader = get_data_loaders(
            file_path=FEATURES_FILE_PATH,
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            train_split=self.train_split
        )

        # Автоматически определяем input_size (количество фич) по первому батчу
        X_sample, _ = next(iter(self.train_loader))
        input_size = X_sample.shape[2] # 17 фич

        # 2. Инициализируем модель и переносим её на видеокарту/процессор
        self.model = QuantiGRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size
        ).to(self.device)

        logger.info(f"Архитектура модели QuantiGRU инициализирована. Входных фич: {input_size}")
        return True

    def train_pipeline(self, patience: int = 7) -> None:
        """Запускает полный цикл обучения модели с валидацией и механизмом Early Stopping."""
        logger.info("=== Запуск пайплайна обучения модели ===")

        if not self._prepare_data_and_model():
            return

        logger.info(f"Используемое устройство: {self.device}")
        logger.info(f"Старт обучения. Максимум эпох: {self.epochs} | Терпение (Early Stopping): {patience}")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        for epoch in range(1, self.epochs + 1):
            # --- ФАЗА ТРЕНИРОВКИ ---
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(self.train_loader.dataset)

            # --- ФАЗА ВАЛИДАЦИИ ---
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item() * X_batch.size(0)

            val_loss /= len(self.val_loader.dataset)

            logger.info(f"Эпоха [{epoch}/{self.epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            # Проверка улучшения ошибки на валидации
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                # Сохранение весов
                os.makedirs(self.model_dir, exist_ok=True)
                torch.save(self.model.state_dict(), self.model_path)
            else:
                patience_counter += 1

            # Триггер ранней остановки
            if patience_counter >= patience:
                logger.warning(f"Early Stopping сработал на эпохе {epoch}! "
                               f"Ошибка на валидации не падала {patience} эпох подряд.")
                break

        logger.info(f"=== Обучение завершено успешно! ===")
        logger.info(f"Лучшая эпоха: {best_epoch} | Лучший Val Loss: {best_val_loss:.6f}")
        logger.info(f"Веса лучшей конфигурации сохранены в: {self.model_path}")

    def backtest_pipeline(self) -> None:
        """Загружает сохраненные веса лучшей модели и проводит бэктестинг на валидационной выборке."""
        logger.info("=== Запуск пайплайна бэктестинга (Аналитика модели) ===")

        # Если лоадеры или модель еще не инициализированы (например, запустили сразу бэктест без обучения)
        if self.model is None or self.val_loader is None:
            if not self._prepare_data_and_model():
                return

        if not os.path.exists(self.model_path):
            logger.error(f"Файл весов {self.model_path} не найден! Нечего тестировать. Сначала обучи модель.")
            return

        # Загрузка весов (безопасный режим weights_only=True для новых версий PyTorch)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        logger.info(f"Веса из {self.model_path} успешно загружены в модель. Симуляция рынка...")

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                predictions = self.model(X_batch)

                all_preds.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(y_batch.numpy().flatten())

        preds = np.array(all_preds)
        targets = np.array(all_targets)

        # Торговая логика: Прогноз > 0 -> Long (1), Иначе -> Cash (0)
        signals = np.where(preds > 0, 1, 0)
        strategy_returns = signals * targets

        # Расчет накопленных сложных процентов (Кумулятивная доходность)
        cum_strategy = np.prod(1.0 + strategy_returns) - 1.0
        cum_market = np.prod(1.0 + targets) - 1.0

        # Точность угадывания знака движения цены
        direction_correct = np.sum(np.sign(preds) == np.sign(targets)) / len(targets)

        logger.info("--- Итоговые результаты бэктеста на валидационном периоде ---")
        logger.info(f"Продолжительность симуляции: {len(targets)} дней.")
        logger.info(f"Точность направления тренда (Accuracy): {direction_correct * 100:.2f}%")
        logger.info(f"Доходность пассивного удержания (Buy & Hold): {cum_market * 100:.2f}%")
        logger.info(f"Доходность торгового ИИ-алгоритма Quanti:    {cum_strategy * 100:.2f}%")

        if cum_strategy > cum_market:
            logger.info("🔥 Результат: Алгоритм модели Quanti успешно переиграл маркет-тренд!")
        else:
            logger.warning("📉 Результат: В текущей конфигурации модель уступила пассивному удержанию.")


# --- АВТОНОМНЫЙ ТЕСТ ДИСПЕТЧЕРА МОДЕЛИ ---
if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)

    logger.info("=== Автономный тест ModelEngine ===")

    # Создаем экземпляр движка модели
    model_engine = ModelEngine()

    # Сначала прогоняем пайплайн обучения с Early Stopping
    model_engine.train_pipeline(patience=7)

    # Сразу после этого запускаем бэктест на сохраненных весах
    model_engine.backtest_pipeline()