import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import MODEL_PARAMS, TRAINING_PARAMS, FEATURES_FILE_PATH, PROJECT_ROOT
from src.dataset import get_data_loaders
from src.model import QuantiGRU
from src.model import DirectionalMSELoss

logger = logging.getLogger(__name__)

def train_model():
    logger.info("=== Инициализация процесса обучения QuantiGRU ===")

    # 1. Определяем устройство (CUDA видеокарта или CPU)
    device = torch.device(TRAINING_PARAMS['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используемое устройство для вычислений: {device}")

    # 2. Загружаем потоки данных (DataLoaders)
    train_loader, val_loader = get_data_loaders(
        file_path=FEATURES_FILE_PATH,
        sequence_length=MODEL_PARAMS['sequence_length'],
        batch_size=TRAINING_PARAMS['batch_size'],
        train_split=TRAINING_PARAMS['train_split']
    )

    # Автоматически определяем количество фич по первому батчу
    X_sample, _ = next(iter(train_loader))
    input_size = X_sample.shape[2]

    # 3. Инициализируем модель с поддержкой Dropout из конфига
    model = QuantiGRU(
        input_size=input_size,
        hidden_size=MODEL_PARAMS['hidden_size'],
        num_layers=MODEL_PARAMS['num_layers'],
        output_size=MODEL_PARAMS['output_size'],
        dropout_rate=MODEL_PARAMS['dropout_rate']
    ).to(device)

    logger.info(f"Архитектура модели инициализирована: {MODEL_PARAMS['architecture']} (Dropout: {MODEL_PARAMS['dropout_rate']})")

    # 4. Функция потерь (MSE для регрессии) и Оптимизатор
    criterion = DirectionalMSELoss(direction_alpha=3.0)
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_PARAMS['learning_rate'])

    # Количество полных циклов (проходов) по датасету
    epochs = TRAINING_PARAMS['epochs']

    # Настройки Early Stopping
    best_val_loss = float('inf')
    patience = TRAINING_PARAMS['patience']  # Сколько эпох ждем улучшения Val Loss перед остановкой
    patience_counter = 0
    best_epoch = 0

    logger.info(f"Старт обучения. Максимум эпох: {epochs} | Patience (Early Stopping): {patience}")

    # 5. Главный цикл обучения
    for epoch in range(1, epochs + 1):
        # --- ФАЗА ТРЕНИРОВКИ ---
        model.train()   # Режим обучения (Дропаут активно режет связи)
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) # Перенос RAM -> VRAM

            optimizer.zero_grad()                       # Обнуление прошлых градиентов
            predictions = model(X_batch)                # Получение предсказаний от модели
            loss = criterion(predictions, y_batch)      # Подсчет потерь (MSE)
            loss.backward()     # Расчет градиентов
            optimizer.step()    # Корректировка весов

            train_loss += loss.item() * X_batch.size(0)     # Подсчет ошибки всего батча

        train_loss /= len(train_loader.dataset)             # Средняя ошибка на одну свечу

        # --- ФАЗА ВАЛИДАЦИИ ---
        model.eval()    # Режим валидации (Дропаут замораживается, веса работают на полную)
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        # Выводим логи каждую эпоху, чтобы четко видеть момент остановки
        logger.info(f"Эпоха [{epoch}/{epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Проверка улучшения Val Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0 # Сбрасываем счетчик, так как нашли улучшение

            # Сохраняем веса лучшей модели
            models_dir = os.path.join(PROJECT_ROOT, "models")
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, "best_quanti_model.pth")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        # Условие ранней остановки
        if patience_counter >= patience:
            logger.warning(f" Early Stopping сработал на epoch {epoch}! Ошибка на валидации не падала {patience} эпох подряд.")
            break

    logger.info(f"=== Обучение завершено! ===")
    logger.info(f"Лучшая эпоха: {best_epoch} | Лучший Val Loss: {best_val_loss:.6f}")
    logger.info(f"Веса сохранены в: models/best_quanti_model.pth")


if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)
    train_model()