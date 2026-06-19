import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import MODEL_PARAMS, TRAINING_PARAMS, FEATURES_FILE_PATH, PROJECT_ROOT
from src.dataset import get_separated_data_loaders
from src.model import QuantiGRU

logger = logging.getLogger(__name__)

def train_model():
    logger.info("=== Инициализация процесса обучения QuantiGRU (Классификация) ===")

    # 1. Определяем устройство (CUDA видеокарта или CPU)
    device = torch.device(TRAINING_PARAMS['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используемое устройство для вычислений: {device}")

    # 2. Загружаем потоки данных (DataLoaders)
    train_loader, val_loader = get_separated_data_loaders(
        sequence_length=MODEL_PARAMS['sequence_length'],
        batch_size=TRAINING_PARAMS['batch_size']
    )

    # Автоматически определяем количество фич по первому батчу
    X_sample, _ = next(iter(train_loader))
    input_size = X_sample.shape[2]

    # 3. Инициализируем модель с поддержкой Dropout из конфига
    model = QuantiGRU(
        input_size=input_size,
        hidden_size=MODEL_PARAMS['hidden_size'],
        num_layers=MODEL_PARAMS['num_layers'],
        output_size=MODEL_PARAMS['output_size'],  # В конфиге должно быть 3!
        dropout_rate=MODEL_PARAMS['dropout_rate']
    ).to(device)

    logger.info(f"Архитектура модели инициализирована: {MODEL_PARAMS['architecture']} (Dropout: {MODEL_PARAMS['dropout_rate']})")

    # 4. Функция потерь (CrossEntropy для классификации) и Оптимизатор
    # Рассчитываем веса обратно пропорционально частоте классов
    # Классы: [Short, Hold, Long] -> штрафуем за ошибку в Short/Long в 11 раз сильнее
    class_weights = torch.tensor([11.0, 1.0, 11.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAINING_PARAMS['learning_rate'],
        weight_decay=1e-3  # Штраф за слишком большие веса (L2-регуляризация)
    )

    # Количество полных циклов (проходов) по датасету
    epochs = TRAINING_PARAMS['epochs']

    # Настройки Early Stopping
    best_val_accuracy = float("-inf")
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
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # CrossEntropyLoss ожидает одномерный вектор тензоров типа Long: [batch_size]
            y_batch = y_batch.squeeze().long()

            optimizer.zero_grad()                       # Обнуление прошлых градиентов
            predictions = model(X_batch)                # Получение сырых логитов [batch_size, 3]
            loss = criterion(predictions, y_batch)      # Подсчет CrossEntropy
            loss.backward()                             # Расчет градиентов
            optimizer.step()                            # Корректировка весов

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # --- ФАЗА ВАЛИДАЦИИ ---
        model.eval()    # Режим валидации (Дропаут заморожен)
        val_loss = 0.0
        correct_preds = 0
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_batch = y_batch.squeeze().long()

                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * X_batch.size(0)

                # Считаем Accuracy на валидации для контроля качества обучения
                _, preds_classes = torch.max(predictions, dim=1)
                correct_preds += (preds_classes == y_batch).sum().item()
                total_samples += y_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_accuracy = (correct_preds / total_samples) * 100

        # Выводим логи каждую эпоху с лоссом и точностью
        logger.info(
            f"Эпоха [{epoch}/{epochs}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        # Проверка улучшения Val Loss
        """
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0 # Сбрасываем счетчик

            # Сохраняем веса лучшей модели
            models_dir = os.path.join(PROJECT_ROOT, "models")
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, "best_quanti_model.pth")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        """
        # --- ФАЗА ВЫБОРА ЛУЧШЕЙ МОДЕЛИ И EARLY STOPPING ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            patience_counter = 0  # Сбрасываем счетчик, так как есть прогресс по Accuracy!

            # Сохраняем веса лучшей модели
            models_dir = os.path.join(PROJECT_ROOT, "models")
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, "best_quanti_model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f" -> Сформирован новый максимум Accuracy: {val_accuracy:.2f}%. Веса обновлены.")
        else:
            patience_counter += 1  # Точность не выросла — увеличиваем счетчик «терпения»

        # Условие ранней остановки (теперь следит за стагнацией Accuracy)
        if patience_counter >= patience:
            logger.warning(
                f" Early Stopping сработал на epoch {epoch}! "
                f"Точность на валидации не росла {patience} эпох подряд."
            )
            break

    logger.info(f"=== Обучение завершено! ===")
    logger.info(f"Лучшая эпоха: {best_epoch} | Лучший Val Loss: {best_val_loss:.6f}")
    logger.info(f"Веса сохранены в: models/best_quanti_model.pth")


if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)
    train_model()