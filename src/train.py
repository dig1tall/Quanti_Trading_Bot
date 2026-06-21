import os
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

from src import config
from src.config import MODEL_PARAMS, TRAINING_PARAMS, PROJECT_ROOT
from src.dataset import get_separated_data_loaders
from src.model import QuantiGRU

logger = logging.getLogger(__name__)

def train_model():
    logger.info("=== Запуск промышленного процесса обучения QuantiGRU ===")

    # 2. Определение устройства вычислений
    device = torch.device(TRAINING_PARAMS['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используемое устройство: {device}")

    # 3. Загрузка потоков данных
    train_loader, val_loader = get_separated_data_loaders(
        sequence_length=MODEL_PARAMS['sequence_length'],
        batch_size=TRAINING_PARAMS['batch_size']
    )

    # Выводим распределение классов
    logger.info("--- Баланс классов в сырых датасетах ---")
    for name, loader in [("Train", train_loader), ("Validation", val_loader)]:
        if hasattr(loader.dataset, 'active_targets'):
            targets = loader.dataset.active_targets
        else:
            seq_len = MODEL_PARAMS['sequence_length']
            targets = loader.dataset.targets[seq_len : seq_len + len(loader.dataset)]

        unique, counts = np.unique(targets, return_counts=True)
        total = len(targets)
        dist_str = ", ".join([f"Класс {k}: {v/total*100:.1f}%" for k, v in zip(unique, counts)])
        logger.info(f" -> {name}: {dist_str}")

    feature_names = train_loader.dataset.feature_names
    input_size = len(feature_names)

    # 4. Инициализация модели
    model = QuantiGRU(
        input_size=input_size,
        hidden_size=MODEL_PARAMS['hidden_size'],
        num_layers=MODEL_PARAMS['num_layers'],
        output_size=MODEL_PARAMS['output_size'],
        dropout_rate=MODEL_PARAMS['dropout_rate']
    ).to(device)

    # 5. Оптимизаторы и Лосс
    # ВРЕМЕННО: принудительно зануляем сглаживание для тестирования чистых вероятностей,
    # игнорируя 0.05 из конфига. Когда таргет стабилизируется, вернешь обратно MODEL_PARAMS.get('label_smoothing', 0.0)
    smoothing = 0.0 #TRAINING_PARAMS['label_smoothing']
    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAINING_PARAMS['lr'],
        weight_decay=TRAINING_PARAMS['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    use_amp = device.type == 'cuda'
    amp_scaler = torch.amp.GradScaler('cuda') if use_amp else None

    epochs = TRAINING_PARAMS['epochs']

    # КРИТЕРИЙ УСПЕХА: Ищем максимум по Combo Score
    best_combo_score = float('-inf')
    patience = TRAINING_PARAMS['patience']
    patience_counter = 0
    best_epoch = 0
    delta = TRAINING_PARAMS['min_delta']

    logger.info(f"Параметры: Эпох={epochs} | Оптимизатор=AdamW (LR={TRAINING_PARAMS['lr']}) | Критерий Early Stopping = Combo Score (delta={delta})")

    # 6. Главный цикл обучения
    for epoch in range(1, epochs + 1):
        # --- ФАЗА ТРЕНИРОВКИ ---
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch = y_batch.squeeze().long()
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                amp_scaler.scale(loss).backward()

                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # --- ФАЗА ВАЛИДАЦИИ ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        all_confidences = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_batch = y_batch.squeeze().long()

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        predictions = model(X_batch)
                        loss = criterion(predictions, y_batch)
                else:
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)

                val_loss += loss.item() * X_batch.size(0)

                probs = torch.softmax(predictions, dim=1)
                batch_confidences, preds_classes = torch.max(probs, dim=1)

                all_preds.extend(preds_classes.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                all_confidences.extend(batch_confidences.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        # Считаем метрики качества сигналов
        val_accuracy = (np.array(all_preds) == np.array(all_targets)).mean() * 100
        val_macro_f1 = f1_score(all_targets, all_preds, average='macro')
        val_trading_f1 = f1_score(all_targets, all_preds, labels=[0, 2], average='macro')

        # РАСЧЕТ КОМБИНИРОВАННОГО СКОРА ДЛЯ КВАНТ-ОТБОРА
        current_score = 0.7 * val_trading_f1 + 0.3 * val_macro_f1
        mean_confidence = np.mean(all_confidences)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(current_score)

        logger.info(
            f"Эпоха [{epoch:02d}/{epochs:02d}] | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Macro F1: {val_macro_f1:.4f} | Trading F1: {val_trading_f1:.4f} | "
            f"**Combo Score: {current_score:.4f}** | Conf: {mean_confidence:.3f}"
        )

        # Честная проверка улучшения с учетом жесткого шага delta
        if current_score > (best_combo_score + delta):
            # Считаем чистый шаг прогресса относительно старого рекорда (для первой эпохи это текущий скор)
            improvement = current_score - (best_combo_score if best_combo_score != float('-inf') else 0.0)

            best_combo_score = current_score
            best_epoch = epoch
            patience_counter = 0

            cm = confusion_matrix(all_targets, all_preds)
            cm_text = (
                f"\n--- Матрица ошибок для Лучшей Эпохи {epoch} (Combo Score: {current_score:.4f}) ---\n"
                f"            Предсказано\n"
                f"            Short  Flat   Long\n"
                f"Факт Short:  {cm[0][0]:<5}  {cm[0][1]:<5}  {cm[0][2]:<5}\n"
                f"Факт Flat :  {cm[1][0]:<5}  {cm[1][1]:<5}  {cm[1][2]:<5}\n"
                f"Факт Long :  {cm[2][0]:<5}  {cm[2][1]:<5}  {cm[2][2]:<5}\n"
                f"-------------------------------------------------------"
            )
            logger.info(cm_text)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_macro_f1': val_macro_f1,
                'val_trading_f1': val_trading_f1,
                'combo_score': current_score,
                'feature_names': feature_names,
                'lr': current_lr
            }

            models_dir = os.path.join(PROJECT_ROOT, "models")
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, "best_quanti_model.pth")
            torch.save(checkpoint, model_path)
            logger.info(f" -> [Запись Чекпоинта] Подтверждено улучшение на +{improvement:.4f}. Модель сохранена.")
        else:
            patience_counter += 1
            if current_score > best_combo_score:
                logger.info(f" -> Скор {current_score:.4f} выше лучшего ({best_combo_score:.4f}), но шаг прироста меньше delta ({delta}). Игнорируем.")

        if patience_counter >= patience:
            logger.warning(f" [Early Stopping] Метрика Combo Score застряла и не росла {patience} эпох. Стоп.")
            break

    logger.info(f"=== Процесс обучения успешно завершен! ===")
    logger.info(f"Лучший результат на эпохе {best_epoch}: Combo Score = {best_combo_score:.4f}")


if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)
    train_model()