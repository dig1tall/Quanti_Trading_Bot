import os
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

from src import config
from src.config import MODEL_PARAMS, TRAINING_PARAMS, PROJECT_ROOT
from src.dataset import get_separated_data_loaders
from src.model import QuantiGRU

logger = logging.getLogger(__name__)

class QuantiTradingLoss(nn.Module):
    def __init__(self, class_weights, alpha_ordinal=1.5, gamma=2.0):
        super(QuantiTradingLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.alpha_ordinal = alpha_ordinal

        # Матрица жестких штрафов за неверное направление (Размерность 3х3)
        # Строки - факт, столбцы - прогноз
        # За ошибку Short <-> Long караем со всей строгостью
        self.cost_matrix = torch.tensor([
            [0.0, 1.0, 2.0],  # Факт 0 (Short): Ошибка в Long (2) весит 2.0
            [1.0, 0.0, 1.0],  # Факт 1 (Flat): Стандарт
            [2.0, 1.0, 0.0]   # Факт 2 (Long): Ошибка в Short (0) весит 2.0
        ], dtype=torch.float32)

    def forward(self, logits, targets):
        # 1. СТАНДАРТНЫЙ МУЛЬТИКЛАССОВЫЙ FOCAL LOSS
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Используем веса классов
        batch_weights = self.class_weights[targets]
        focal_loss = batch_weights * ((1 - pt) ** self.gamma) * ce_loss
        focal_loss = focal_loss.mean()

        # 2. НАПРАВЛЕННЫЙ ТОРГОВЫЙ ШТРАФ ПО МАТРИЦЕ СТОИМОСТИ
        probs = F.softmax(logits, dim=1) # [Batch, 3]

        # Вытаскиваем нужные строки из cost_matrix под текущие таргеты батча
        # P.S. Матрица должна быть на том же девайсе, что и логиты
        batch_cost = self.cost_matrix.to(logits.device)[targets] # [Batch, 3]

        # Перемножаем вероятности предсказаний на штрафные коэффициенты
        # Если факт Short, а probs[2] (Long) высокая -> penalty улетит в космос
        directional_penalty = torch.sum(probs * batch_cost, dim=1).mean()

        # Общий лосс
        return focal_loss + self.alpha_ordinal * directional_penalty

def train_model():
    logger.info("=== Запуск стабилизированного процесса обучения QuantiGRU (Focal + Ordinal) ===")

    device = torch.device(TRAINING_PARAMS.get('device', 'cpu') if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_separated_data_loaders()

    # Измени этот блок в train_model():
    logger.info("--- Баланс классов и расчет весов ---")
    train_targets = train_loader.dataset.targets.numpy()
    unique, counts = np.unique(train_targets, return_counts=True)
    total_samples = len(train_targets)

    # ВМЕСТО АВТОМАТИКИ: Намеренно занижаем вес флэта, чтобы модель хотела искать Long/Short
    # Индексы: [Short, Flat, Long]
    class_weights = np.array([1.0, 0.8, 1.0], dtype=np.float32)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    for k, v, w in zip(unique, counts, class_weights):
        logger.info(f" ->  Класс {k}: {v} шт ({v/total_samples*100:.1f}%) | Принудительный Вес: {w:.4f}")

    feature_names = train_loader.dataset.feature_names
    input_size = len(feature_names)

    model = QuantiGRU(
        input_size=input_size,
        hidden_size=MODEL_PARAMS['hidden_size'],
        num_layers=MODEL_PARAMS['num_layers'],
        output_size=MODEL_PARAMS['output_size'],
        dropout_rate=MODEL_PARAMS['dropout_rate']
    ).to(device)

    # Инициализируем наш кастомный комбинированный лосс
    criterion = QuantiTradingLoss(class_weights=class_weights_tensor, alpha_ordinal=0.6, gamma=2.0)

    optimizer = optim.AdamW(model.parameters(), lr=TRAINING_PARAMS['learning_rate'], weight_decay=TRAINING_PARAMS['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    use_amp = device.type == 'cuda'
    amp_scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_combo_score = float('-inf')
    patience = TRAINING_PARAMS['patience']
    patience_counter = 0
    best_epoch = 0
    delta = TRAINING_PARAMS['min_delta']

    for epoch in range(1, TRAINING_PARAMS['epochs'] + 1):
        # --- TRAIN LOOP ---
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

        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0.0
        all_preds, all_targets, all_confidences = [], [], []

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
        val_accuracy = (np.array(all_preds) == np.array(all_targets)).mean() * 100
        val_macro_f1 = f1_score(all_targets, all_preds, average='macro', labels=[0, 1, 2], zero_division=0)

        current_score = val_macro_f1
        scheduler.step(current_score)

        logger.info(
            f"Эпоха [{epoch:02d}/{TRAINING_PARAMS['epochs']:02d}] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Accuracy: {val_accuracy:.2f}% | **Macro F1: {current_score:.4f}**"
        )

        if current_score > (best_combo_score + delta):
            best_combo_score = current_score
            best_epoch = epoch
            patience_counter = 0

            cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
            cm_text = (
                f"\n--- Матрица ошибок 3x3 для Лучшей Эпохи {epoch} ---\n"
                f"             Предсказано\n"
                f"             Short(0)   Flat(1)    Long(2)\n"
                f"Факт Short:  {cm[0][0]:<10} {cm[0][1]:<10} {cm[0][2]:<10}\n"
                f"Факт Flat :  {cm[1][0]:<10} {cm[1][1]:<10} {cm[1][2]:<10}\n"
                f"Факт Long :  {cm[2][0]:<10} {cm[2][1]:<10} {cm[2][2]:<10}\n"
                f"-------------------------------------------------------"
            )
            logger.info(cm_text)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'feature_names': feature_names,
                'val_macro_f1': val_macro_f1
            }
            torch.save(checkpoint, os.path.join(PROJECT_ROOT, "models", "best_quanti_model.pth"))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.warning(f" Early Stopping по макро F1.")
            break

    logger.info(f"Процесс завершен. Лучшая epoch {best_epoch} с Macro F1 = {best_combo_score:.4f}")

if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging(level=logging.INFO)
    train_model()