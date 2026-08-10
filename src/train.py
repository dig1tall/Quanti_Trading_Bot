"""Module for training the QuantiGRU neural network using Focal Loss and ordinal penalty optimization."""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

from src.config import MODEL_PARAMS, TRAINING_PARAMS, PROJECT_ROOT
from src.dataset import get_separated_data_loaders
from src.model import QuantiGRU

logger = logging.getLogger(__name__)

class QuantiTradingLoss(nn.Module):
    """Custom loss function combining Focal Loss with a directional cost penalty matrix."""

    def __init__(self, class_weights, alpha_ordinal=1.5, gamma=2.0):
        """Initializes class weights, focal scaling, and cost matrix penalties.

            Args:
                class_weights: Tensor containing weights for each target class.
                alpha_ordinal: Weight multiplier for directional penalty.
                gamma: Focal loss focusing parameter.
        """

        super(QuantiTradingLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.alpha_ordinal = alpha_ordinal

        # penalty matrix penalizing opposing directional misclassifications
        self.cost_matrix = torch.tensor([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0]
        ], dtype=torch.float32)

    def forward(self, logits, targets):
        """Calculates total combined loss."""

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        batch_weights = self.class_weights[targets]
        focal_loss = batch_weights * ((1 - pt) ** self.gamma) * ce_loss
        focal_loss = focal_loss.mean()

        probs = F.softmax(logits, dim=1) # [Batch, 3]

        batch_cost = self.cost_matrix.to(logits.device)[targets] # [Batch, 3]

        directional_penalty = torch.sum(probs * batch_cost, dim=1).mean()

        return focal_loss + self.alpha_ordinal * directional_penalty

def train_model():
    """Executes model training and validation loops with early stopping and checkpointing."""

    logger.info("Initializing QuantiGRU training process (Focal + Ordinal Loss)...")

    device = torch.device(TRAINING_PARAMS.get('device', 'cpu') if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_separated_data_loaders()

    logger.info("Calculating class distributions and target loss weights...")
    train_targets = train_loader.dataset.targets.numpy()
    unique, counts = np.unique(train_targets, return_counts=True)
    total_samples = len(train_targets)

    class_weights = total_samples / (len(unique) * counts.astype(np.float32))
    class_weights = class_weights / class_weights[0]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    for k, v, w in zip(unique, counts, class_weights):
        logger.info(f" ->  Class {k}: {v} samples ({v/total_samples*100:.1f}%) | Assigned Weight: {w:.4f}")

    feature_names = train_loader.dataset.feature_names
    input_size = len(feature_names)

    model = QuantiGRU(
        input_size=input_size,
        hidden_size=MODEL_PARAMS['hidden_size'],
        num_layers=MODEL_PARAMS['num_layers'],
        output_size=MODEL_PARAMS['output_size'],
        dropout_rate=MODEL_PARAMS['dropout_rate']
    ).to(device)

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
        # train phase
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

        # validation phase
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
            f"Epoch [{epoch:02d}/{TRAINING_PARAMS['epochs']:02d}] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Accuracy: {val_accuracy:.2f}% | **Macro F1: {current_score:.4f}**"
        )

        if current_score > (best_combo_score + delta):
            best_combo_score = current_score
            best_epoch = epoch
            patience_counter = 0

            cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
            cm_text = (
                f"\n--- Confusion Matrix (3x3) for Best Epoch {epoch} ---\n"
                f"             Predicted\n"
                f"              Short(0)   Flat(1)    Long(2)\n"
                f"Actual Short:  {cm[0][0]:<10} {cm[0][1]:<10} {cm[0][2]:<10}\n"
                f"Actual Flat :  {cm[1][0]:<10} {cm[1][1]:<10} {cm[1][2]:<10}\n"
                f"Actual Long :  {cm[2][0]:<10} {cm[2][1]:<10} {cm[2][2]:<10}\n"
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
            logger.warning(f"Early stopping triggered on Macro F1 stagnancy.")
            break

    logger.info(f"Training process completed. Best epoch: {best_epoch} with Macro F1 = {best_combo_score:.4f}")

if __name__ == "__main__":
    from src.config import setup_logging

    setup_logging(level=logging.INFO)
    train_model()