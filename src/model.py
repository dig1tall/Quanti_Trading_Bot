"""Module defining the QuantiGRU neural network architecture and self-test verification routines."""

import logging
import torch
import torch.nn as nn
import pandas as pd

from src.config import MODEL_PARAMS, TRAINING_PARAMS, setup_logging


logger = logging.getLogger(__name__)


class QuantiGRU(nn.Module):
    """GRU network architecture optimized for daily crypto sequence classification tasks.

        Accepts 3D sequence tensors of shape: [Batch_Size, Sequence_Length, Input_Size]
        Returns raw logits of shape: [Batch_Size, Output_Size]
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size=3, dropout_rate=0.0):
        """Initializes GRU layers, dropout, and linear projection head.

            Args:
                input_size: Number of input features per time step.
                hidden_size: Number of features in GRU hidden states.
                num_layers: Number of recurrent layers.
                output_size: Dimensionality of output projection (e.g., 2 classes).
                dropout_rate: Dropout probability between recurrent layers and linear head.
        """

        super(QuantiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size) # Теперь на выходе 3 логита

    def forward(self, x):
        """Performs forward pass over sequence tensor."""

        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


if __name__ == "__main__":
    setup_logging()

    logger.info("Running standalone test for QuantiGRU architecture...")

    try:
        from src import config
        df_temp = pd.read_parquet(config.TRAIN_FEATURES_PATH)

        # Исключаем служебные колонки, все остальное — фичи (должен быть 21 признак)
        feature_cols = [col for col in df_temp.columns if col not in ['target', 'Date', 'date']]
        INPUT_SIZE = len(feature_cols)
        logger.info(f"Dynamically calculated input feature count from dataset: {INPUT_SIZE}")
    except Exception as e:
        INPUT_SIZE = 21  # Наш новый базис ортогональных фич под 1d
        logger.warning(f"Target parquet file not accessible ({e}). Falling back to default: {INPUT_SIZE}")

    HIDDEN_SIZE = MODEL_PARAMS.get('hidden_size', 64)
    NUM_LAYERS = MODEL_PARAMS.get('num_layers', 2)
    OUTPUT_SIZE = MODEL_PARAMS.get('output_size', 2)  # Теперь ожидаем 2 класса
    DROPOUT_RATE = MODEL_PARAMS.get('dropout_rate', 0.2)

    try:
        model = QuantiGRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            output_size=OUTPUT_SIZE,
            dropout_rate=DROPOUT_RATE
        )
        logger.info(f"Model successfully instantiated: hidden_size={HIDDEN_SIZE}, num_layers={NUM_LAYERS}, dropout={DROPOUT_RATE}")

        device = torch.device(TRAINING_PARAMS.get('device', 'cpu') if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logger.info(f"Model transferred to target execution device: {device}")

        seq_len = MODEL_PARAMS.get('sequence_length', 30)
        test_input = torch.randn(32, seq_len, INPUT_SIZE).to(device)
        logger.info(f"Generated test tensor batch of shape: {list(test_input.shape)}")

        test_output = model(test_input)

        logger.info("Tensor dimension verification complete.")
        logger.info(f"Input batch shape: {list(test_input.shape)}")
        logger.info(f"Output logits shape:   {list(test_output.shape)} -> (target classes: {OUTPUT_SIZE})")

        if test_output.shape[1] != 2:
            logger.error(f"Output dimension mismatch")
        else:
            logger.info("Standalone QuantiGRU test executed successfully.")

    except Exception as e:
        logger.error(f"Critical failure during QuantiGRU architecture test: {e}", exc_info=True)