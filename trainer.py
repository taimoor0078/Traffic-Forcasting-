"""
Step 8: Training.

Trains a compiled Keras forecasting model with the full set of production
callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard,
optional mixed precision, and a fixed global random seed for reproducibility.
"""

from __future__ import annotations

import os
from datetime import datetime

import tensorflow as tf

from src.utils.logger import get_logger
from src.utils.seed import set_global_seed


class Trainer:
    def __init__(self, config):
        self.cfg = config
        self.logger = get_logger("Trainer", config.paths.logs_dir)
        self.training_dir = config.paths.training_dir

        set_global_seed(config.project.seed)

        if config.training.mixed_precision:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                self.logger.info("Mixed precision policy enabled (mixed_float16).")
            except Exception as e:  # pragma: no cover - depends on hardware
                self.logger.warning("Could not enable mixed precision: %s", e)

    def _build_callbacks(self, model_name: str):
        t = self.cfg.training
        run_dir = os.path.join(self.training_dir, model_name)
        os.makedirs(run_dir, exist_ok=True)

        checkpoint_path = os.path.join(run_dir, f"best_{model_name}.keras")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=t.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=t.reduce_lr_factor,
                patience=t.reduce_lr_patience,
                min_lr=t.min_lr,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        if t.use_tensorboard:
            tb_dir = os.path.join(run_dir, "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=1))

        csv_log_path = os.path.join(run_dir, "training_log.csv")
        callbacks.append(tf.keras.callbacks.CSVLogger(csv_log_path))

        return callbacks, checkpoint_path

    def train(self, model: tf.keras.Model, model_name: str, X_train, y_train, X_val, y_val):
        t = self.cfg.training
        callbacks, checkpoint_path = self._build_callbacks(model_name)

        self.logger.info("Starting training for %s | epochs=%d batch_size=%d", model_name, t.epochs, t.batch_size)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=t.epochs,
            batch_size=t.batch_size,
            callbacks=callbacks,
            verbose=2,
        )
        self.logger.info("Finished training %s. Best checkpoint: %s", model_name, checkpoint_path)
        return history, checkpoint_path
