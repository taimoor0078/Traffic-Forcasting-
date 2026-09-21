"""
Step 11: Save Models.

Copies the best checkpoint for each architecture into saved_models/ using
the canonical naming (best_lstm.keras, best_gru.keras, best_transformer.keras),
alongside the fitted scaler, run metadata, and a snapshot of the configuration.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

from src.utils.logger import get_logger


class ModelSaver:
    def __init__(self, config):
        self.cfg = config
        self.logger = get_logger("ModelSaver", config.paths.logs_dir)
        self.saved_models_dir = config.paths.saved_models_dir

    def save_final_model(self, checkpoint_path: str, model_name: str) -> str:
        dest = os.path.join(self.saved_models_dir, f"best_{model_name}.keras")
        shutil.copyfile(checkpoint_path, dest)
        self.logger.info("Saved final model %s -> %s", model_name, dest)
        return dest

    def save_scaler(self, preprocessor) -> str:
        dest = os.path.join(self.saved_models_dir, "scaler.pkl")
        preprocessor.save_scaler(dest)
        return dest

    def save_metadata(self, feature_columns, target_column: str, best_model: str, comparison_summary: dict) -> str:
        metadata = {
            "created_at": datetime.now().isoformat(),
            "feature_columns": feature_columns,
            "target_column": target_column,
            "window_size": self.cfg.preprocessing.sliding_window_size,
            "forecast_horizon": self.cfg.preprocessing.forecast_horizon,
            "best_model": best_model,
            "comparison_summary": comparison_summary,
        }
        dest = os.path.join(self.saved_models_dir, "metadata.json")
        with open(dest, "w") as f:
            json.dump(metadata, f, indent=2)
        self.logger.info("Saved metadata to %s", dest)
        return dest

    def save_config_snapshot(self) -> str:
        dest = os.path.join(self.saved_models_dir, "config_snapshot.yaml")
        self.cfg.save_snapshot(dest)
        return dest
