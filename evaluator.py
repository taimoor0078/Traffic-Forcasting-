"""
Step 9: Evaluation.
Step 10: Model Comparison.

Computes MAE/MSE/RMSE/MAPE/R^2, generates diagnostic plots (loss curve,
actual-vs-prediction, residuals, error distribution), writes a prediction
CSV per model, and automatically determines the best model across all
trained architectures.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from src.utils.logger import get_logger


class Evaluator:
    def __init__(self, config):
        self.cfg = config
        self.logger = get_logger("Evaluator", config.paths.logs_dir)
        self.eval_dir = config.paths.evaluation_dir
        self.plots_dir = config.paths.plots_dir
        self.results_dir = config.paths.results_dir
        self.metric_for_best = config.inference.best_model_metric

    # ------------------------------------------------------------------
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        # avoid div-by-zero in MAPE by masking near-zero true values
        nonzero_mask = np.abs(y_true) > 1e-6
        mape = (
            mean_absolute_percentage_error(y_true[nonzero_mask], y_pred[nonzero_mask]) * 100
            if nonzero_mask.any() else float("nan")
        )
        r2 = r2_score(y_true, y_pred)
        return {"mae": float(mae), "mse": float(mse), "rmse": rmse, "mape": float(mape), "r2": float(r2)}

    # ------------------------------------------------------------------
    def plot_loss_curve(self, history, model_name: str) -> str:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history["loss"], label="train_loss")
        ax.plot(history.history["val_loss"], label="val_loss")
        ax.set_title(f"{model_name} - Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        path = os.path.join(self.plots_dir, f"{model_name}_loss_curve.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_actual_vs_prediction(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_true, label="Actual", linewidth=1.5)
        ax.plot(y_pred, label="Predicted", linewidth=1.5, linestyle="--")
        ax.set_title(f"{model_name} - Actual vs Predicted")
        ax.set_xlabel("Timestep")
        ax.set_ylabel(self.cfg.features.target_column)
        ax.legend()
        path = os.path.join(self.plots_dir, f"{model_name}_actual_vs_pred.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> str:
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(range(len(residuals)), residuals, s=10, alpha=0.6)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_title(f"{model_name} - Residual Plot")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Residual (Actual - Predicted)")
        path = os.path.join(self.plots_dir, f"{model_name}_residuals.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> str:
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(residuals, bins=30, alpha=0.75, color="steelblue", edgecolor="black")
        ax.set_title(f"{model_name} - Error Distribution")
        ax.set_xlabel("Error")
        ax.set_ylabel("Frequency")
        path = os.path.join(self.plots_dir, f"{model_name}_error_distribution.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def save_predictions_csv(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> str:
        df = pd.DataFrame({"actual": y_true, "predicted": y_pred, "error": y_true - y_pred})
        path = os.path.join(self.results_dir, f"{model_name}_predictions.csv")
        df.to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------
    def evaluate_model(self, model_name: str, history, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        metrics = self.compute_metrics(y_true, y_pred)
        self.logger.info("[%s] metrics: %s", model_name, metrics)

        plots = {
            "loss_curve": self.plot_loss_curve(history, model_name) if history is not None else None,
            "actual_vs_prediction": self.plot_actual_vs_prediction(y_true, y_pred, model_name),
            "residuals": self.plot_residuals(y_true, y_pred, model_name),
            "error_distribution": self.plot_error_distribution(y_true, y_pred, model_name),
        }
        predictions_csv = self.save_predictions_csv(y_true, y_pred, model_name)

        metrics_path = os.path.join(self.eval_dir, f"{model_name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return {"model_name": model_name, "metrics": metrics, "plots": plots, "predictions_csv": predictions_csv}

    # ------------------------------------------------------------------
    def compare_models(self, all_results: List[Dict]) -> Dict:
        """
        Builds comparison.csv, comparison.png, metrics.json and automatically
        determines the best model based on config.inference.best_model_metric.
        """
        rows = []
        for r in all_results:
            row = {"model": r["model_name"], **r["metrics"]}
            rows.append(row)
        comparison_df = pd.DataFrame(rows)

        # lower-is-better for mae/mse/rmse/mape, higher-is-better for r2
        lower_is_better = self.metric_for_best in ("mae", "mse", "rmse", "mape")
        if lower_is_better:
            best_idx = comparison_df[self.metric_for_best].idxmin()
        else:
            best_idx = comparison_df[self.metric_for_best].idxmax()
        best_model = comparison_df.loc[best_idx, "model"]

        comparison_csv_path = os.path.join(self.results_dir, "comparison.csv")
        comparison_df.to_csv(comparison_csv_path, index=False)

        fig, ax = plt.subplots(figsize=(9, 5))
        metrics_to_plot = ["mae", "rmse", "mape"]
        x = np.arange(len(comparison_df))
        width = 0.25
        for i, m in enumerate(metrics_to_plot):
            ax.bar(x + i * width, comparison_df[m], width=width, label=m.upper())
        ax.set_xticks(x + width)
        ax.set_xticklabels(comparison_df["model"])
        ax.set_title("Model Comparison")
        ax.legend()
        comparison_png_path = os.path.join(self.results_dir, "comparison.png")
        fig.savefig(comparison_png_path, bbox_inches="tight")
        plt.close(fig)

        summary = {
            "best_model": best_model,
            "best_model_metric": self.metric_for_best,
            "all_metrics": {r["model_name"]: r["metrics"] for r in all_results},
        }
        metrics_json_path = os.path.join(self.results_dir, "metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info("Model comparison complete. Best model: %s (by %s)", best_model, self.metric_for_best)
        return {
            "comparison_csv": comparison_csv_path,
            "comparison_png": comparison_png_path,
            "metrics_json": metrics_json_path,
            "best_model": best_model,
        }
