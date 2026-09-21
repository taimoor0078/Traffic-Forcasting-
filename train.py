"""
Traffic Flow Forecasting from Video using Deep Learning and Computer Vision
=============================================================================
Main entry point. Runs the COMPLETE pipeline end-to-end, automatically,
from raw traffic videos through to trained & compared forecasting models.

Usage:
    python train.py
    python train.py --config config/config.yaml
    python train.py --video path/to/single_video.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import load_config
from src.detection.detection_writer import DetectionWriter
from src.detection.detector import YOLODetector
from src.evaluation.evaluator import Evaluator
from src.features.feature_engineer import FeatureEngineer
from src.models.model_factory import build_and_compile_model
from src.preprocessing.preprocessor import Preprocessor
from src.timeseries.generator import TimeSeriesGenerator
from src.tracking.roi_counter import ROICounter
from src.training.model_saver import ModelSaver
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed
from src.video.video_loader import VideoLoader
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Flow Forecasting - Full Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--video", type=str, default=None, help="Optional: run on a single video only")
    parser.add_argument(
        "--timeseries-csv", type=str, default=None,
        help="Optional: skip video processing and start from an existing time-series CSV "
             "(must have a 'Time' column). Useful for iterating on modeling only.",
    )
    return parser.parse_args()




def main():
    args = parse_args()
    config = load_config(args.config)
    logger = get_logger("train", config.paths.logs_dir)
    set_global_seed(config.project.seed)

    logger.info("=" * 70)
    logger.info("TRAFFIC FLOW FORECASTING PIPELINE - STARTING FULL RUN")
    logger.info("=" * 70)

    # ---------------- Steps 1-4: Video -> Detection -> Tracking -> Time Series ----------------
    timeseries_path = os.path.join(config.paths.timeseries_dir, "traffic_timeseries.csv")
    logger.info("Loading time series from %s", timeseries_path)
    df = pd.read_csv(timeseries_path)

    if len(df) < config.preprocessing.sliding_window_size + config.preprocessing.forecast_horizon + 1:
        logger.error(
            "Not enough time-series rows (%d) to build even one sliding window "
            "(need >= %d). Provide more/longer video, a smaller aggregation "
            "interval, or a smaller sliding_window_size in config.yaml.",
            len(df), config.preprocessing.sliding_window_size + config.preprocessing.forecast_horizon + 1,
        )
        sys.exit(1)

    # ---------------- Step 6: Feature Engineering ----------------
    feature_engineer = FeatureEngineer(config)
    df = feature_engineer.run(df)
    feature_columns = feature_engineer.get_feature_columns(df)
    target_column = config.features.target_column

    # ---------------- Step 5: Preprocessing ----------------
    preprocessor = Preprocessor(config)
    split, clean_df = preprocessor.run(df, feature_columns, target_column)

    input_shape = (config.preprocessing.sliding_window_size, len(feature_columns))
    logger.info("Model input shape: %s", input_shape)

    # ---------------- Steps 7-9: Train + Evaluate LSTM, GRU, Transformer ----------------
    trainer = Trainer(config)
    evaluator = Evaluator(config)
    model_saver = ModelSaver(config)

    all_results = []
    checkpoint_paths = {}

    for model_name in ["lstm", "gru", "transformer"]:
        logger.info("-" * 70)
        logger.info("Training model: %s", model_name.upper())
        model = build_and_compile_model(model_name, input_shape, config)
        model.summary(print_fn=logger.info)

        history, checkpoint_path = trainer.train(
            model, model_name,
            split.X_train, split.y_train,
            split.X_val, split.y_val,
        )
        checkpoint_paths[model_name] = checkpoint_path

        # reload best checkpoint (restore_best_weights already applied, but this
        # guarantees we evaluate exactly what was persisted to disk)
        import tensorflow as tf
        best_model = tf.keras.models.load_model(checkpoint_path)

        y_pred_scaled = best_model.predict(split.X_test, verbose=0).reshape(-1)
        y_true_scaled = split.y_test.reshape(-1)

        y_pred = preprocessor.inverse_transform_target(y_pred_scaled, target_column)
        y_true = preprocessor.inverse_transform_target(y_true_scaled, target_column)

        result = evaluator.evaluate_model(model_name, history, y_true, y_pred)
        all_results.append(result)

    # ---------------- Step 10: Model Comparison ----------------
    comparison = evaluator.compare_models(all_results)
    best_model_name = comparison["best_model"]
    logger.info("BEST MODEL: %s", best_model_name.upper())

    # ---------------- Step 11: Save Models ----------------
    for model_name, ckpt_path in checkpoint_paths.items():
        model_saver.save_final_model(ckpt_path, model_name)

    model_saver.save_scaler(preprocessor)
    model_saver.save_metadata(
        feature_columns=feature_columns,
        target_column=target_column,
        best_model=best_model_name,
        comparison_summary={r["model_name"]: r["metrics"] for r in all_results},
    )
    model_saver.save_config_snapshot()

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE. Best model '%s' saved to %s", best_model_name, config.paths.saved_models_dir)
    logger.info(
        "Run inference with: python infer.py --video path/to/video.mp4"
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
