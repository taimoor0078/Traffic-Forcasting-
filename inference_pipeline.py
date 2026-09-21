"""
Step 12: Inference.

Given a raw traffic video, automatically:
    1. Extract frames
    2. Run YOLO detection + tracking
    3. Generate the time series
    4. Apply the saved feature engineering + scaler
    5. Load the best forecasting model
    6. Predict the next N timestamps (recursive multi-step forecasting)
    7. Save predictions to disk
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

from src.detection.detector import YOLODetector
from src.features.feature_engineer import FeatureEngineer
from src.timeseries.generator import TimeSeriesGenerator
from src.tracking.roi_counter import ROICounter
from src.utils.logger import get_logger
from src.video.video_loader import VideoLoader
from src.models.transformer_model import PositionalEncoding


class InferencePipeline:
    def __init__(self, config):
        self.cfg = config
        self.logger = get_logger("InferencePipeline", config.paths.logs_dir)
        self.video_loader = VideoLoader(config)
        self.detector = YOLODetector(config)
        self.ts_generator = TimeSeriesGenerator(config)
        self.feature_engineer = FeatureEngineer(config)
        self.results_dir = config.paths.results_dir
        self.saved_models_dir = config.paths.saved_models_dir

        self._metadata = None
        self._scaler_bundle = None
        self._model = None

    # ------------------------------------------------------------------
    def _load_artifacts(self) -> None:
        meta_path = os.path.join(self.saved_models_dir, "metadata.json")
        with open(meta_path, "r") as f:
            self._metadata = json.load(f)

        scaler_path = os.path.join(self.saved_models_dir, "scaler.pkl")
        with open(scaler_path, "rb") as f:
            self._scaler_bundle = pickle.load(f)

        best_model_name = self._metadata["best_model"]
        model_path = os.path.join(self.saved_models_dir, f"best_{best_model_name}.keras")
        self.logger.info("Loading best model for inference: %s", model_path)
        self._model = tf.keras.models.load_model( model_path, custom_objects={ "PositionalEncoding": PositionalEncoding
    }
)
        self._best_model_name = best_model_name

    # ------------------------------------------------------------------
    def _extract_and_detect(self, video_path: str) -> pd.DataFrame:
        """Runs steps 1-4 of the pipeline on a single input video."""
        roi_counter = ROICounter(self.cfg)
        all_detections = []

        for frame in self.video_loader.extract_frames(video_path):
            dets = self.detector.detect_and_track_frame(
                frame_index=frame.index,
                timestamp_sec=frame.timestamp_sec,
                image=frame.image,
                tracker_yaml=f"{self.cfg.tracking.tracker_type}.yaml",
            )
            roi_counter.process_detections(dets)
            all_detections.extend(dets)

        df = self.ts_generator.build(all_detections)
        df = self.feature_engineer.run(df)
        return df

    # ------------------------------------------------------------------
    def _prepare_model_input(self, df: pd.DataFrame) -> np.ndarray:
        feature_columns = self._metadata["feature_columns"]
        window_size = self._metadata["window_size"]

        missing = [c for c in feature_columns if c not in df.columns]
        for c in missing:
            df[c] = 0.0  # gracefully handle features absent from a short clip

        scaler = self._scaler_bundle["scaler"]
        scaled = scaler.transform(df[feature_columns].values)

        if len(scaled) < window_size:
            pad = np.repeat(scaled[:1], window_size - len(scaled), axis=0)
            scaled = np.vstack([pad, scaled])

        window = scaled[-window_size:]
        return window[np.newaxis, ...], feature_columns

    def _inverse_scale_target(self, values: np.ndarray, feature_columns: List[str]) -> np.ndarray:
        scaler = self._scaler_bundle["scaler"]
        target_column = self._metadata["target_column"]
        col_idx = feature_columns.index(target_column)
        dummy = np.zeros((len(values), len(feature_columns)))
        dummy[:, col_idx] = values.reshape(-1)
        inv = scaler.inverse_transform(dummy)
        return inv[:, col_idx]

    # ------------------------------------------------------------------
    def predict_future(self, window_input: np.ndarray, feature_columns: List[str], steps: int) -> np.ndarray:
        """
        Recursive multi-step forecasting: predicts one step, appends it back
        into the window (holding other engineered features at their last
        observed value), and repeats until `steps` predictions are produced.
        """
        target_column = self._metadata["target_column"]
        target_idx = feature_columns.index(target_column)

        current_window = window_input.copy()
        predictions_scaled = []

        for _ in range(steps):
            pred_scaled = self._model.predict(current_window, verbose=0)[0, 0]
            predictions_scaled.append(pred_scaled)

            next_step = current_window[0, -1, :].copy()
            next_step[target_idx] = pred_scaled

            current_window = np.concatenate(
                [current_window[:, 1:, :], next_step[np.newaxis, np.newaxis, :]], axis=1
            )

        predictions_scaled = np.array(predictions_scaled)
        return self._inverse_scale_target(predictions_scaled, feature_columns)

    # ------------------------------------------------------------------
    def run(self, video_path: str) -> Dict[str, str]:
        """Full inference entry point: video in, forecast CSVs out."""
        if self._model is None:
            self._load_artifacts()

        self.logger.info("Running inference pipeline on: %s", video_path)
        df = self._extract_and_detect(video_path)
        window_input, feature_columns = self._prepare_model_input(df)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_paths = {}

        for steps in self.cfg.inference.forecast_steps:
            forecast = self.predict_future(window_input, feature_columns, steps)
            out_df = pd.DataFrame({
                "step_ahead": np.arange(1, steps + 1),
                f"predicted_{self._metadata['target_column']}": forecast,
            })
            out_path = os.path.join(self.results_dir, f"{video_name}_forecast_{steps}.csv")
            out_df.to_csv(out_path, index=False)
            output_paths[f"forecast_{steps}"] = out_path
            self.logger.info("Saved %d-step forecast to %s", steps, out_path)

        return output_paths
if __name__ == "__main__":
    from config import Config
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    args = parser.parse_args()
    config = Config()
    pipeline = InferencePipeline(config)
    results = pipeline.run(args.video)
    print(results)