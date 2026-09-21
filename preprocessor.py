"""
Step 5: Preprocessing.

Handles missing values, duplicate removal, outlier detection, scaling,
sliding-window sequence generation, and a strictly sequential
(never shuffled) train/val/test split, as required for time series data.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.utils.logger import get_logger


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


class Preprocessor:
    """
    End-to-end preprocessing pipeline for the traffic time series.

    Pipeline order (all configurable via config.yaml):
        1. drop duplicate rows
        2. handle missing values
        3. detect & clip/remove outliers
        4. scale features
        5. build sliding windows (X: window of past steps, y: horizon ahead)
        6. sequential split into train/val/test (no shuffling, ever)
    """

    def __init__(self, config):
        self.cfg = config
        self.logger = get_logger("Preprocessor", config.paths.logs_dir)
        p = config.preprocessing
        self.missing_strategy = p.missing_value_strategy
        self.outlier_method = p.outlier_method
        self.zscore_thresh = p.outlier_zscore_threshold
        self.iqr_mult = p.outlier_iqr_multiplier
        self.scaler_type = p.scaler_type
        self.window_size = p.sliding_window_size
        self.horizon = p.forecast_horizon
        self.train_split = p.train_split
        self.val_split = p.val_split
        self.test_split = p.test_split
        assert config.preprocessing.shuffle is False, "Time series must never be shuffled."

        self.scaler = None
        self.feature_columns: List[str] = []

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=["Time"]).reset_index(drop=True)
        removed = before - len(df)
        if removed:
            self.logger.info("Removed %d duplicate rows.", removed)
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_missing = df[numeric_cols].isna().sum().sum()
        if n_missing == 0:
            return df

        self.logger.info("Handling %d missing values using strategy '%s'.", n_missing, self.missing_strategy)
        if self.missing_strategy == "interpolate":
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
        elif self.missing_strategy == "ffill":
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
        elif self.missing_strategy == "bfill":
            df[numeric_cols] = df[numeric_cols].bfill().ffill()
        elif self.missing_strategy == "drop":
            df = df.dropna(subset=numeric_cols).reset_index(drop=True)
        else:
            raise ValueError(f"Unknown missing_value_strategy: {self.missing_strategy}")
        return df

    def handle_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        if self.outlier_method == "none":
            return df

        df = df.copy()
        for col in columns:
            series = df[col]
            if self.outlier_method == "zscore":
                mean, std = series.mean(), series.std()
                if std == 0 or np.isnan(std):
                    continue
                z = (series - mean) / std
                mask = z.abs() > self.zscore_thresh
                clip_low, clip_high = mean - self.zscore_thresh * std, mean + self.zscore_thresh * std
            elif self.outlier_method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                clip_low, clip_high = q1 - self.iqr_mult * iqr, q3 + self.iqr_mult * iqr
                mask = (series < clip_low) | (series > clip_high)
            else:
                raise ValueError(f"Unknown outlier_method: {self.outlier_method}")

            n_outliers = int(mask.sum())
            if n_outliers:
                self.logger.info("Clipping %d outliers in column '%s'.", n_outliers, col)
                df[col] = series.clip(lower=clip_low, upper=clip_high)
        return df

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------
    def fit_scaler(self, df: pd.DataFrame, columns: List[str]) -> None:
        self.feature_columns = columns
        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {self.scaler_type}")

        self.scaler.fit(df[columns].values)
        self.logger.info("Fitted %s on %d feature columns.", self.scaler_type, len(columns))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fit yet. Call fit_scaler first.")
        return self.scaler.transform(df[self.feature_columns].values)

    def inverse_transform_target(self, values: np.ndarray, target_column: str) -> np.ndarray:
        """Inverse-scales a 1D array of predictions/targets for a single column."""
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fit yet.")
        col_idx = self.feature_columns.index(target_column)
        dummy = np.zeros((len(values), len(self.feature_columns)))
        dummy[:, col_idx] = values.reshape(-1)
        inv = self.scaler.inverse_transform(dummy)
        return inv[:, col_idx]

    def save_scaler(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "feature_columns": self.feature_columns}, f)
        self.logger.info("Saved fitted scaler to %s", path)

    def load_scaler(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.feature_columns = data["feature_columns"]

    # ------------------------------------------------------------------
    # Sliding window + sequential split
    # ------------------------------------------------------------------
    def create_sliding_windows(self, scaled_array: np.ndarray, target_col_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds supervised sequences:
            X[i] = scaled_array[i : i + window_size]                 (all features)
            y[i] = scaled_array[i + window_size + horizon - 1, target] (single target)
        """
        X, y = [], []
        n = len(scaled_array)
        last_start = n - self.window_size - self.horizon + 1
        for i in range(max(0, last_start)):
            X.append(scaled_array[i : i + self.window_size])
            y.append(scaled_array[i + self.window_size + self.horizon - 1, target_col_idx])
        X = np.array(X)
        y = np.array(y)
        self.logger.info("Created %d sliding windows (window_size=%d, horizon=%d).", len(X), self.window_size, self.horizon)
        return X, y

    def sequential_split(self, X: np.ndarray, y: np.ndarray) -> SplitData:
        """Splits X/y sequentially (no shuffling) into train/val/test."""
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            "train/val/test splits must sum to 1.0"

        n = len(X)
        train_end = int(n * self.train_split)
        val_end = train_end + int(n * self.val_split)

        split = SplitData(
            X_train=X[:train_end], y_train=y[:train_end],
            X_val=X[train_end:val_end], y_val=y[train_end:val_end],
            X_test=X[val_end:], y_test=y[val_end:],
        )
        self.logger.info(
            "Sequential split -> train=%d, val=%d, test=%d",
            len(split.X_train), len(split.X_val), len(split.X_test),
        )
        return split

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame, feature_columns: List[str], target_column: str) -> Tuple[SplitData, pd.DataFrame]:
        df = df.sort_values("Time").reset_index(drop=True)
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.handle_outliers(df, feature_columns)

        self.fit_scaler(df, feature_columns)
        scaled = self.transform(df)

        target_idx = feature_columns.index(target_column)
        X, y = self.create_sliding_windows(scaled, target_idx)
        split = self.sequential_split(X, y)
        return split, df
