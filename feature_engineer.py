"""
Step 6: Feature Engineering.

Derives additional traffic-domain features on top of the raw time series:
Vehicle Density, Heavy Vehicle Ratio, Bike Ratio, Pedestrian Density,
Traffic Flow Rate, Moving Average, Rolling Mean/Std, Peak Hour Indicator.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

HEAVY_VEHICLES = ["Bus", "Truck", "Van"]
BIKE_CLASSES = ["Motorcycle", "Bicycle"]
PERSON_CLASSES = ["Person", "Cyclist", "Motorcyclist", "ScooterRider"]


class FeatureEngineer:
    def __init__(self, config):
        self.cfg = config
        self.logger = get_logger("FeatureEngineer", config.paths.logs_dir)
        self.rolling_windows = config.features.rolling_window_sizes
        self.peak_hours = config.features.peak_hours  # list of [start, end)
        self.target_column = config.features.target_column

    def add_ratio_and_density_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        eps = 1e-6
        total = df["TotalVehicles"].clip(lower=0)

        df["VehicleDensity"] = total / (df.index.to_series().diff().fillna(1).clip(lower=1))
        heavy_cols = [c for c in HEAVY_VEHICLES if c in df.columns]
        bike_cols = [c for c in BIKE_CLASSES if c in df.columns]
        person_cols = [c for c in PERSON_CLASSES if c in df.columns]

        df["HeavyVehicleRatio"] = df[heavy_cols].sum(axis=1) / (total + eps) if heavy_cols else 0.0
        df["BikeRatio"] = df[bike_cols].sum(axis=1) / (total + eps) if bike_cols else 0.0
        df["PedestrianDensity"] = df[person_cols].sum(axis=1) if person_cols else 0.0

        return df

    def add_flow_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Traffic flow rate: change in total vehicles per unit time interval."""
        df = df.copy()
        interval_sec = self.cfg.timeseries.aggregation_interval_seconds
        df["TrafficFlowRate"] = df["TotalVehicles"].diff().fillna(0) / (interval_sec / 60.0)  # vehicles/min change
        return df

    def add_rolling_features(self, df: pd.DataFrame, column: str = "TotalVehicles") -> pd.DataFrame:
        df = df.copy()
        for w in self.rolling_windows:
            df[f"MovingAvg_{w}"] = df[column].rolling(window=w, min_periods=1).mean()
            df[f"RollingMean_{w}"] = df[column].rolling(window=w, min_periods=1).mean()
            df[f"RollingStd_{w}"] = df[column].rolling(window=w, min_periods=1).std().fillna(0)
        return df

    def add_peak_hour_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        time_col = pd.to_datetime(df["Time"])
        hours = time_col.dt.hour

        def is_peak(hour: int) -> int:
            for start, end in self.peak_hours:
                if start <= hour < end:
                    return 1
            return 0

        df["PeakHourIndicator"] = hours.apply(is_peak)
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_ratio_and_density_features(df)
        df = self.add_flow_rate(df)
        df = self.add_rolling_features(df, column=self.target_column)
        df = self.add_peak_hour_indicator(df)
        self.logger.info("Feature engineering complete. Total columns: %d", len(df.columns))
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """All numeric columns except 'Time' are candidate model features."""
        return [c for c in df.columns if c != "Time" and pd.api.types.is_numeric_dtype(df[c])]
