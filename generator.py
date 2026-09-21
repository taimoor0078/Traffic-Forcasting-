"""
Step 4: Time Series Generation.

Aggregates per-frame detections (deduplicated by track_id within each
bucket, so a vehicle present for multiple frames in the same interval is
counted once) into fixed-interval rows:

    Time, Cars, Buses, Trucks, Motorcycles, Bicycles, Persons,
    EmergencyVehicles, TotalVehicles

The resulting dataframe is saved to CSV automatically.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

from src.detection.detector import Detection
from src.utils.logger import get_logger

VEHICLE_CLASSES = ["Car", "Bus", "Truck", "Van", "Pickup", "Taxi", "Motorcycle", "Bicycle"]
EMERGENCY_CLASSES = ["Ambulance", "PoliceVehicle", "FireTruck"]
PERSON_CLASSES = ["Person", "Cyclist", "Motorcyclist", "ScooterRider"]


class TimeSeriesGenerator:
    """
    Builds the traffic time series dataframe from a stream of per-frame
    Detection objects.
    """

    def __init__(self, config, start_time: datetime | None = None):
        self.cfg = config
        self.logger = get_logger("TimeSeriesGenerator", config.paths.logs_dir)
        self.interval_sec = config.timeseries.aggregation_interval_seconds
        self.out_dir = config.paths.timeseries_dir
        self.out_name = config.timeseries.output_csv_name
        self.start_time = start_time or datetime.now()

    def _bucket_index(self, timestamp_sec: float) -> int:
        return int(timestamp_sec // self.interval_sec)

    def build(self, detections: List[Detection]) -> pd.DataFrame:
        """
        Groups detections into time buckets and counts unique track_ids
        per class within each bucket (falls back to raw detection counts
        if track_id is unavailable).
        """
        if not detections:
            self.logger.warning("No detections supplied to TimeSeriesGenerator.")
            return pd.DataFrame(
                columns=["Time"] + VEHICLE_CLASSES + PERSON_CLASSES + ["EmergencyVehicles", "TotalVehicles"]
            )

        buckets: Dict[int, Dict[str, set]] = {}

        for det in detections:
            b_idx = self._bucket_index(det.timestamp_sec)
            buckets.setdefault(b_idx, {})
            class_set = buckets[b_idx].setdefault(det.class_name, set())
            # dedupe by track_id when available, else by a synthetic unique key
            identity = det.track_id if det.track_id is not None else id(det)
            class_set.add(identity)

        rows = []
        for b_idx in sorted(buckets.keys()):
            bucket_time = self.start_time + timedelta(seconds=b_idx * self.interval_sec)
            class_counts = buckets[b_idx]

            row = {"Time": bucket_time}
            for cls in VEHICLE_CLASSES:
                row[cls] = len(class_counts.get(cls, set()))
            for cls in PERSON_CLASSES:
                row[cls] = len(class_counts.get(cls, set()))
            row["EmergencyVehicles"] = sum(len(class_counts.get(cls, set())) for cls in EMERGENCY_CLASSES)
            row["TotalVehicles"] = sum(row[cls] for cls in VEHICLE_CLASSES) + row["EmergencyVehicles"]
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("Time").reset_index(drop=True)
        self.logger.info("Generated time series with %d rows.", len(df))
        return df

    def save(self, df: pd.DataFrame, filename: str | None = None) -> str:
        out_path = os.path.join(self.out_dir, filename or self.out_name)
        df.to_csv(out_path, index=False)
        self.logger.info("Saved time series CSV to %s", out_path)
        return out_path
