"""Persists Detection records to disk as CSV, per source video."""

from __future__ import annotations

import os
from typing import List

import pandas as pd

from src.detection.detector import Detection
from src.utils.logger import get_logger


class DetectionWriter:
    def __init__(self, config):
        self.cfg = config
        self.logger = get_logger("DetectionWriter", config.paths.logs_dir)
        self.out_dir = config.paths.detections_dir

    def to_dataframe(self, detections: List[Detection]) -> pd.DataFrame:
        rows = [
            {
                "frame_index": d.frame_index,
                "timestamp_sec": d.timestamp_sec,
                "class_name": d.class_name,
                "source_class": d.source_class,
                "confidence": d.confidence,
                "x1": d.bbox[0],
                "y1": d.bbox[1],
                "x2": d.bbox[2],
                "y2": d.bbox[3],
                "track_id": d.track_id,
            }
            for d in detections
        ]
        return pd.DataFrame(rows)

    def save(self, detections: List[Detection], video_name: str) -> str:
        df = self.to_dataframe(detections)
        out_path = os.path.join(self.out_dir, f"{video_name}_detections.csv")
        df.to_csv(out_path, index=False)
        self.logger.info("Saved %d detections to %s", len(df), out_path)
        return out_path
