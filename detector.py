"""
Step 2: Object Detection using pretrained YOLOv8 (Ultralytics).

This module ONLY performs inference with pretrained weights. It never
fine-tunes or trains YOLO. Detections are mapped from the underlying
COCO taxonomy onto the project's extended road-object taxonomy via the
`class_map` defined in config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger


@dataclass
class Detection:
    """A single detected object in a single frame."""

    frame_index: int
    timestamp_sec: float
    class_name: str          # project taxonomy name, e.g. "Car"
    source_class: str        # underlying model class name, e.g. "car"
    confidence: float
    bbox: List[float]        # [x1, y1, x2, y2] in pixel coords
    track_id: Optional[int] = None


class YOLODetector:
    """
    Thin wrapper around ultralytics.YOLO that:
      1. Loads pretrained weights only (no training).
      2. Runs detection (optionally with built-in tracking) per frame.
      3. Maps COCO class names to the project's extended taxonomy.
    """

    def __init__(self, config):
        self.cfg = config
        self.logger = get_logger("YOLODetector", config.paths.logs_dir)

        self.weights = config.detection.model_weights
        self.conf_thresh = config.detection.confidence_threshold
        self.iou_thresh = config.detection.iou_threshold
        self.device = config.detection.device
        self.class_map: Dict[str, List[str]] = config.detection.class_map.to_dict()

        # inverse map: source coco class -> list of project classes it can map to
        self._source_to_targets: Dict[str, List[str]] = {}
        for target, sources in self.class_map.items():
            for src in sources:
                self._source_to_targets.setdefault(src, []).append(target)

        self._model = None  # lazy-loaded

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO

            self.logger.info("Loading pretrained YOLOv8 weights: %s", self.weights)
            self._model = YOLO(self.weights)  # pretrained weights only, no .train() call
        return self._model

    def _map_class(self, coco_name: str) -> Optional[str]:
        """Map a raw COCO class name to our project taxonomy (first match)."""
        targets = self._source_to_targets.get(coco_name)
        if not targets:
            return None
        return targets[0]

    def detect_frame(self, frame_index: int, timestamp_sec: float, image: np.ndarray) -> List[Detection]:
        """Run detection (no tracking) on a single frame."""
        results = self.model.predict(
            source=image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=None if self.device == "auto" else self.device,
            verbose=False,
        )
        return self._parse_results(results, frame_index, timestamp_sec)

    def detect_and_track_frame(
        self, frame_index: int, timestamp_sec: float, image: np.ndarray, tracker_yaml: str, persist: bool = True
    ) -> List[Detection]:
        """Run detection + built-in ByteTrack/BoT-SORT tracking on a single frame."""
        results = self.model.track(
            source=image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=None if self.device == "auto" else self.device,
            tracker=tracker_yaml,
            persist=persist,
            verbose=False,
        )
        return self._parse_results(results, frame_index, timestamp_sec, with_tracking=True)

    def _parse_results(self, results, frame_index: int, timestamp_sec: float, with_tracking: bool = False) -> List[Detection]:
        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = None
        if with_tracking and boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            coco_name = names[cls_ids[i]]
            mapped = self._map_class(coco_name)
            if mapped is None:
                continue  # not a class of interest (e.g. "chair")

            detections.append(
                Detection(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    class_name=mapped,
                    source_class=coco_name,
                    confidence=float(confs[i]),
                    bbox=[float(x) for x in xyxy[i]],
                    track_id=int(track_ids[i]) if track_ids is not None else None,
                )
            )
        return detections
