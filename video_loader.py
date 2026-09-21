"""
Step 1: Video Loading & Frame Extraction.

Loads videos from a configurable dataset directory (compatible with the
raw video layout of AI City Challenge, BDD100K, UA-DETRAC, or any custom
folder of video files) and extracts frames at a configurable FPS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Generator, List, Optional

import cv2
import numpy as np

from src.utils.logger import get_logger


@dataclass
class Frame:
    """A single extracted frame with its provenance metadata."""

    index: int
    timestamp_sec: float
    image: np.ndarray
    source_video: str


class VideoLoader:
    """
    Discovers video files under `videos_dir` and yields frames sampled at
    `extract_fps`, resized to a configured resolution.
    """

    def __init__(self, config):
        self.cfg = config
        self.logger = get_logger("VideoLoader", config.paths.logs_dir)
        self.videos_dir = config.paths.videos_dir
        self.extensions = tuple(config.dataset.video_extensions)
        self.extract_fps = config.video.extract_fps
        self.resize_width = config.video.resize_width
        self.resize_height = config.video.resize_height
        self.save_to_disk = config.video.save_frames_to_disk
        self.frames_dir = config.paths.frames_dir

    def discover_videos(self) -> List[str]:
        """Return sorted list of absolute video file paths found in videos_dir."""
        if not os.path.isdir(self.videos_dir):
            self.logger.warning("Videos directory does not exist: %s", self.videos_dir)
            return []

        videos = []
        for root, _, files in os.walk(self.videos_dir):
            for fname in files:
                if fname.lower().endswith(self.extensions):
                    videos.append(os.path.join(root, fname))
        videos.sort()
        self.logger.info("Discovered %d video(s) in %s", len(videos), self.videos_dir)
        return videos

    def extract_frames(self, video_path: str) -> Generator[Frame, None, None]:
        """
        Generator that yields Frame objects sampled at self.extract_fps
        from a single video file.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_every = max(1, round(native_fps / self.extract_fps))

        self.logger.info(
            "Extracting frames from %s (native_fps=%.2f, sample_every=%d frames, target_fps=%s)",
            video_path, native_fps, sample_every, self.extract_fps,
        )

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        if self.save_to_disk:
            out_dir = os.path.join(self.frames_dir, video_name)
            os.makedirs(out_dir, exist_ok=True)

        raw_idx = 0
        yielded_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if raw_idx % sample_every == 0:
                resized = cv2.resize(frame, (self.resize_width, self.resize_height))
                timestamp_sec = raw_idx / native_fps

                if self.save_to_disk:
                    fname = os.path.join(out_dir, f"frame_{yielded_idx:06d}.jpg")
                    cv2.imwrite(fname, resized)

                yield Frame(
                    index=yielded_idx,
                    timestamp_sec=timestamp_sec,
                    image=resized,
                    source_video=video_name,
                )
                yielded_idx += 1

            raw_idx += 1

        cap.release()
        self.logger.info("Finished extracting %d frames from %s", yielded_idx, video_name)

    def extract_all(self, video_path: Optional[str] = None) -> Generator[Frame, None, None]:
        """
        Convenience generator over either a single video (if provided) or
        every video discovered in videos_dir, concatenated in order.
        """
        videos = [video_path] if video_path else self.discover_videos()
        if not videos:
            self.logger.warning("No videos to process.")
            return
        for vp in videos:
            yield from self.extract_frames(vp)
