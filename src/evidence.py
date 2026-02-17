"""
Evidence capture for safety violations: snapshots and CSV logs.
Saves UNSAFE / PARTIAL frames to outputs/violations/ and logs to outputs/logs.csv.
"""

import csv
import re
from datetime import datetime
from pathlib import Path

import cv2


class EvidenceManager:
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.violations_dir = self.output_root / "outputs" / "violations"
        self.logs_path = self.output_root / "outputs" / "logs.csv"
        self.violations_dir.mkdir(parents=True, exist_ok=True)
        if not self.logs_path.exists():
            self.logs_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.logs_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "violation_type", "confidence", "frame"])

    def _sanitize(self, s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_") or "violation"

    def save_snapshot(self, frame, violation_type: str, confidence: float, frame_idx: int = 0) -> Path:
        prefix = self._sanitize(violation_type)
        now = datetime.now()
        name = f"{prefix}_{now:%Y-%m-%d_%H-%M-%S}_f{frame_idx}.jpg"
        path = self.violations_dir / name
        cv2.imwrite(str(path), frame)
        return path

    def log_violation(self, timestamp: str, violation_type: str, confidence: float, frame_idx: int = 0) -> None:
        with open(self.logs_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, violation_type, confidence, frame_idx])
