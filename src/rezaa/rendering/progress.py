"""FFmpeg progress monitoring."""

import re
from collections.abc import Callable


class FFmpegProgressMonitor:
    """Monitor FFmpeg rendering progress from stderr output."""

    def __init__(self, total_duration: float, callback: Callable[[float], None] | None = None):
        self.total_duration = total_duration
        self.callback = callback
        self.current_time = 0.0

    def parse_line(self, line: str) -> float | None:
        """Parse an FFmpeg stderr line for time= progress."""
        match = re.search(r"time=(\d+):(\d+):(\d+\.?\d*)", line)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = float(match.group(3))
            self.current_time = hours * 3600 + minutes * 60 + seconds
            if self.total_duration > 0:
                progress = min(1.0, self.current_time / self.total_duration)
            else:
                progress = 0.0
            if self.callback:
                self.callback(progress)
            return progress
        return None

    @property
    def progress(self) -> float:
        """Current progress as fraction [0, 1]."""
        if self.total_duration <= 0:
            return 0.0
        return min(1.0, self.current_time / self.total_duration)
