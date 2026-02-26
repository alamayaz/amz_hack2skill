"""Temporary file lifecycle management."""

import logging
import shutil
import time
from pathlib import Path

from rezaa.config import get_settings

logger = logging.getLogger(__name__)


class TempFileManager:
    """Manages per-job temporary directories with automatic cleanup."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or get_settings().temp_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._job_dirs: dict[str, tuple[Path, float]] = {}

    def create_job_dir(self, job_id: str) -> Path:
        """Create a temporary directory for a job."""
        job_dir = self.base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        self._job_dirs[job_id] = (job_dir, time.time())
        return job_dir

    def get_job_dir(self, job_id: str) -> Path | None:
        """Get the temporary directory for a job."""
        if job_id in self._job_dirs:
            return self._job_dirs[job_id][0]
        job_dir = self.base_dir / job_id
        if job_dir.exists():
            return job_dir
        return None

    def cleanup_job(self, job_id: str) -> None:
        """Clean up temporary files for a job."""
        if job_id in self._job_dirs:
            job_dir = self._job_dirs[job_id][0]
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
            del self._job_dirs[job_id]
            logger.info(f"Cleaned up temp files for job {job_id}")
        else:
            job_dir = self.base_dir / job_id
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)

    def cleanup_expired(self, ttl_seconds: int | None = None) -> int:
        """Clean up all expired job directories."""
        ttl = ttl_seconds or get_settings().temp_file_ttl_seconds
        now = time.time()
        cleaned = 0
        expired_jobs = [jid for jid, (_, created) in self._job_dirs.items() if now - created > ttl]
        for job_id in expired_jobs:
            self.cleanup_job(job_id)
            cleaned += 1
        return cleaned
