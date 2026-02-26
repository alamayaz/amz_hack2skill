"""Feature persistence (JSON-based)."""

import logging
import shutil
from pathlib import Path

from pydantic import BaseModel

from rezaa.config import get_settings

logger = logging.getLogger(__name__)


class FeatureStore:
    """Stores and retrieves extracted features as JSON."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or get_settings().feature_store_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _job_dir(self, job_id: str) -> Path:
        d = self.base_dir / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_features(self, job_id: str, name: str, model: BaseModel) -> Path:
        """Save a Pydantic model as JSON."""
        path = self._job_dir(job_id) / f"{name}.json"
        path.write_text(model.model_dump_json(indent=2))
        return path

    def load_features(self, job_id: str, name: str, model_class: type[BaseModel]) -> BaseModel:
        """Load a Pydantic model from JSON."""
        path = self._job_dir(job_id) / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        return model_class.model_validate_json(path.read_text())

    def save_edl(self, job_id: str, edl: BaseModel) -> Path:
        """Save an EDL for potential retry."""
        return self.save_features(job_id, "edl", edl)

    def load_edl(self, job_id: str, model_class: type[BaseModel]) -> BaseModel:
        """Load a saved EDL."""
        return self.load_features(job_id, "edl", model_class)

    def delete_job_data(self, job_id: str) -> None:
        """Delete all stored data for a job (privacy compliance)."""
        job_dir = self.base_dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
            logger.info(f"Deleted all data for job {job_id}")

    def list_jobs(self) -> list[str]:
        """List all job IDs with stored features."""
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
