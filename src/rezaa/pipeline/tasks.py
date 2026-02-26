"""Celery task definitions."""

from pathlib import Path

from celery import Celery

from rezaa.config import get_settings
from rezaa.models.preferences import UserPreferences

settings = get_settings()

celery_app = Celery(
    "rezaa",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)


@celery_app.task(bind=True, name="rezaa.process_video")
def process_video_task(
    self,
    job_id: str,
    audio_path: str,
    video_paths: list[str],
    preferences: dict | None = None,
):
    """Celery task wrapping PipelineManager.process()."""
    from rezaa.pipeline.manager import PipelineManager

    manager = PipelineManager()

    # Ensure job exists
    if not manager.get_job_state(job_id):
        state = manager.create_job()
        job_id = state.job_id

    prefs = UserPreferences(**(preferences or {}))

    try:
        result = manager.process(
            job_id=job_id,
            audio_path=Path(audio_path),
            video_paths=[Path(p) for p in video_paths],
            preferences=prefs,
        )
        return {
            "job_id": result.job_id,
            "status": result.stage.value,
            "output_path": result.output_path,
        }
    except Exception as e:
        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
        }
