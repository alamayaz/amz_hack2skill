"""Status endpoint."""

from fastapi import APIRouter, Depends

from rezaa.api.dependencies import get_pipeline_manager
from rezaa.models.errors import ValidationError
from rezaa.pipeline.manager import PipelineManager

router = APIRouter(prefix="/api/v1", tags=["status"])


@router.get("/status/{job_id}")
async def get_status(
    job_id: str,
    manager: PipelineManager = Depends(get_pipeline_manager),
):
    """Get the processing status of a job."""
    state = manager.get_job_state(job_id)
    if not state:
        raise ValidationError(f"Job {job_id} not found")

    return {
        "job_id": state.job_id,
        "stage": state.stage.value,
        "progress": state.progress,
        "message": state.message,
        "started_at": state.started_at.isoformat() if state.started_at else None,
        "updated_at": state.updated_at.isoformat() if state.updated_at else None,
        "completed_at": state.completed_at.isoformat() if state.completed_at else None,
        "error": state.error,
        "output_path": state.output_path,
    }
