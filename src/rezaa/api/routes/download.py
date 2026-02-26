"""Download endpoint."""

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from rezaa.api.dependencies import get_pipeline_manager
from rezaa.models.errors import ValidationError
from rezaa.models.pipeline import PipelineStage
from rezaa.pipeline.manager import PipelineManager

router = APIRouter(prefix="/api/v1", tags=["download"])


@router.get("/download/{job_id}")
async def download_output(
    job_id: str,
    manager: PipelineManager = Depends(get_pipeline_manager),
):
    """Download the rendered output video."""
    state = manager.get_job_state(job_id)
    if not state:
        raise ValidationError(f"Job {job_id} not found")

    if state.stage != PipelineStage.COMPLETE:
        raise ValidationError(f"Job is not complete (current stage: {state.stage.value})")

    if not state.output_path or not Path(state.output_path).exists():
        raise ValidationError("Output file not found")

    return FileResponse(
        path=state.output_path,
        media_type="video/mp4",
        filename=f"rezaa_{job_id}.mp4",
    )
