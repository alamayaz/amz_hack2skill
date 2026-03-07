"""Benchmark / performance report endpoint."""

from fastapi import APIRouter, Depends

from rezaa.api.dependencies import get_pipeline_manager
from rezaa.models.errors import ValidationError
from rezaa.pipeline.manager import PipelineManager

router = APIRouter(prefix="/api/v1", tags=["benchmark"])


@router.get("/benchmark/{job_id}")
async def get_benchmark(
    job_id: str,
    manager: PipelineManager = Depends(get_pipeline_manager),
):
    """Return the benchmark report for a completed (or partially completed) job."""
    state = manager.get_job_state(job_id)
    if not state:
        raise ValidationError(f"Job {job_id} not found")
    if state.benchmark_report is None:
        raise ValidationError(f"Benchmark report not yet available for job {job_id}")
    return state.benchmark_report
