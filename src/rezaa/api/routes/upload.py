"""Upload endpoints."""

import shutil

from fastapi import APIRouter, Depends, UploadFile

from rezaa.api.dependencies import get_pipeline_manager, get_temp_store
from rezaa.extractors.validators import validate_audio_upload, validate_video_upload
from rezaa.models.errors import ValidationError
from rezaa.pipeline.manager import PipelineManager
from rezaa.storage.temp_store import TempFileManager

router = APIRouter(prefix="/api/v1", tags=["upload"])


@router.post("/upload/audio")
async def upload_audio(
    file: UploadFile,
    manager: PipelineManager = Depends(get_pipeline_manager),
    temp: TempFileManager = Depends(get_temp_store),
):
    """Upload an audio file for processing."""
    if not file.filename:
        raise ValidationError("No filename provided")

    # Create job and save file
    state = manager.create_job()
    job_dir = temp.create_job_dir(state.job_id)
    file_path = job_dir / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Validate
    validate_audio_upload(file_path)

    state.audio_path = str(file_path)
    return {
        "job_id": state.job_id,
        "filename": file.filename,
        "path": str(file_path),
        "message": "Audio uploaded successfully",
    }


@router.post("/upload/video")
async def upload_video(
    file: UploadFile,
    job_id: str | None = None,
    manager: PipelineManager = Depends(get_pipeline_manager),
    temp: TempFileManager = Depends(get_temp_store),
):
    """Upload a video clip for processing."""
    if not file.filename:
        raise ValidationError("No filename provided")

    # Use existing job or create new one
    if job_id:
        state = manager.get_job_state(job_id)
        if not state:
            raise ValidationError(f"Job {job_id} not found")
    else:
        state = manager.create_job()

    job_dir = temp.get_job_dir(state.job_id) or temp.create_job_dir(state.job_id)
    file_path = job_dir / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Validate
    validate_video_upload(file_path)

    state.video_paths.append(str(file_path))
    return {
        "job_id": state.job_id,
        "filename": file.filename,
        "path": str(file_path),
        "clip_count": len(state.video_paths),
        "message": "Video uploaded successfully",
    }
