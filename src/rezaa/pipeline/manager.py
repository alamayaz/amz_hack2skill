"""Pipeline manager — orchestrates the full processing pipeline."""

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from rezaa.agents.alignment_agent import BeatClipAlignmentAgent
from rezaa.agents.audio_agent import AudioAnalysisAgent
from rezaa.agents.video_agent import VideoUnderstandingAgent
from rezaa.config import get_settings
from rezaa.models.errors import ProcessingError, RezaaError
from rezaa.models.pipeline import PipelineStage, PipelineState
from rezaa.models.preferences import UserPreferences
from rezaa.orchestrator.decision import DecisionOrchestrator
from rezaa.rendering.engine import RenderingEngine
from rezaa.storage.feature_store import FeatureStore
from rezaa.storage.temp_store import TempFileManager

logger = logging.getLogger(__name__)


class PipelineManager:
    """Manages the end-to-end processing pipeline."""

    def __init__(self):
        self.settings = get_settings()
        self.temp_store = TempFileManager()
        self.feature_store = FeatureStore()
        self.audio_agent = AudioAnalysisAgent()
        self.video_agent = VideoUnderstandingAgent()
        self.alignment_agent = BeatClipAlignmentAgent()
        self.orchestrator = DecisionOrchestrator()
        self.renderer = RenderingEngine()
        self._jobs: dict[str, PipelineState] = {}
        self._cancelled: set[str] = set()

    def create_job(self) -> PipelineState:
        """Create a new processing job."""
        job_id = str(uuid.uuid4())
        state = PipelineState(
            job_id=job_id,
            stage=PipelineStage.UPLOAD,
            started_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        self._jobs[job_id] = state
        self.temp_store.create_job_dir(job_id)
        return state

    def get_job_state(self, job_id: str) -> PipelineState | None:
        """Get current state of a job."""
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        if job_id in self._jobs:
            self._cancelled.add(job_id)
            self._update_state(job_id, PipelineStage.CANCELLED, message="Job cancelled")
            return True
        return False

    def process(
        self,
        job_id: str,
        audio_path: Path,
        video_paths: list[Path],
        preferences: UserPreferences | None = None,
    ) -> PipelineState:
        """Run the full processing pipeline.

        Strict ordering: extract → analyze → orchestrate → render
        """
        if job_id not in self._jobs:
            raise ProcessingError(f"Job {job_id} not found")

        state = self._jobs[job_id]
        state.audio_path = str(audio_path)
        state.video_paths = [str(p) for p in video_paths]
        prefs = preferences or UserPreferences()

        try:
            # Stage 1: Audio Extraction + Analysis
            self._check_cancelled(job_id)
            self._update_state(
                job_id,
                PipelineStage.AUDIO_EXTRACTION,
                0.1,
                "Extracting audio features...",
            )
            audio_analysis = self.audio_agent.analyze(audio_path)
            self.feature_store.save_features(job_id, "audio_analysis", audio_analysis)

            # Stage 2: Video Extraction + Analysis
            self._check_cancelled(job_id)
            self._update_state(
                job_id,
                PipelineStage.VIDEO_EXTRACTION,
                0.3,
                "Extracting video features...",
            )
            video_analyses = []
            for i, vpath in enumerate(video_paths):
                self._check_cancelled(job_id)
                clip_id = f"clip_{i:03d}"
                va = self.video_agent.analyze(vpath, clip_id)
                video_analyses.append(va)
                self.feature_store.save_features(job_id, f"video_analysis_{clip_id}", va)

            # Stage 3: Alignment
            self._check_cancelled(job_id)
            self._update_state(job_id, PipelineStage.ALIGNMENT, 0.5, "Aligning clips to beats...")
            alignment = self.alignment_agent.align(
                audio_analysis, video_analyses, prefs.target_duration
            )
            self.feature_store.save_features(job_id, "alignment", alignment)

            # Stage 4: Orchestration
            self._check_cancelled(job_id)
            self._update_state(
                job_id,
                PipelineStage.ORCHESTRATION,
                0.6,
                "Making editing decisions...",
            )
            edl = self.orchestrator.orchestrate(audio_analysis, video_analyses, alignment, prefs)
            self.feature_store.save_edl(job_id, edl)

            # Stage 5: Rendering
            self._check_cancelled(job_id)
            self._update_state(job_id, PipelineStage.RENDERING, 0.7, "Rendering video...")

            output_dir = Path(self.settings.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{job_id}.mp4"

            video_clip_map = {f"clip_{i:03d}": vpath for i, vpath in enumerate(video_paths)}

            def on_render_progress(progress: float):
                self._update_state(
                    job_id,
                    PipelineStage.RENDERING,
                    0.7 + progress * 0.25,
                    f"Rendering: {progress * 100:.0f}%",
                )

            render_result = self.renderer.render(
                edl, video_clip_map, audio_path, output_path, on_render_progress
            )

            # Complete
            self._update_state(job_id, PipelineStage.COMPLETE, 1.0, "Processing complete!")
            state = self._jobs[job_id]
            state.output_path = render_result.output_path
            state.completed_at = datetime.now(UTC)

            # Cleanup temp files
            self.temp_store.cleanup_job(job_id)

            return state

        except RezaaError:
            self._update_state(job_id, PipelineStage.FAILED, message=str(state.error))
            raise
        except Exception as e:
            self._update_state(job_id, PipelineStage.FAILED, message=str(e))
            raise ProcessingError(f"Pipeline failed: {e}")

    def _update_state(
        self, job_id: str, stage: PipelineStage, progress: float | None = None, message: str = ""
    ):
        """Update pipeline state."""
        if job_id in self._jobs:
            state = self._jobs[job_id]
            state.stage = stage
            if progress is not None:
                state.progress = progress
            state.message = message
            state.updated_at = datetime.now(UTC)
            if stage == PipelineStage.FAILED:
                state.error = message

    def _check_cancelled(self, job_id: str):
        """Check if job has been cancelled and raise if so."""
        if job_id in self._cancelled:
            raise ProcessingError("Job was cancelled", component="pipeline")

    def delete_job_data(self, job_id: str) -> None:
        """Delete all data for a job (privacy compliance)."""
        self.feature_store.delete_job_data(job_id)
        self.temp_store.cleanup_job(job_id)
        if job_id in self._jobs:
            del self._jobs[job_id]
        self._cancelled.discard(job_id)
