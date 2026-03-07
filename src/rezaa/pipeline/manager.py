"""Pipeline manager — orchestrates the full processing pipeline."""

import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from rezaa.agents.alignment_agent import BeatClipAlignmentAgent
from rezaa.agents.audio_agent import AudioAnalysisAgent
from rezaa.agents.video_agent import VideoUnderstandingAgent
from rezaa.config import get_settings
from rezaa.models.benchmark import (
    AlignmentMetrics,
    AudioMetrics,
    BenchmarkReport,
    InputSummary,
    OrchestrationMetrics,
    RenderMetrics,
    StageTiming,
    VideoClipMetrics,
)
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

        pipeline_start = time.perf_counter()
        audio_analysis = None
        video_analyses = []
        alignment = None
        edl = None
        render_result = None

        try:
            # Stage 1: Audio Extraction + Analysis
            self._check_cancelled(job_id)
            self._update_state(
                job_id,
                PipelineStage.AUDIO_EXTRACTION,
                0.1,
                "Extracting audio features...",
            )
            t0 = time.perf_counter()
            audio_analysis = self.audio_agent.analyze(audio_path)
            state.stage_timings["audio_analysis"] = round(time.perf_counter() - t0, 4)
            self.feature_store.save_features(job_id, "audio_analysis", audio_analysis)

            # Stage 2: Video Extraction + Analysis
            self._check_cancelled(job_id)
            self._update_state(
                job_id,
                PipelineStage.VIDEO_EXTRACTION,
                0.3,
                "Extracting video features...",
            )
            t0 = time.perf_counter()
            for i, vpath in enumerate(video_paths):
                self._check_cancelled(job_id)
                clip_id = f"clip_{i:03d}"
                va = self.video_agent.analyze(vpath, clip_id)
                video_analyses.append(va)
                self.feature_store.save_features(job_id, f"video_analysis_{clip_id}", va)
            state.stage_timings["video_analysis"] = round(time.perf_counter() - t0, 4)

            # Stage 3: Alignment
            self._check_cancelled(job_id)
            self._update_state(job_id, PipelineStage.ALIGNMENT, 0.5, "Aligning clips to beats...")
            t0 = time.perf_counter()
            alignment = self.alignment_agent.align(
                audio_analysis, video_analyses, prefs.target_duration
            )
            state.stage_timings["alignment"] = round(time.perf_counter() - t0, 4)
            self.feature_store.save_features(job_id, "alignment", alignment)

            # Stage 4: Orchestration
            self._check_cancelled(job_id)
            self._update_state(
                job_id,
                PipelineStage.ORCHESTRATION,
                0.6,
                "Making editing decisions...",
            )
            t0 = time.perf_counter()
            edl = self.orchestrator.orchestrate(audio_analysis, video_analyses, alignment, prefs)
            state.stage_timings["orchestration"] = round(time.perf_counter() - t0, 4)
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

            t0 = time.perf_counter()
            render_result = self.renderer.render(
                edl, video_clip_map, audio_path, output_path, on_render_progress
            )
            state.stage_timings["rendering"] = round(time.perf_counter() - t0, 4)

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
        finally:
            state.stage_timings["total"] = round(time.perf_counter() - pipeline_start, 4)
            state.benchmark_report = self._build_benchmark_report(
                job_id, state, audio_analysis, video_analyses, alignment, edl, render_result
            ).model_dump()

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

    def _build_benchmark_report(
        self,
        job_id: str,
        state: PipelineState,
        audio_analysis,
        video_analyses: list,
        alignment,
        edl,
        render_result,
    ) -> BenchmarkReport:
        """Assemble a BenchmarkReport from whatever data is available."""
        timings = [
            StageTiming(stage_name=k, duration_seconds=v)
            for k, v in state.stage_timings.items()
            if k != "total"
        ]

        input_summary = InputSummary()
        audio_metrics = AudioMetrics()
        if audio_analysis:
            input_summary.audio_duration_seconds = audio_analysis.features.duration
            audio_metrics = AudioMetrics(
                bpm=audio_analysis.features.bpm,
                beat_count=len(audio_analysis.features.beat_timestamps),
                confidence=audio_analysis.confidence,
                beat_strength_avg=(
                    sum(audio_analysis.beat_strength) / len(audio_analysis.beat_strength)
                    if audio_analysis.beat_strength
                    else 0.0
                ),
                drop_count=len(audio_analysis.features.drop_timestamps),
            )

        video_metrics = []
        if video_analyses:
            input_summary.num_clips = len(video_analyses)
            input_summary.total_video_duration_seconds = sum(
                va.features.duration for va in video_analyses
            )
            video_metrics = [
                VideoClipMetrics(
                    clip_id=va.clip_id,
                    duration=va.features.duration,
                    motion_score=va.features.motion_score,
                    energy_score=va.features.energy_score,
                    temporal_consistency=va.temporal_consistency,
                    recommended_usage=va.recommended_usage,
                )
                for va in video_analyses
            ]

        alignment_metrics = AlignmentMetrics()
        if alignment:
            alignment_metrics = AlignmentMetrics(
                coverage=alignment.coverage,
                average_energy_match=alignment.average_energy_match,
                clips_used=len(alignment.clips_used),
                total_placements=len(alignment.placements),
            )

        orchestration_metrics = OrchestrationMetrics()
        if edl:
            transition_types = list({cd.transition_type for cd in edl.clip_decisions})
            orchestration_metrics = OrchestrationMetrics(
                method=edl.edl_metadata.get("orchestration_method", "unknown"),
                clip_decisions_count=len(edl.clip_decisions),
                transition_types_used=transition_types,
                total_duration=edl.total_duration,
            )

        render_metrics = RenderMetrics()
        if render_result:
            render_metrics = RenderMetrics(
                output_duration=render_result.duration,
                file_size_mb=render_result.file_size_mb,
                video_codec=render_result.video_codec,
                resolution=f"{render_result.width}x{render_result.height}",
                fps=render_result.fps,
            )

        return BenchmarkReport(
            job_id=job_id,
            total_pipeline_duration_seconds=state.stage_timings.get("total", 0.0),
            stage_timings=timings,
            input_summary=input_summary,
            audio_metrics=audio_metrics,
            video_metrics=video_metrics,
            alignment_metrics=alignment_metrics,
            orchestration_metrics=orchestration_metrics,
            render_metrics=render_metrics,
        )

    def delete_job_data(self, job_id: str) -> None:
        """Delete all data for a job (privacy compliance)."""
        self.feature_store.delete_job_data(job_id)
        self.temp_store.cleanup_job(job_id)
        if job_id in self._jobs:
            del self._jobs[job_id]
        self._cancelled.discard(job_id)
