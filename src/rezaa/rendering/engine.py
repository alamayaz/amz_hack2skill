"""Rendering engine — assembles final video from EDL using FFmpeg."""

import json
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path

from rezaa.config import get_settings
from rezaa.models.edl import EditDecisionList
from rezaa.models.errors import RenderingError
from rezaa.models.render import RenderResult
from rezaa.rendering.ffmpeg_builder import FFmpegFilterGraphBuilder
from rezaa.rendering.progress import FFmpegProgressMonitor

logger = logging.getLogger(__name__)


class RenderingEngine:
    """Renders final video from an EditDecisionList using FFmpeg."""

    def __init__(self):
        self.builder = FFmpegFilterGraphBuilder()
        self.settings = get_settings()

    def render(
        self,
        edl: EditDecisionList,
        video_clips: dict[str, Path],
        audio_path: Path,
        output_path: Path,
        progress_callback: Callable[[float], None] | None = None,
    ) -> RenderResult:
        """Render the EDL to a final video file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = self.build_ffmpeg_command(edl, video_clips, audio_path, output_path)

        monitor = FFmpegProgressMonitor(edl.total_duration, progress_callback)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            stderr_lines = []
            for line in process.stderr:
                stderr_lines.append(line)
                monitor.parse_line(line)

            process.wait()

            if process.returncode != 0:
                stderr_text = "".join(stderr_lines[-30:])
                # Dump full command to a debug file for inspection
                debug_path = output_path.parent / "ffmpeg_debug.txt"
                debug_path.write_text(
                    "COMMAND:\n" + " ".join(cmd) + "\n\nSTDERR:\n" + "".join(stderr_lines)
                )
                logger.error("FFmpeg failed (code %d). Debug at: %s", process.returncode, debug_path)
                raise RenderingError(
                    f"FFmpeg exited with code {process.returncode}",
                    details={"stderr": stderr_text, "debug_file": str(debug_path)},
                )

            # Notify 100%
            if progress_callback:
                progress_callback(1.0)

            return self.validate_output(output_path, edl)

        except FileNotFoundError:
            raise RenderingError(
                "FFmpeg not found. Please install FFmpeg.",
                details={"command": "ffmpeg"},
            )
        except RenderingError:
            raise
        except Exception as e:
            raise RenderingError(
                f"Rendering failed: {e}",
                details={"error": str(e)},
            )

    def build_ffmpeg_command(
        self,
        edl: EditDecisionList,
        video_clips: dict[str, Path],
        audio_path: Path,
        output_path: Path,
    ) -> list[str]:
        """Build the complete FFmpeg command."""
        if not edl.clip_decisions:
            raise RenderingError("No clip decisions in EDL")

        # Prepare clip decisions with paths
        clip_dicts = []
        for cd in edl.clip_decisions:
            if cd.clip_id not in video_clips:
                raise RenderingError(f"Clip '{cd.clip_id}' not found in provided video clips")
            clip_dicts.append(
                {
                    "clip_path": str(video_clips[cd.clip_id]),
                    "source_start": cd.source_start,
                    "source_end": cd.source_end,
                    "transition_type": cd.transition_type,
                    "transition_duration": cd.transition_duration,
                }
            )

        input_args, filter_complex = self.builder.build_filter_graph(
            clip_dicts,
            target_width=edl.target_width,
            target_height=edl.target_height,
            target_fps=edl.target_fps,
        )

        # Build audio filter
        audio_filter = self.builder.build_audio_filter(
            str(audio_path),
            trim_start=edl.audio_decision.trim_start,
            trim_end=edl.audio_decision.trim_end or edl.total_duration,
            fade_in=edl.audio_decision.fade_in,
            fade_out=edl.audio_decision.fade_out,
            volume=edl.audio_decision.volume,
        )

        # Audio input comes after all clip inputs (one per clip decision)
        audio_idx = len(clip_dicts)

        cmd = ["ffmpeg", "-y"]
        cmd.extend(input_args)
        cmd.extend(["-i", str(audio_path)])

        if audio_filter:
            full_filter = f"{filter_complex};\n[{audio_idx}:a]{audio_filter}[outa]"
            cmd.extend(["-filter_complex", full_filter])
            cmd.extend(["-map", "[outv]", "-map", "[outa]"])
        else:
            cmd.extend(["-filter_complex", filter_complex])
            cmd.extend(["-map", "[outv]", "-map", f"{audio_idx}:a"])

        cmd.extend(
            [
                "-c:v",
                self.settings.output_video_codec,
                "-crf",
                str(self.settings.output_crf),
                "-preset",
                self.settings.output_preset,
                "-c:a",
                self.settings.output_audio_codec,
                str(output_path),
            ]
        )

        return cmd

    def validate_output(self, output_path: Path, edl: EditDecisionList) -> RenderResult:
        """Validate the rendered output using ffprobe."""
        if not output_path.exists():
            raise RenderingError("Output file was not created")

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            probe = json.loads(result.stdout)
        except Exception as e:
            raise RenderingError(
                f"Failed to validate output: {e}",
                details={"output": str(output_path)},
            )

        duration = float(probe.get("format", {}).get("duration", 0))
        file_size = int(probe.get("format", {}).get("size", 0))

        # Extract codec info
        video_codec = ""
        audio_codec = ""
        width = edl.target_width
        height = edl.target_height
        fps = edl.target_fps

        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "video":
                video_codec = stream.get("codec_name", "h264")
                width = int(stream.get("width", width))
                height = int(stream.get("height", height))
                r_fps = stream.get("r_frame_rate", "30/1")
                if "/" in str(r_fps):
                    num, den = r_fps.split("/")
                    fps = int(num) / int(den) if int(den) > 0 else 30.0
            elif stream.get("codec_type") == "audio":
                audio_codec = stream.get("codec_name", "aac")

        return RenderResult(
            output_path=str(output_path),
            duration=duration,
            file_size_bytes=file_size,
            video_codec=video_codec,
            audio_codec=audio_codec,
            container_format="mp4",
            width=width,
            height=height,
            fps=fps,
        )
