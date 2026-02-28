"""FFmpeg filter graph construction."""


class FFmpegFilterGraphBuilder:
    """Builds FFmpeg filter graphs for video editing."""

    def build_filter_graph(
        self,
        clip_decisions: list[dict],
        target_width: int = 1920,
        target_height: int = 1080,
        target_fps: float = 30.0,
    ) -> tuple[list[str], str]:
        """Build a filter graph with trim+setpts+concat chain.

        Returns:
            Tuple of (input_args, filter_complex string)
        """
        if not clip_decisions:
            return [], ""

        input_args = []
        filter_parts = []
        concat_inputs = []

        # Each clip decision gets its own -i input so the same file can be
        # referenced multiple times without stream exhaustion.
        for i, cd in enumerate(clip_decisions):
            clip_path = cd.get("clip_path", "")
            input_args.extend(["-i", clip_path])

            src_start = cd["source_start"]
            src_end = cd["source_end"]

            # Trim, scale, setpts
            vid_label = f"v{i}"
            filter_parts.append(
                f"[{i}:v]trim=start={src_start:.4f}:end={src_end:.4f},"
                f"setpts=PTS-STARTPTS,"
                f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,"
                f"fps={target_fps}"
                f"[{vid_label}]"
            )
            concat_inputs.append(f"[{vid_label}]")

            # Build transition filter if needed
            transition = cd.get("transition_type", "cut")
            if transition != "cut" and i > 0:
                trans_dur = cd.get("transition_duration", 0.5)
                trans_filter = self.build_transition_filter(transition, trans_dur, i)
                if trans_filter:
                    filter_parts.append(trans_filter)

        # Concat
        n = len(clip_decisions)
        concat_str = "".join(concat_inputs) + f"concat=n={n}:v=1:a=0[outv]"
        filter_parts.append(concat_str)

        filter_complex = ";\n".join(filter_parts)
        return input_args, filter_complex

    def build_transition_filter(self, transition_type: str, duration: float, index: int) -> str:
        """Build a transition filter string."""
        if transition_type == "fade":
            return f"[v{index}]fade=t=in:d={duration:.2f}[v{index}]"
        elif transition_type == "crossfade":
            # xfade requires special handling in the concat approach
            return ""
        return ""

    def build_audio_filter(
        self,
        audio_path: str,
        trim_start: float = 0.0,
        trim_end: float = 0.0,
        fade_in: float = 0.0,
        fade_out: float = 0.0,
        volume: float = 1.0,
    ) -> str:
        """Build audio filter string."""
        parts = []
        if trim_start > 0 or trim_end > 0:
            parts.append(f"atrim=start={trim_start:.4f}:end={trim_end:.4f}")
            parts.append("asetpts=PTS-STARTPTS")
        if fade_in > 0:
            parts.append(f"afade=t=in:d={fade_in:.2f}")
        if fade_out > 0:
            fade_start = max(0, trim_end - trim_start - fade_out)
            parts.append(f"afade=t=out:st={fade_start:.2f}:d={fade_out:.2f}")
        if volume != 1.0:
            parts.append(f"volume={volume:.2f}")
        return ",".join(parts)
