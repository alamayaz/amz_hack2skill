"""FFmpeg filter graph construction."""

# Maps our transition names to FFmpeg xfade transition names
_XFADE_MAP: dict[str, str] = {
    "fade": "fade",
    "crossfade": "fade",
    "wipeleft": "wipeleft",
    "wiperight": "wiperight",
    "slideleft": "slideleft",
    "slideright": "slideright",
    "fadeblack": "fadeblack",
    "fadewhite": "fadewhite",
    "dissolve": "dissolve",
    "zoomin": "zoomin",
    "circleopen": "circleopen",
    "radial": "radial",
}

_DEFAULT_TRANSITION_DURATION = 0.5


class FFmpegFilterGraphBuilder:
    """Builds FFmpeg filter graphs for video editing."""

    @staticmethod
    def _map_transition_to_xfade(transition_type: str) -> str | None:
        """Map a transition type name to its FFmpeg xfade name.

        Returns None for 'cut' (no xfade needed).
        """
        if transition_type == "cut":
            return None
        return _XFADE_MAP.get(transition_type)

    @staticmethod
    def _all_cuts(clip_decisions: list[dict]) -> bool:
        """Return True if every clip uses a 'cut' transition."""
        return all(cd.get("transition_type", "cut") == "cut" for cd in clip_decisions)

    def build_filter_graph(
        self,
        clip_decisions: list[dict],
        target_width: int = 1920,
        target_height: int = 1080,
        target_fps: float = 30.0,
    ) -> tuple[list[str], str, float]:
        """Build a filter graph for the given clip decisions.

        Uses concat when all transitions are cuts, xfade chain otherwise.

        Returns:
            Tuple of (input_args, filter_complex string, actual_video_duration)
        """
        if not clip_decisions:
            return [], "", 0.0

        if self._all_cuts(clip_decisions):
            return self._build_concat_graph(clip_decisions, target_width, target_height, target_fps)
        return self._build_xfade_graph(clip_decisions, target_width, target_height, target_fps)

    # ------------------------------------------------------------------
    # Concat graph (all cuts)
    # ------------------------------------------------------------------

    def _build_concat_graph(
        self,
        clip_decisions: list[dict],
        target_width: int,
        target_height: int,
        target_fps: float,
    ) -> tuple[list[str], str, float]:
        """Build a simple concat filter graph (used when all transitions are cuts)."""
        input_args: list[str] = []
        filter_parts: list[str] = []
        concat_inputs: list[str] = []
        total_dur = 0.0

        for i, cd in enumerate(clip_decisions):
            clip_path = cd.get("clip_path", "")
            input_args.extend(["-i", clip_path])

            src_start = cd["source_start"]
            src_end = cd["source_end"]
            total_dur += src_end - src_start

            vid_label = f"v{i}"
            filter_parts.append(
                f"[{i}:v]trim=start={src_start:.4f}:end={src_end:.4f},"
                f"setpts=PTS-STARTPTS,"
                f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,"
                f"fps={target_fps}"
                f"[{vid_label}]"
            )
            concat_inputs.append(f"[{vid_label}]")

        n = len(clip_decisions)
        concat_str = "".join(concat_inputs) + f"concat=n={n}:v=1:a=0[outv]"
        filter_parts.append(concat_str)

        filter_complex = ";\n".join(filter_parts)
        return input_args, filter_complex, total_dur

    # ------------------------------------------------------------------
    # Xfade graph (non-cut transitions)
    # ------------------------------------------------------------------

    def _build_xfade_graph(
        self,
        clip_decisions: list[dict],
        target_width: int,
        target_height: int,
        target_fps: float,
    ) -> tuple[list[str], str, float]:
        """Build an xfade-chain filter graph for non-cut transitions."""
        input_args: list[str] = []
        filter_parts: list[str] = []

        # Step 1: trim + scale each clip
        for i, cd in enumerate(clip_decisions):
            clip_path = cd.get("clip_path", "")
            input_args.extend(["-i", clip_path])

            src_start = cd["source_start"]
            src_end = cd["source_end"]

            vid_label = f"v{i}"
            filter_parts.append(
                f"[{i}:v]trim=start={src_start:.4f}:end={src_end:.4f},"
                f"setpts=PTS-STARTPTS,"
                f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,"
                f"fps={target_fps}"
                f"[{vid_label}]"
            )

        n = len(clip_decisions)

        # Single clip — no xfade needed, just passthrough
        if n == 1:
            clip_dur = clip_decisions[0]["source_end"] - clip_decisions[0]["source_start"]
            filter_parts.append("[v0]copy[outv]")
            return input_args, ";\n".join(filter_parts), clip_dur

        # Step 2: build xfade chain
        accumulated = clip_decisions[0]["source_end"] - clip_decisions[0]["source_start"]
        prev_label = "[v0]"

        for i in range(1, n):
            cd = clip_decisions[i]
            transition = cd.get("transition_type", "cut")
            trans_dur = cd.get("transition_duration", _DEFAULT_TRANSITION_DURATION)

            xfade_name = self._map_transition_to_xfade(transition)
            if xfade_name is None:
                # cut: simulate as ultra-short xfade
                xfade_name = "fade"
                trans_dur = 0.001

            offset = max(0, accumulated - trans_dur)
            out_label = "outv" if i == n - 1 else f"x{i - 1}"

            filter_parts.append(
                f"{prev_label}[v{i}]xfade=transition={xfade_name}"
                f":duration={trans_dur:.4f}:offset={offset:.4f}[{out_label}]"
            )

            clip_dur = cd["source_end"] - cd["source_start"]
            accumulated = offset + clip_dur
            prev_label = f"[{out_label}]"

        return input_args, ";\n".join(filter_parts), accumulated

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
