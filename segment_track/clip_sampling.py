"""Clip → frames, shared by the builder, the collator and the evaluator.

The segment track gives a `[start_time, end_time]` window over a source video.
Every consumer must turn that window into the *same* frame indices, the same
pixels and the same `VideoMetadata`, or training and inference silently disagree
about what the model was shown -- and, because Qwen3-VL derives its per-frame
timestamp markers from that metadata, about what time it is in the video.

Two rules the processor imposes (see segment_track/plan.md §2.7):
  * `n_frames` must be even -- `temporal_patch_size=2` fuses frames in pairs.
  * `frames_indices` must be ABSOLUTE indices into the source video, and `fps`
    must be the real fps, or the rendered `<... seconds>` markers are wrong.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
from PIL import Image
from transformers.video_utils import VideoMetadata

DEFAULT_N_FRAMES = 80
DEFAULT_FRAME_SIZE = (640, 360)  # (width, height)


def video_stem(video_id: str) -> str:
    return Path(video_id).stem


def frame_dir(root_dir: Path, dataset: str, video_id: str, frames_folder: str = "frames") -> Path:
    """Directory holding the extracted JPEGs for one video.

    Mirrors `focus`'s own layout (`<root>/<dataset>/<frames_folder>/<stem>/`), the
    same one `build_frame_sft_dataset.frame_path()` resolves against.
    """
    return Path(root_dir) / dataset / frames_folder / video_stem(video_id)


def frame_file(directory: Path | str, index: int) -> Path:
    """Coerces `directory` -- it arrives as a plain string when read back from the
    exported JSONL."""
    return Path(directory) / f"frame{index:07d}.jpg"


@functools.lru_cache(maxsize=512)
def frame_count(directory: Path) -> int:
    """Number of extracted frames, as `max_index + 1`, by binary search.

    `os.listdir` on these directories means enumerating up to ~370k entries over
    NFS; probing existence doubles-then-bisects in ~40 stat calls instead.
    Returns 0 when the directory holds no frames.
    """
    directory = Path(directory)
    if not frame_file(directory, 0).exists():
        return 0
    hi = 1
    while frame_file(directory, hi).exists():
        hi *= 2
        if hi > 2**26:  # ~67M frames; far beyond any real video
            raise RuntimeError(f"runaway frame probe in {directory}")
    lo = hi // 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if frame_file(directory, mid).exists():
            lo = mid
        else:
            hi = mid
    return lo + 1


def sample_frame_indices(
    start_time: float,
    end_time: float,
    base_fps: float,
    n_frames: int = DEFAULT_N_FRAMES,
    n_available: int | None = None,
) -> list[int]:
    """`n_frames` absolute frame indices spanning `[start_time, end_time]`.

    Fixed count, not fixed rate: clip durations span 1 s to 300 s, so a fixed rate
    would vary the sequence length ~300x and make the memory ceiling unpredictable.

    Clips too short to hold `n_frames` distinct frames (a 1 s clip at 25 fps has
    25) yield repeated indices rather than a shorter list -- constant sequence
    length is worth more than the duplicated tokens.
    """
    if n_frames % 2:
        raise ValueError(f"n_frames must be even (temporal_patch_size=2), got {n_frames}")
    if end_time < start_time:
        raise ValueError(f"end_time {end_time} precedes start_time {start_time}")

    first, last = round(start_time * base_fps), round(end_time * base_fps)
    if n_available is not None:
        first = min(first, max(n_available - 1, 0))
        last = min(last, max(n_available - 1, 0))
    return [int(round(x)) for x in np.linspace(first, last, n_frames)]


def load_frames(paths: list[Path], size: tuple[int, int] = DEFAULT_FRAME_SIZE) -> np.ndarray:
    """Stack the JPEGs into one `(T, H, W, 3) uint8` array -- a "video" as far as
    the processor is concerned. No video file is ever decoded."""
    frames = [np.asarray(Image.open(p).convert("RGB").resize(size, Image.BILINEAR)) for p in paths]
    return np.stack(frames)


def build_metadata(
    frames_indices: list[int],
    base_fps: float,
    total_num_frames: int,
    size: tuple[int, int] = DEFAULT_FRAME_SIZE,
) -> VideoMetadata:
    """Metadata that makes the processor emit ABSOLUTE video timestamps.

    `Qwen3VLProcessor` renders one `<t seconds>` marker per fused frame pair, with
    `t` averaged from `frames_indices / fps`. Omit either field and it warns once,
    falls back to `fps=24`, and silently produces clip-relative nonsense.
    """
    if not frames_indices:
        raise ValueError("frames_indices is empty")
    if not base_fps or base_fps <= 0:
        raise ValueError(f"base_fps must be positive, got {base_fps!r}")
    width, height = size
    return VideoMetadata(
        total_num_frames=total_num_frames,
        fps=float(base_fps),
        width=width,
        height=height,
        duration=total_num_frames / float(base_fps),
        frames_indices=list(frames_indices),
    )


def marker_times(frames_indices: list[int], base_fps: float) -> list[float]:
    """The timestamps the processor will render, in seconds.

    One per fused pair (`temporal_patch_size=2`), each the mean of its two frames.
    Mirrors `Qwen3VLProcessor._calculate_timestamps` so tests can assert against
    an independent computation rather than the processor's own output.
    """
    t = [i / float(base_fps) for i in frames_indices]
    return [(t[i] + t[i + 1]) / 2 for i in range(0, len(t) - 1, 2)]


def hhmmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
