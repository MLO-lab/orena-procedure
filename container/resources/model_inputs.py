"""One question -- Request fields plus its clip -- into model inputs.

Deliberately depends on nothing the training export carried. `frames_indices` is
recomputed from `(start_time, end_time, question)` by the same `sampling.py` the export
used, the pixels come from the delivered 5 fps clip, and the metadata is built on that
same 5 fps timeline. Verified against all 10,000 exported rows -- see
`procedure_track/phase0/FINDINGS.md`.

`build_metadata`, `with_system` and `_encode` are copied from `segment_track/` and
`procedure_track/collate_procedure.py`; `check_vendored.py` asserts they still match.
"""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import numpy as np
import torch
from PIL import Image
from transformers.video_utils import VideoMetadata

import clip_decode
from prompts_segpro import build_system_prompt, window_line
from sampling import DEFAULT_FRAME_SIZE, DEFAULT_N_MAX, FPS, local_lookup_index, sample_indices_5fps

# The video processor caps a whole video at size["longest_edge"] TOTAL pixels
# (default ~25M) and silently downscales frames to fit -- 768 frames at 640x360 came
# back as 224x128, 14 tokens/frame, with the burned-in clock unreadable.
VIDEO_TOTAL_PIXEL_BUDGET = 1024 * 640 * 360


def build_metadata(frames_indices: list[int], base_fps: float, total_num_frames: int,
                   size: tuple[int, int] = DEFAULT_FRAME_SIZE) -> VideoMetadata:
    """Metadata that makes the processor emit ABSOLUTE video timestamps.

    `Qwen3VLProcessor` renders one `<t seconds>` marker per fused frame pair, with `t`
    averaged from `frames_indices / fps`. Omit either field and it warns once, falls
    back to `fps=24`, and silently produces clip-relative nonsense.
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


def with_system(messages: list[dict], system_prompt: str | None) -> list[dict]:
    if not system_prompt:
        return messages
    return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}] + messages


def encode(processor, text: str, video, meta):
    out = processor(text=[text], videos=[video], video_metadata=[meta],
                    do_sample_frames=False, return_tensors="pt",
                    size={"longest_edge": VIDEO_TOTAL_PIXEL_BUDGET, "shortest_edge": 4096})
    if "pixel_values_videos" in out:
        out["pixel_values_videos"] = out["pixel_values_videos"].to(torch.bfloat16)
    return out


def read_frames(clip: Path, idx5: list[int], start_time: float,
                size=DEFAULT_FRAME_SIZE, pool=None) -> np.ndarray:
    n_clip = clip_decode.frame_count(clip)
    local = [local_lookup_index(i, start_time, n_clip) for i in idx5]
    raw = clip_decode.read(clip, local, pool)
    if raw.shape[2] == size[0] and raw.shape[1] == size[1]:
        return raw
    return np.stack([np.asarray(Image.fromarray(f).resize(size, Image.BILINEAR)) for f in raw])


def build_inputs(processor, clip: Path, start_time: float, end_time: float, question: str,
                 system_prompt: str | None = None, n_max: int = DEFAULT_N_MAX,
                 size=DEFAULT_FRAME_SIZE, pool=None, frames_indices: list[int] | None = None):
    """`frames_indices` overrides the sampler; prompt, window line and metadata timeline
    stay identical, so a second pass differs from the first only in which frames it sees."""
    timings = {}
    t0 = monotonic()
    idx5 = (list(frames_indices) if frames_indices is not None
            else sample_indices_5fps(start_time, end_time, question, n_max=n_max))
    video = read_frames(clip, idx5, start_time, size, pool)
    timings["decode"] = monotonic() - t0

    total5 = int(round(end_time * FPS)) + 1
    meta = build_metadata(idx5, FPS, total5, size)
    messages = [{"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": f"{window_line(start_time, end_time)} {question}"},
    ]}]
    text = processor.apply_chat_template(
        with_system(messages, system_prompt or build_system_prompt()),
        tokenize=False, add_generation_prompt=True, enable_thinking=False)

    t1 = monotonic()
    encoded = encode(processor, text, video, meta)
    timings["encode"] = monotonic() - t1
    return encoded, idx5, timings
