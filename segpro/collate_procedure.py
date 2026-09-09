"""Exported procedure records -> model inputs, for training and for generation.

Mirrors `segment_track/collate.py` and imports its stable primitives rather than
re-deriving them. Named `collate_procedure` and not `collate` on purpose: both
directories land on `sys.path`, so a module named `collate` here would shadow the
segment one it imports from and deadlock on a circular import.

One thing genuinely differs and it is the reason this file exists at all:

    segment    frames_indices are NATIVE-fps indices, metadata carries fps=25 or 30
    procedure  frames_indices are 5 fps indices,      metadata carries fps=5.0

Training reads native-fps JPEGs while the container reads a 5 fps MP4, so 5 fps is the
canonical timeline and the pixel lookup converts at the edge (`to_native_index`). Get
this wrong and the rendered `<... seconds>` markers are silently 5x off -- exactly what
`verify_fps_roundtrip.py` exists to catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROC_DIR))
sys.path.insert(0, str(PROC_DIR.parent / "segment_track"))

from clip_sampling import build_metadata, frame_count, frame_file, load_frames  # noqa: E402
from collate import ASSISTANT_MARKER, with_system  # noqa: E402,F401 (re-exported)

from sampling import DEFAULT_FRAME_SIZE, FPS, native_lookup_index  # noqa: E402

# The two clusters this pipeline runs on. Scripts and tests that need pixels detect
# whichever root exists instead of hardcoding one cluster's mount.
KNOWN_DATA_ROOTS = (
    "/projects/datasets_ML/orena",
    "/mnt/vast/workspaces/VL_LeJepa/data/orena",
)


def detect_data_root() -> str | None:
    import os

    env = os.environ.get("ORENA_DATA_ROOT")
    if env:
        return env
    return next((r for r in KNOWN_DATA_ROOTS if Path(r).is_dir()), None)


def remap_frame_dir(frame_dir: str, frames_root: str | None) -> str:
    """Re-root an exported `frame_dir` onto a different filesystem.

    The export stores absolute paths, so a JSONL built on one cluster does not resolve
    on another. The trailing three components (`<dataset>/<folder>/<stem>`) are the
    stable part, so re-rooting is exact and needs no re-export.
    """
    if not frames_root:
        return frame_dir
    parts = Path(frame_dir).parts[-3:]
    return str(Path(frames_root).joinpath(*parts))


def _load_frames_parallel(paths, size, workers: int):
    """`load_frames` with the JPEG decode spread over a thread pool.

    768 frames decode in ~6.4 s serially, which is 57% of a generation call. PIL
    releases the GIL inside decode and resize, so threads scale this well.
    """
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    def one(p):
        return np.asarray(Image.open(p).convert("RGB").resize(size, Image.BILINEAR))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return np.stack(list(pool.map(one, paths)))


def clip_inputs(record: dict, frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
                frames_root: str | None = None, load_workers: int = 0):
    """`(video_array, VideoMetadata)` for one exported record.

    `total_num_frames` is the length of the observed window in 5 fps frames, not the
    source video's native frame count: the processor only uses it for duration
    bookkeeping, and at inference the container has no access to anything else.
    """
    idx5 = record["frames_indices"]
    base_fps = record["base_fps"]
    # `base_fps` no longer reaches build_metadata (the metadata is 5 fps), so nothing
    # downstream would notice a bad value: to_native_index would just collapse every
    # index onto frame 0 and the model would train on one repeated frame with
    # perfectly correct timestamps. Validate it here or not at all.
    if not base_fps or base_fps <= 0:
        raise ValueError(f"base_fps must be positive, got {base_fps!r}")
    if len(idx5) < 2 or len(idx5) % 2:
        raise ValueError(f"frames_indices must be even and >= 2, got {len(idx5)}")
    directory = remap_frame_dir(record["frame_dir"], frames_root)
    n_avail = frame_count(directory)
    if n_avail == 0:
        raise FileNotFoundError(f"no frames under {directory!r} -- wrong --frames-root?")
    paths = [frame_file(directory, native_lookup_index(i, base_fps, n_avail)) for i in idx5]
    video = (_load_frames_parallel(paths, frame_size, load_workers) if load_workers > 1
             else load_frames(paths, frame_size))
    total5 = int(round(record["end_time"] * FPS)) + 1
    meta = build_metadata(idx5, FPS, total5, frame_size)
    return video, meta


# The video processor caps a whole video at size["longest_edge"] TOTAL pixels
# (default ~25M) and silently downscales frames to fit -- 768 frames at 640x360
# came back as 224x128, 14 tokens/frame, with the burned-in clock unreadable.
# Frame size is decided by our sampler config, so lift the cap out of the way.
VIDEO_TOTAL_PIXEL_BUDGET = 1024 * 640 * 360


def _encode(processor, text: str, video, meta):
    out = processor(text=[text], videos=[video], video_metadata=[meta],
                    do_sample_frames=False, return_tensors="pt",
                    size={"longest_edge": VIDEO_TOTAL_PIXEL_BUDGET, "shortest_edge": 4096})
    # 768 frames is 1.36 GB of fp32 pixels per sample, and 64 of those are in flight
    # across the dataloader workers -- enough to OOM 400 GB of host RAM. The patch
    # embedding casts to the bf16 weight dtype on entry anyway, so the rounding is
    # identical and this only moves it earlier.
    if "pixel_values_videos" in out:
        out["pixel_values_videos"] = out["pixel_values_videos"].to(torch.bfloat16)
    return out


def build_collate_fn(processor, system_prompt: str | None = None,
                     frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
                     frames_root: str | None = None):
    """Full-sequence inputs with everything up to the assistant marker masked out.

    Sequence length is VARIABLE here, unlike the segment track: the capped-5s policy
    gives short windows far fewer frames than long ones. Padding is therefore doing
    real work every batch rather than being a formality.
    """

    def collate_fn(examples: list[dict]) -> dict[str, torch.Tensor]:
        input_ids_list, labels_list, mm_type_list = [], [], []
        pixel_list, grid_list = [], []

        for ex in examples:
            video, meta = clip_inputs(ex, frame_size, frames_root)

            full_text = processor.apply_chat_template(
                with_system(ex["messages"], system_prompt), tokenize=False)
            cut = full_text.rindex(ASSISTANT_MARKER) + len(ASSISTANT_MARKER)

            full = _encode(processor, full_text, video, meta)
            prompt = _encode(processor, full_text[:cut], video, meta)
            prompt_len = prompt["input_ids"].shape[1]

            ids = full["input_ids"][0]
            labels = ids.clone()
            labels[:prompt_len] = -100

            input_ids_list.append(ids)
            labels_list.append(labels)
            mm_type_list.append(full["mm_token_type_ids"][0])
            pixel_list.append(full["pixel_values_videos"])
            grid_list.append(full["video_grid_thw"])

        pad_id = processor.tokenizer.pad_token_id
        max_len = max(x.shape[0] for x in input_ids_list)
        n = len(examples)

        batch_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
        batch_mask = torch.zeros((n, max_len), dtype=torch.long)
        batch_labels = torch.full((n, max_len), -100, dtype=torch.long)
        batch_mm = torch.zeros((n, max_len), dtype=torch.long)

        for i, (ids, lbl, mm) in enumerate(zip(input_ids_list, labels_list, mm_type_list)):
            k = ids.shape[0]
            batch_ids[i, :k] = ids
            batch_mask[i, :k] = 1
            batch_labels[i, :k] = lbl
            batch_mm[i, :k] = mm

        return {
            "input_ids": batch_ids,
            "attention_mask": batch_mask,
            "labels": batch_labels,
            "mm_token_type_ids": batch_mm,
            "pixel_values_videos": torch.cat(pixel_list, dim=0),
            "video_grid_thw": torch.cat(grid_list, dim=0),
        }

    return collate_fn


def build_generation_inputs(processor, record: dict, system_prompt: str | None = None,
                            frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
                            frames_root: str | None = None, load_workers: int = 0):
    """Prompt-only inputs for `model.generate()` -- the inference counterpart."""
    video, meta = clip_inputs(record, frame_size, frames_root, load_workers)
    prompt_text = processor.apply_chat_template(
        with_system(record["messages"][:1], system_prompt),
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return _encode(processor, prompt_text, video, meta)
