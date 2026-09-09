"""Turns exported segment records into model inputs, for training and generation.

Shared by the trainer, the eval-sample callback and the evaluator so all three
render a clip identically -- the frame track learned this the hard way: the moment
two paths disagree about what the model is shown, every number becomes a
train/inference mismatch.

Two processor contracts this file exists to enforce (segment_track/plan.md §2.7):
  * `do_sample_frames=False` -- otherwise the video processor RE-SAMPLES the
    already-sampled frames using indices derived from `total_num_frames`, which
    either raises IndexError or silently overwrites `frames_indices`.
  * `video_metadata` carrying real `fps` and ABSOLUTE `frames_indices` -- omit it
    and the processor warns once, assumes `fps=24`, and renders clip-relative
    timestamps while the targets are video-absolute.
"""

from __future__ import annotations

import torch

from clip_sampling import (
    DEFAULT_FRAME_SIZE, build_metadata, frame_count, frame_file, load_frames,
)

# Qwen3.5 renders an already-answered assistant turn with an empty-but-present
# think block. Everything up to and including this literal is prompt.
ASSISTANT_MARKER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def with_system(messages: list[dict], system_prompt: str | None) -> list[dict]:
    """Prepend the system turn, or return `messages` untouched when there is none."""
    if not system_prompt:
        return messages
    return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}] + messages


def clip_inputs(record: dict, frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE):
    """`(video_array, VideoMetadata)` for one exported record.

    `total_num_frames` is the SOURCE video's length, recomputed here rather than
    stored: it is only used for `duration`/`sampled_fps` bookkeeping, and
    `frame_count` is lru_cached per directory so this costs ~40 stat calls per
    video per process, once.
    """
    directory = record["frame_dir"]
    indices = record["frames_indices"]
    video = load_frames([frame_file(directory, i) for i in indices], frame_size)
    meta = build_metadata(indices, record["base_fps"], frame_count(directory), frame_size)
    return video, meta


def _encode(processor, text: str, video, meta):
    return processor(text=[text], videos=[video], video_metadata=[meta],
                     do_sample_frames=False, return_tensors="pt")


def build_collate_fn(processor, system_prompt: str | None = None,
                     frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE):
    """Full-sequence inputs with everything up to the assistant marker masked out
    of the loss, so only the answer is a training target.

    The marker is located inside the rendered text and sliced, rather than
    re-derived from a separate `add_generation_prompt=True` call: that call renders
    a *fresh* turn (thinking left open) and would land on a different boundary. A
    literal string prefix guarantees the tokenized prompt is an exact prefix of the
    full tokenization -- asserted in tests/test_collate.py.
    """

    def collate_fn(examples: list[dict]) -> dict[str, torch.Tensor]:
        input_ids_list, labels_list, mm_type_list = [], [], []
        pixel_list, grid_list = [], []

        for ex in examples:
            video, meta = clip_inputs(ex, frame_size)

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
        # 0 = text, 1 = image, 2 = video (Qwen's M-RoPE convention); padding is
        # typed 0 like any other non-visual filler.
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
                            frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE):
    """Prompt-only inputs for `model.generate()` -- the inference counterpart.

    Uses `add_generation_prompt=True, enable_thinking=False` so the model is asked
    for a bare answer, matching what training supervised.
    """
    video, meta = clip_inputs(record, frame_size)
    prompt_text = processor.apply_chat_template(
        with_system(record["messages"][:1], system_prompt),
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return _encode(processor, prompt_text, video, meta)
