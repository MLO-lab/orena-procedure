"""Which frames to show the model, in the 5 fps index space the challenge delivers.

Everything here is a pure function of `(window, question, n_max)`. It runs twice --
once at export time against native-fps JPEGs, once inside the submission container
against a 5 fps MP4 -- and the two MUST agree, so this module is the single source of
truth for both. See procedure_track/plan.md §2.1-§2.3.

Indices are always ABSOLUTE positions on a 5 fps grid over the source procedure, which
makes `index / 5.0` the real time-of-operation the model is asked to answer in. The
callers convert at the edges:

    export     native_idx = to_native_index(idx5, base_fps)   # find the JPEG
    inference  local_idx  = to_local_index(idx5, start_time)  # seek in the clip
    always     VideoMetadata(fps=5.0, frames_indices=idx5)

The sampling policy is "one frame every 5 s, capped at n_max": 5 s is the coarsest
spacing that puts a `time` answer inside its ±5 s tolerance, since frames fuse in pairs
and one frame per 5 s gives one timestamp anchor per 10 s. Windows too long for that
budget stretch to a uniform n_max grid, and for the 27.8% of questions that quote a
timestamp, half the budget is spent at full 5 s density around it instead.
"""

from __future__ import annotations

import re

FPS = 5.0                  # the challenge encodes every video track clip at exactly 5 fps
TARGET_PERIOD_S = 5.0      # one frame per 5 s -> one anchor per 10 s, = the time tolerance
DEFAULT_N_MAX = 576
# The cap the joint adapter trained on. Raising it to 768 lifts the `time` ceiling
# (heico 0.684 -> 0.780) but measured no accuracy gain, and stage 1 here is only a
# pointer -- the frames are better spent on the zoom (see inference.py).
#
# SEGMENT rows are untouched: a 5 min window at 5 s spacing is ~61 frames, far under
# either cap. This constant only moves the procedure half of the joint model.
#
# 512x288, not the segment track's 640x360: the vision tower is geometrically identical
# to the 9B's (patch 16, spatial_merge 2, temporal_patch 2), so 576 frames at this size
# is 41.5k visual tokens. The burned-in clock stays legible (checked on the worst case,
# natively-1280x720 lapchole), so the frame budget is worth more than the pixels.
DEFAULT_FRAME_SIZE = (512, 288)
DEFAULT_DENSE_FRAC = 0.5

_TS_RE = re.compile(r"\b(\d{1,2}):(\d{2}):(\d{2})\b")


def parse_question_timestamps(question: str) -> list[float]:
    """Seconds for every `hh:mm:ss` in the question, in order of appearance.

    Must be called on the RAW question. `build_procedure_sft_dataset` prepends a
    `Procedure observed from 00:00:00 to 02:47:31.` line, and parsing after that would
    make every row look anchored at its own window edges.
    """
    return [h * 3600 + m * 60 + s
            for h, m, s in ((int(a), int(b), int(c)) for a, b, c in _TS_RE.findall(question or ""))]


def to_native_index(idx5: int, base_fps: float) -> int:
    """5 fps index -> index into the extracted native-fps JPEGs."""
    return int(round(idx5 * base_fps / FPS))


def to_local_index(idx5: int, start_time: float) -> int:
    """5 fps index -> frame offset inside the delivered clip.

    The clip is trimmed and decoded from its own beginning, so its frame 0 is
    `start_time`. For PROCEDURE that is always 0 and this is the identity.
    """
    return int(idx5 - round(start_time * FPS))


def native_lookup_index(idx5: int, base_fps: float, n_available: int) -> int:
    """JPEG lookup for one sampled index, clamped to what is on disk.

    `frames_indices` must stay a pure function of (window, question, n_max) so the
    export and the container agree -- so availability NEVER filters the index list
    (the markers are derived from it). Only the pixel fetch clamps: a window whose
    tail extends past the extracted frames repeats the last frame instead.
    """
    if n_available <= 0:
        raise ValueError(f"n_available must be positive, got {n_available}")
    return min(to_native_index(idx5, base_fps), n_available - 1)


def local_lookup_index(idx5: int, start_time: float, n_clip: int) -> int:
    """The container-side counterpart of `native_lookup_index`, against the
    delivered 5 fps clip's own frame count."""
    if n_clip <= 0:
        raise ValueError(f"n_clip must be positive, got {n_clip}")
    return min(to_local_index(idx5, start_time), n_clip - 1)


def _uniform(lo: int, hi: int, n: int) -> list[int]:
    if n <= 1 or hi <= lo:
        return [lo] * max(n, 1)
    step = (hi - lo) / (n - 1)
    return [int(round(lo + i * step)) for i in range(n)]


def _every(lo: int, hi: int, period_s: float = TARGET_PERIOD_S) -> list[int]:
    """Indices at `period_s` spacing, always including both ends."""
    stride = max(1, int(round(period_s * FPS)))
    out = list(range(lo, hi + 1, stride))
    if out[-1] != hi:
        out.append(hi)
    return out


def _finalise(indices: list[int], n_max: int) -> list[int]:
    """Sorted, de-duplicated, at most `n_max`, and an EVEN count.

    Even because `temporal_patch_size=2` fuses frames in pairs; a ragged pair would
    silently drop a frame and shift every timestamp marker after it.
    """
    out = sorted(set(int(i) for i in indices))
    if len(out) > n_max:
        out = [out[i] for i in _uniform(0, len(out) - 1, n_max)]
        out = sorted(set(out))
    if len(out) % 2:
        # Never drop the window end -- it is the frame "so far" questions hinge on.
        # Subdivide the widest gap instead; fall back to a duplicate-free drop only
        # when every gap is already a single index.
        if len(out) == 1:
            out = out * 2
        else:
            width, k = max((b - a, i) for i, (a, b) in enumerate(zip(out, out[1:])))
            if width >= 2:
                out.insert(k + 1, out[k] + width // 2)
            else:
                del out[-2]
    return out


def sample_indices_5fps(start_time: float, end_time: float, question: str | None = None,
                        n_max: int = DEFAULT_N_MAX,
                        dense_frac: float = DEFAULT_DENSE_FRAC) -> list[int]:
    """Absolute 5 fps indices spanning `[start_time, end_time]`.

    Short enough windows come back at true 5 s spacing and use FAR fewer than `n_max`
    frames -- that is the point of the cap, and it is why sequence length is variable
    here where the segment track kept it fixed.
    """
    if end_time < start_time:
        raise ValueError(f"end_time {end_time} precedes start_time {start_time}")
    if n_max < 2 or n_max % 2:
        raise ValueError(f"n_max must be even and >= 2, got {n_max}")

    first, last = int(round(start_time * FPS)), int(round(end_time * FPS))
    if last <= first:
        return [first, first]

    full = _every(first, last)
    if len(full) <= n_max:
        return _finalise(full, n_max)

    anchors = [t for t in parse_question_timestamps(question or "")
               if start_time <= t <= end_time]
    if not anchors:
        return _finalise(_uniform(first, last, n_max), n_max)

    # Half the budget at full 5 s density where the question is pointing, half spread
    # over everything else so the rest of the procedure is still visible.
    n_dense = max(2, min(int(round(n_max * dense_frac)), n_max - 2))
    n_sparse = n_max - n_dense

    a_lo, a_hi = int(round(min(anchors) * FPS)), int(round(max(anchors) * FPS))
    if a_lo == a_hi:                                   # single anchor: centre a window on it
        half = int(round(n_dense / 2 * TARGET_PERIOD_S * FPS))
        a_lo, a_hi = a_lo - half, a_hi + half
    a_lo, a_hi = max(first, a_lo), min(last, a_hi)

    dense = _every(a_lo, a_hi)
    if len(dense) > n_dense:                           # anchor span wider than the budget
        dense = _uniform(a_lo, a_hi, n_dense)

    return _finalise(dense + _uniform(first, last, n_sparse), n_max)


def describe(indices: list[int]) -> dict:
    """Spacing summary, for the export's sanity print and the tests."""
    if len(indices) < 2:
        return {"n": len(indices), "min_gap_s": 0.0, "max_gap_s": 0.0, "at_target": 0.0}
    gaps = [(b - a) / FPS for a, b in zip(indices, indices[1:])]
    return {
        "n": len(indices),
        "min_gap_s": min(gaps),
        "max_gap_s": max(gaps),
        "at_target": sum(g <= TARGET_PERIOD_S + 1e-6 for g in gaps) / len(gaps),
    }
