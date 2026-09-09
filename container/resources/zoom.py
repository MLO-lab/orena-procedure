"""Stage-2 frame grid: re-sample densely around stage 1's guess.

The window and the prompt are unchanged -- only `frames_indices` differ -- so stage 2
still looks like the question-anchored rows the model trains on rather than an
out-of-distribution sub-clip. `to_seconds` doubles as the router: a stage-1 answer that
parses as HH:MM:SS is a `time` answer and gets a second pass.
"""

from __future__ import annotations

import re

from sampling import FPS, _finalise, _uniform

_TS = re.compile(r"^\s*(\d+):(\d\d):(\d\d)\s*$")


def to_seconds(text) -> float | None:
    """`"01:23:45"` -> 5025.0; anything that is not a bare timestamp -> None."""
    m = _TS.match(str(text))
    return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) if m else None


def zoom_indices(start_time: float, end_time: float, center_s: float, n_max: int,
                 half_width_s: float, dense_frac: float) -> list[int]:
    """`dense_frac` of the budget dense around `center_s`, the rest uniform over the span.

    The sparse remainder is not decoration: questions like "the 2nd clip" or "how many so
    far" are defined over the whole prefix. dense_frac 0.85 measured worse than 0.65.
    """
    first, last = int(round(start_time * FPS)), int(round(end_time * FPS))
    if last <= first:
        return [first, first]
    centre = int(round(center_s * FPS))
    lo = max(first, centre - int(round(half_width_s * FPS)))
    hi = min(last, centre + int(round(half_width_s * FPS)))
    n_dense = max(2, min(int(round(n_max * dense_frac)), n_max - 2))
    dense = _uniform(lo, hi, n_dense) if hi > lo else [lo] * n_dense
    return _finalise(dense + _uniform(first, last, n_max - n_dense), n_max)


def stage2_budget(start_time: float, end_time: float, n_max: int, half_width_s: float,
                  dense_frac: float, min_marker_gap_s: float) -> int:
    """Largest even frame count whose band markers stay at least `min_marker_gap_s` apart.

    Frames fuse in pairs, so `n` frames over a span give `n/2` markers. The band is fed
    by the dense half plus whatever share of the uniform half happens to fall inside it,
    which is what sets the density on short windows where the band is the whole clip.
    """
    duration = max(end_time - start_time, 0.0)
    span = min(2.0 * half_width_s, duration)
    if span <= 0 or duration <= 0:
        return 2
    share = dense_frac + (1.0 - dense_frac) * span / duration
    n = int(2.0 * span / (min_marker_gap_s * share))
    return max(2, min(n_max, n - n % 2))
