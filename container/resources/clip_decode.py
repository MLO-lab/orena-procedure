"""Pull the sampled frames out of the delivered 5 fps clip.

Seeking is serial inside one decoder, so a long procedure is split across worker
processes -- one decoder each, one region each. The pool is created once per
container run, not per question: forking it costs ~3.5 s, which is a tenth of the
per-question budget if paid every time.
"""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np

GOP_FRAMES = 25  # keyframe every 5 s at 5 fps, per the platform clip spec


def _fetch(job):
    """Seek per index, decode on when the next target is inside this GOP."""
    import av

    path, indices = job
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    stream.codec_context.thread_count = 1
    tb, rate = stream.time_base, stream.average_rate

    out, cursor, decoder = [], -1, None
    for i in indices:
        if decoder is None or i < cursor or i > cursor + GOP_FRAMES:
            container.seek(int(round(i / float(rate) / float(tb))),
                           stream=stream, backward=True, any_frame=False)
            decoder = container.decode(stream)
            cursor = -1
        got = None
        for frame in decoder:
            cursor = int(round(float(frame.pts * tb) * float(rate)))
            if cursor >= i:
                got = frame
                break
        if got is None:
            container.seek(int(round(i / float(rate) / float(tb))), stream=stream, backward=True)
            decoder = container.decode(stream)
            got = next(decoder)
            cursor = i
        out.append(got.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(out)


def frame_count(clip: Path) -> int:
    import av

    container = av.open(str(clip))
    stream = container.streams.video[0]
    n = stream.frames
    if not n:
        n = int(round(float(stream.duration * stream.time_base) * float(stream.average_rate)))
    container.close()
    return n


def read(clip: Path, local_indices: list[int], pool: Pool | None) -> np.ndarray:
    if pool is None:
        return _fetch((str(clip), local_indices))
    workers = pool._processes
    size = (len(local_indices) + workers - 1) // max(workers, 1)
    chunks = [local_indices[i:i + size] for i in range(0, len(local_indices), size)]
    return np.concatenate(pool.map(_fetch, [(str(clip), c) for c in chunks]), axis=0)
