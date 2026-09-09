"""Builds one SFT dataset from BOTH the SEGMENT and PROCEDURE tracks.

The two tracks cover the same 130 videos and largely the same questions; what differs is
the span each question is about. Segment windows are 1-300 s and never start at 0,
procedure windows are 309 s-4 h 56 m and always do. So the union is only well posed once
every row states its span -- see `prompts_segpro.window_line`.

Everything else is rendered in ONE convention, the procedure track's, so a joint model
sees a single input format:

  * absolute 5 fps indices from `sampling.sample_indices_5fps`, one frame per 5 s until
    the per-track cap binds. A 300 s segment clip comes back at ~60 frames, a 53 min
    procedure prefix at the 576 cap -- same density, different span.
  * `frames_overlay` (burned-in clock) at 512x288, for both tracks.

    python segpro/build_segpro_sft_dataset.py --out-dir segpro/sft_export_segpro

Writes {procedure,segment}_{train,eval,test}.jsonl plus joint_train.jsonl, which is the
only file the A/B differs in. The eval split is held out BY VIDEO and is shared across
the two tracks: the same videos appear in both, so splitting them independently would
put a segment clip of an eval video into training.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR.parent / "segment_track"))
sys.path.insert(0, str(EXP_DIR.parent / "orena_sft"))

from focus import DatasetSplit, FocusConfig, FocusDataset, Track, set_config  # noqa: E402
from focus.config import DATASET_BASE_FPS  # noqa: E402

from build_frame_sft_dataset import make_eval_video_split  # noqa: E402
from clip_sampling import frame_count, frame_dir  # noqa: E402

from collate_procedure import detect_data_root  # noqa: E402
from prompts_segpro import window_line  # noqa: E402
from sampling import describe, parse_question_timestamps, sample_indices_5fps  # noqa: E402

DEFAULT_ROOT_DIR = Path(detect_data_root() or "/projects/datasets_ML/orena/")
DEFAULT_OUT_DIR = EXP_DIR / "sft_export_segpro"

TRACKS = {"procedure": Track.PROCEDURE, "segment": Track.SEGMENT}

# Per-track frame caps. Procedure keeps the shipped 576. Segment's cap almost never
# binds -- at 5 s spacing a 300 s clip is ~60 frames -- and exists only so a malformed
# window cannot produce a huge row.
DEFAULT_N_MAX = {"procedure": 576, "segment": 96}


def load_records(track: str, dataset: str, split: DatasetSplit, root_dir: Path,
                 n_max: int, frames_folder: str) -> list[dict]:
    base_fps = float(DATASET_BASE_FPS[dataset])
    ds = FocusDataset(dataset, split, TRACKS[track])

    records, dropped = [], 0
    for req, ref in ds:
        directory = frame_dir(root_dir, dataset, req.videoID, frames_folder)
        n_avail = frame_count(directory)
        if n_avail == 0:
            dropped += 1
            continue

        # The RAW question, as on the procedure track: window_line carries timestamps of
        # its own and parsing after prepending would anchor every row at its own edges.
        idx5 = sample_indices_5fps(req.start_time, req.end_time, req.question, n_max=n_max)

        records.append({
            "track": track,
            "uid": f"{track}/{dataset}/{req.qID}",
            "qID": req.qID,
            "source_dataset": dataset,
            "videoID": f"{dataset}/{req.videoID}",
            "procedure_type": getattr(req, "procedure_type", None),
            "primary_capability": ref.primary.name,
            "secondary_capabilities": [c.name for c in ref.secondaries],
            "format": ref._format,
            "question": req.question,
            "answer": ref.answer,
            "start_time": req.start_time,
            "end_time": req.end_time,
            "duration": req.end_time - req.start_time,
            "base_fps": base_fps,
            "frame_dir": str(directory),
            "frames_indices": idx5,
            "n_frames": len(idx5),
            "n_quoted_timestamps": len(parse_question_timestamps(req.question)),
            **{f"gap_{k}": v for k, v in describe(idx5).items() if k != "n"},
        })
    if dropped:
        print(f"  [{track}/{dataset}/{split.value}] dropped {dropped} rows, no usable frames")
    return records


def to_chat_record(r: dict) -> dict:
    """One record for the trainer; pixels stay out of the JSONL."""
    out = {k: r[k] for k in (
        "track", "uid", "qID", "source_dataset", "videoID", "procedure_type",
        "primary_capability", "secondary_capabilities", "format", "start_time",
        "end_time", "duration", "base_fps", "frame_dir", "frames_indices", "n_frames",
        "n_quoted_timestamps", "gap_min_gap_s", "gap_max_gap_s", "gap_at_target",
    )}
    out["messages"] = [
        {"role": "user", "content": [
            {"type": "video"},
            {"type": "text",
             "text": f"{window_line(r['start_time'], r['end_time'])} {r['question']}"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": str(r["answer"])}]},
    ]
    return out


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(to_chat_record(r)) + "\n")


def summarise(name: str, recs: list[dict]) -> None:
    if not recs:
        return
    n = [r["n_frames"] for r in recs]
    fmt = collections.Counter(r["format"] for r in recs)
    tr = collections.Counter(r["track"] for r in recs)
    print(f"  {name:18s} {len(recs):6d} rows | frames med {statistics.median(n):5.0f} "
          f"max {max(n):4d} sum {sum(n)/1e6:5.2f}M | "
          + " ".join(f"{k}:{v}" for k, v in tr.most_common()) + " | "
          + " ".join(f"{k}:{v}" for k, v in fmt.most_common(3)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["heico", "lapchole"],
                    choices=["heico", "lapchole"])
    ap.add_argument("--tracks", nargs="+", default=["procedure", "segment"],
                    choices=["procedure", "segment"])
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--procedure-frames", type=int, default=DEFAULT_N_MAX["procedure"],
                    help="frame cap for procedure rows (even); shorter windows use fewer")
    ap.add_argument("--segment-frames", type=int, default=DEFAULT_N_MAX["segment"],
                    help="frame cap for segment rows (even); rarely binds at 5 s spacing")
    ap.add_argument("--frames-folder", default="frames_overlay")
    ap.add_argument("--eval-frac", type=float, default=0.10,
                    help="fraction of TRAIN videos held out for eval, shared by both tracks")
    ap.add_argument("--include-test", action="store_true",
                    help="also write {procedure,segment,joint}_all.jsonl = train + eval + "
                         "test. The public test split ships with labels and the platform "
                         "scores against its own hidden set, so those rows are ordinary "
                         "training data -- but nothing trained on them can be evaluated "
                         "locally, and the leaderboard becomes the only signal. This is "
                         "what the shipped procedure 27B was trained on. A flag, never a "
                         "shell pipeline.")
    ap.add_argument("--segment-frac", type=float, default=1.0,
                    help="fraction of segment TRAIN rows to put in joint_train.jsonl; "
                         "the mix ratio is a knob, not a finding")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    caps = {"procedure": args.procedure_frames, "segment": args.segment_frames}
    for track, cap in caps.items():
        if cap % 2:
            ap.error(f"--{track}-frames must be even, got {cap}")

    set_config(FocusConfig(root_dir=args.root_dir))

    train, test = [], []
    for track in args.tracks:
        for dataset in args.datasets:
            print(f"Loading {dataset!r} {track} track (cap {caps[track]} frames)...")
            tr = load_records(track, dataset, DatasetSplit.TRAIN, args.root_dir,
                              caps[track], args.frames_folder)
            te = load_records(track, dataset, DatasetSplit.TEST, args.root_dir,
                              caps[track], args.frames_folder)
            print(f"  train {len(tr):6d}   test {len(te):6d}")
            train += tr
            test += te

    # One split for both tracks, computed on the union: the same 130 videos carry both
    # kinds of question, so an independent split would leak an eval video's clips into
    # training and quietly inflate every number that follows.
    eval_videos = make_eval_video_split(train, args.eval_frac, args.seed)
    eval_rows = [r for r in train if r["videoID"] in eval_videos]
    train_rows = [r for r in train if r["videoID"] not in eval_videos]
    print(f"\nShared video-level split: {len(eval_videos)} eval videos, "
          f"{len({r['videoID'] for r in train_rows})} train videos")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[tuple[str, list[dict]]] = []
    for track in args.tracks:
        for name, rows in (("train", train_rows), ("eval", eval_rows), ("test", test)):
            outputs.append((f"{track}_{name}", [r for r in rows if r["track"] == track]))

    rng = random.Random(args.seed)
    joint = [r for r in train_rows if r["track"] != "segment"]
    seg_rows = [r for r in train_rows if r["track"] == "segment"]
    if args.segment_frac < 1.0:
        seg_rows = rng.sample(seg_rows, int(round(len(seg_rows) * args.segment_frac)))
    joint += seg_rows
    rng.shuffle(joint)
    outputs.append(("joint_train", joint))

    print()
    for name, rows in outputs:
        write_jsonl(rows, args.out_dir / f"{name}.jsonl")
        summarise(name, rows)

    if args.include_test:
        every = train_rows + eval_rows + test
        for track in args.tracks:
            rows = [r for r in every if r["track"] == track]
            write_jsonl(rows, args.out_dir / f"{track}_all.jsonl")
            summarise(f"{track}_all", rows)
        joint_all = list(every)
        rng.shuffle(joint_all)
        write_jsonl(joint_all, args.out_dir / "joint_all.jsonl")
        summarise("joint_all", joint_all)
        outputs.append(("joint_all", joint_all))

    print(f"\nWrote {len(outputs)} files to {args.out_dir}/")
    print("A/B:  --train-file procedure_train.jsonl   (baseline)")
    print("      --train-file joint_train.jsonl       (treatment)")
    print("Score both on procedure_eval.jsonl and segment_eval.jsonl.")


if __name__ == "__main__":
    main()
