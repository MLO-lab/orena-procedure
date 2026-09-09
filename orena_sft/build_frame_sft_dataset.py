"""Builds a chat-formatted SFT dataset (train/eval/test) for VLM fine-tuning
on the FOCUS frame track.

Unlike `build_probing_dataset.py`, this script does NOT read from the
`webapp/cache/qa_cache.parquet` cache — it loads the QA pairs itself, straight
from HuggingFace (or its local cache, same as `data_exploration.ipynb`), via
`focus.FocusDataset`. Only the frame track is used (single still image per
question, `duration == 0`).

Images are never copied or embedded: each record just references the
already-extracted frame JPEG on disk (under `<root_dir>/<dataset>/frames/`),
resolved with the exact same formula `FocusFrameDataset` uses internally
(`round(start_time * base_fps)`). This keeps the output at KB scale instead of
duplicating gigabytes of pixel data that's already sitting on local disk.

`eval` is carved out of `train` at the *video* level (never at the question
level, since many frame-track questions from the same video are correlated),
stratified by procedure_type so every split keeps all procedure types
represented. The official `test` split from HuggingFace is left untouched, as
the final held-out benchmark.

Pass multiple `--datasets` to merge them into one combined export (e.g. both
heico and lapchole) -- videoIDs are qualified per-dataset so the eval split
never treats same-named videos from different datasets as one video, and
each record keeps a `source_dataset` field for later filtering/analysis.

Usage:
    .venv/bin/python orena_sft/build_frame_sft_dataset.py --datasets heico lapchole
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from focus import DatasetSplit, FocusConfig, FocusDataset, Track, set_config
from focus.config import DATASET_BASE_FPS

DEFAULT_ROOT_DIR = Path("/projects/datasets_ML/orena/")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "sft_export"


# ── frame path resolution (matches FocusFrameDataset internals) ────────────

def frame_path(cfg: FocusConfig, dataset: str, base_fps: float, video_id: str, t: float) -> Path:
    video_stem = Path(video_id).stem
    frame_idx = round(t * base_fps)
    return cfg.get_dataset_dir(dataset) / cfg.FRAMES_FOLDER / video_stem / f"frame{frame_idx:07d}.jpg"


# ── loading QA data directly (not from cache) ───────────────────────────────

def load_frame_track_records(dataset: str, split: DatasetSplit, cfg: FocusConfig) -> list[dict]:
    base_fps = float(DATASET_BASE_FPS[dataset])
    ds = FocusDataset(dataset, split, Track.FRAME)

    records = []
    dropped = 0
    for req, ref in ds:
        p = frame_path(cfg, dataset, base_fps, req.videoID, req.start_time)
        if not p.exists():
            dropped += 1
            continue
        records.append({
            "qID": req.qID,
            "source_dataset": dataset,
            # videoID is only unique within one dataset; qualify it so a
            # video-level split never accidentally merges same-named videos
            # from different datasets into one bucket.
            "videoID": f"{dataset}/{req.videoID}",
            "procedure_type": req.procedure_type,
            "primary_capability": ref.primary.name,
            "secondary_capabilities": [c.name for c in ref.secondaries],
            "format": ref._format,
            "question": req.question,
            "answer": ref.answer,
            "image_path": str(p),
        })
    if dropped:
        print(f"  [{dataset}/{split.value}] dropped {dropped} rows with no frame on disk "
              f"(run FrameExtractorPreprocessor first if this is unexpectedly high).")
    return records


# ── video-level eval split, stratified by procedure_type ───────────────────

def make_eval_video_split(records: list[dict], eval_frac: float, seed: int,
                          min_videos_per_stratum: int = 1) -> set[str]:
    """Return the set of videoIDs from `records` held out for eval.

    Strata are (source_dataset, procedure_type): heico and lapchole videos carry
    very different question densities (400 vs ~80 rows), so stratifying on
    procedure alone lets one dataset dominate the eval set.
    """
    rng = np.random.RandomState(seed)
    video_to_stratum: dict[str, tuple[str, str]] = {}
    for r in records:
        video_to_stratum.setdefault(r["videoID"], (r["source_dataset"], r["procedure_type"]))

    stratum_to_videos: dict[tuple[str, str], list[str]] = {}
    for v, s in video_to_stratum.items():
        stratum_to_videos.setdefault(s, []).append(v)

    eval_videos: set[str] = set()
    for stratum, videos in sorted(stratum_to_videos.items()):
        videos = sorted(videos)  # deterministic order before shuffling
        rng.shuffle(videos)
        n_eval = max(min_videos_per_stratum, round(len(videos) * eval_frac))
        n_eval = min(n_eval, len(videos) - 1)  # never empty a stratum's train side
        eval_videos.update(videos[:n_eval])
    return eval_videos


def cap_rows_per_video(records: list[dict], max_rows: int, seed: int) -> list[dict]:
    """Subsample each video down to `max_rows`, keeping the format mix intact.

    Without this a handful of dense videos dominate the eval set: heico videos
    carry 400 rows each against lapchole's ~80, so 2 heico videos were 52% of
    eval and the effective sample size was 6.1 videos out of 11.
    """
    if max_rows <= 0:
        return records
    rng = np.random.RandomState(seed)
    by_video: dict[str, list[dict]] = {}
    for r in records:
        by_video.setdefault(r["videoID"], []).append(r)

    kept: list[dict] = []
    for video in sorted(by_video):
        rows = sorted(by_video[video], key=lambda r: str(r["qID"]))
        if len(rows) <= max_rows:
            kept += rows
            continue
        by_fmt: dict[str, list[dict]] = {}
        for r in rows:
            by_fmt.setdefault(r["format"], []).append(r)
        # Largest-remainder allocation so the capped video keeps the video's own
        # format proportions rather than whatever the shuffle happened to pick.
        exact = {f: len(rs) * max_rows / len(rows) for f, rs in by_fmt.items()}
        quota = {f: int(e) for f, e in exact.items()}
        for f in sorted(exact, key=lambda f: (-(exact[f] - quota[f]), f))[:max_rows - sum(quota.values())]:
            quota[f] += 1
        for fmt in sorted(by_fmt):
            rs = by_fmt[fmt]
            idx = rng.permutation(len(rs))[:quota[fmt]]
            kept += [rs[i] for i in sorted(idx)]
    return kept


# ── chat-format serialization ───────────────────────────────────────────────

def to_chat_record(r: dict) -> dict:
    return {
        "qID": r["qID"],
        "source_dataset": r["source_dataset"],
        "videoID": r["videoID"],
        "procedure_type": r["procedure_type"],
        "primary_capability": r["primary_capability"],
        "secondary_capabilities": r["secondary_capabilities"],
        "format": r["format"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": r["image_path"]},
                    {"type": "text", "text": r["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(r["answer"])}],
            },
        ],
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(to_chat_record(r)) + "\n")


# ── reporting + reproducibility ──────────────────────────────────────────

def _kish(counts: list[int]) -> float:
    """Effective sample size across videos: n_eff == n only when rows are even."""
    return sum(counts) ** 2 / sum(c * c for c in counts) if counts else 0.0


def report_eval_balance(eval_records: list[dict], test_records: list[dict]) -> None:
    from collections import Counter

    vids = Counter(r["videoID"] for r in eval_records)
    print(f"\n  eval concentration: {len(vids)} videos, "
          f"effective {_kish(list(vids.values())):.1f}; "
          f"largest video = {max(vids.values()) / len(eval_records):.1%} of rows")
    for label, rows in (("eval", eval_records), ("test", test_records)):
        fmt = Counter(r["format"] for r in rows)
        n = len(rows)
        print(f"  {label:5s} formats: "
              + ", ".join(f"{f} {fmt[f] / n:.1%}" for f in sorted(fmt)))
    ev_cls = {c.strip() for r in eval_records if r["format"] == "fo_class"
              for c in str(r["answer"]).split(",") if c.strip()}
    te_cls = {c.strip() for r in test_records if r["format"] == "fo_class"
              for c in str(r["answer"]).split(",") if c.strip()}
    missing = sorted(te_cls - ev_cls)
    if missing:
        print(f"  NOTE: fo_class labels in test but absent from eval: {missing}")


def write_manifest(args, train_records, eval_records, test_records) -> None:
    """Pin the split so a regenerated export can be proven identical.

    Video assignment alone is no longer enough: eval rows are subsampled, so the
    manifest also hashes the exact qID set of every split.
    """
    import hashlib

    def digest(rows) -> str:
        keys = sorted(f"{r['source_dataset']}:{r['qID']}" for r in rows)
        return hashlib.sha256("\n".join(keys).encode()).hexdigest()

    manifest = {
        "_comment": "Pinned split for the combined FRAME-track export. The JSONL exports are "
                    "gitignored; regenerate with build_frame_sft_dataset.py and check with "
                    "verify_split_manifest.py.",
        "generator": "orena_sft/build_frame_sft_dataset.py",
        "params": {
            "datasets": sorted(args.datasets),
            "eval_frac": args.eval_frac,
            "min_eval_videos_per_stratum": args.min_eval_videos_per_stratum,
            "max_eval_rows_per_video": args.max_eval_rows_per_video,
            "seed": args.seed,
            "level": "video",
            "stratified_by": "(source_dataset, procedure_type)",
            "rule": "n_eval = clip(round(n_videos * eval_frac), min_per_stratum, n_videos - 1), "
                    "then rows per eval video capped with format proportions preserved",
        },
        "splits": {
            name: {
                "n_rows": len(rows),
                "n_videos": len({r["videoID"] for r in rows}),
                "videos": sorted({r["videoID"] for r in rows}),
                "qid_sha256": digest(rows),
            }
            for name, rows in (("train", train_records), ("eval", eval_records),
                                ("test", test_records))
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


# ── orchestration ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["heico"], choices=["heico", "lapchole"],
                     help="one or more FOCUS datasets to load and merge, e.g. --datasets heico lapchole")
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR,
                     help="root dir containing <dataset>/frames/ for each dataset (passed to FocusConfig)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--eval-frac", type=float, default=0.12,
                     help="fraction of TRAIN videos (per dataset, per procedure_type) held out for eval")
    ap.add_argument("--min-eval-videos-per-stratum", type=int, default=3,
                     help="floor on eval videos per (dataset, procedure). 1 leaves heico's two "
                          "procedures with a single video each, which is what made eval unreliable.")
    ap.add_argument("--max-eval-rows-per-video", type=int, default=80,
                     help="cap eval rows per video (format mix preserved). 0 disables. Stops dense "
                          "heico videos (400 rows) from swamping lapchole ones (~80).")
    ap.add_argument("--all-data", action="store_true",
                     help="final-submission mode: put EVERY non-test row in train.jsonl (no eval "
                          "holdout) and assert no test row leaks in. Fix the step count in advance "
                          "-- there is nothing left to early-stop on.")
    ap.add_argument("--include-test", action="store_true",
                     help="with --all-data, ALSO train on the labelled public test split (20000 "
                          "rows total). Valid for a final submission scored on the platform's own "
                          "hidden set, but it destroys every local yardstick: test accuracy from "
                          "evaluate_qwen_frame.py then measures memorisation, not generalisation.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", type=Path, default=None,
                     help="write a manifest pinning the split (videos + eval qIDs + hashes) here")
    args = ap.parse_args()

    cfg = FocusConfig(root_dir=args.root_dir)
    set_config(cfg)

    train_records, test_records = [], []
    for dataset in args.datasets:
        print(f"Loading {dataset!r} frame track from HuggingFace (train + test)...")
        ds_train = load_frame_track_records(dataset, DatasetSplit.TRAIN, cfg)
        ds_test = load_frame_track_records(dataset, DatasetSplit.TEST, cfg)
        print(f"  train: {len(ds_train)} usable rows")
        print(f"  test:  {len(ds_test)} usable rows")
        train_records += ds_train
        test_records += ds_test

    if args.all_data:
        test_keys = {(r["source_dataset"], r["qID"]) for r in test_records}
        if args.include_test:
            train_records = train_records + test_records
            print(f"\nALL-DATA + TEST mode: {len(train_records)} rows "
                  f"({len(train_records) - len(test_records)} train/eval + {len(test_records)} test).")
            print("  WARNING: this checkpoint can no longer be evaluated locally -- every row of "
                  "the test set is now training data. Judge it only on the submission platform.")
        else:
            # sft_export/combined_all was built with test folded in silently, so
            # anything evaluated on it scored its own training data.
            leaked = [r for r in train_records if (r["source_dataset"], r["qID"]) in test_keys]
            if leaked:
                raise SystemExit(f"ABORT: {len(leaked)} test rows present in the train pool. "
                                 "Pass --include-test if that is intended.")
            print(f"\nALL-DATA mode: {len(train_records)} rows (train + eval, no holdout); "
                  f"verified disjoint from {len(test_records)} test rows.")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(train_records, args.out_dir / "train.jsonl")
        write_jsonl(test_records, args.out_dir / "test.jsonl")
        print(f"Wrote train.jsonl, test.jsonl to {args.out_dir}/")
        if args.manifest:
            write_manifest(args, train_records, [], test_records)
            print(f"Wrote split manifest to {args.manifest}")
        return

    eval_videos = make_eval_video_split(train_records, args.eval_frac, args.seed,
                                        args.min_eval_videos_per_stratum)
    eval_records = [r for r in train_records if r["videoID"] in eval_videos]
    final_train_records = [r for r in train_records if r["videoID"] not in eval_videos]
    eval_records = cap_rows_per_video(eval_records, args.max_eval_rows_per_video, args.seed)

    # qIDs are not unique across datasets; a row in both splits would leak.
    train_qids = {(r["source_dataset"], r["qID"]) for r in final_train_records}
    eval_records = [r for r in eval_records
                    if (r["source_dataset"], r["qID"]) not in train_qids]

    n_train_videos = len({r["videoID"] for r in final_train_records})
    print(f"\nVideo-level eval split: {len(eval_videos)} eval videos, {n_train_videos} train videos "
          f"(eval_frac={args.eval_frac}, min/stratum={args.min_eval_videos_per_stratum}, "
          f"cap={args.max_eval_rows_per_video}).")
    print(f"  final train: {len(final_train_records)} rows")
    print(f"  eval:        {len(eval_records)} rows")
    print(f"  test:        {len(test_records)} rows (untouched HF test split)")
    report_eval_balance(eval_records, test_records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(final_train_records, args.out_dir / "train.jsonl")
    write_jsonl(eval_records, args.out_dir / "eval.jsonl")
    write_jsonl(test_records, args.out_dir / "test.jsonl")
    print(f"\nWrote train.jsonl, eval.jsonl, test.jsonl to {args.out_dir}/")

    if args.manifest:
        write_manifest(args, final_train_records, eval_records, test_records)
        print(f"Wrote split manifest to {args.manifest}")


if __name__ == "__main__":
    main()
