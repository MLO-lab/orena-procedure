"""Do the copies in resources/ still match what the model was trained with?

The container cannot import the training tree, so five modules were copied into
resources/. A copy that silently drifts from its original is the failure this whole
submission is most exposed to, because it changes the prompt or the frame grid without
changing anything visible. Run this before every build.

    python check_vendored.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEGPRO = REPO / "segpro"                           # the tree this adapter was trained in
EXPORT = SEGPRO / "sft_export_segpro"              # built by segpro/build_segpro_sft_dataset.py
RES = Path(__file__).resolve().parent / "resources"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


def identical(name: str, original: Path) -> None:
    copy = RES / name
    check(f"{name} is byte-identical to {original.relative_to(REPO)}",
          copy.read_bytes() == original.read_bytes())


TRAINED_N_MAX = 576      # what sft_export_segpro / this adapter were built with
SHIPPED_N_MAX = 576      # back to the trained cap: stage 1 is a pointer, stage 2 spends the frames


def sampling_differs_only_in_n_max() -> None:
    """Vendored sampling.py == training sampling.py, modulo the frame cap and comments."""
    def code(path: Path) -> list[str]:
        return [ln for ln in path.read_text().splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]

    mine, theirs = code(RES / "sampling.py"), code(SEGPRO / "sampling.py")
    diff = [(a, b) for a, b in zip(theirs, mine) if a != b]
    # Zero differing lines is legal: the training tree also defaults to 768, so at
    # SHIPPED_N_MAX=768 the files agree exactly. Demanding exactly one diff would fail
    # on the safest configuration.
    check("sampling.py differs from training in nothing but DEFAULT_N_MAX",
          len(mine) == len(theirs) and len(diff) <= 1
          and all(a.startswith("DEFAULT_N_MAX = ")
                  and b == f"DEFAULT_N_MAX = {SHIPPED_N_MAX}" for a, b in diff),
          f"{len(diff)} differing line(s)")


def reproduces_the_joint_export() -> None:
    """The frame grid is baked into the export; the container must recompute it exactly.

    The export stores only the window-prefixed question, so the raw one -- which is what
    the sampler keys its dense region off -- has to be recovered by stripping the prefix.
    """
    import re
    sys.path.insert(0, str(RES))
    import sampling

    import prompts_segpro

    for track in ("procedure", "segment"):
        rows = idx_bad = text_bad = zoom_bad = denser = 0
        for line in (EXPORT / f"{track}_all.jsonl").open():
            row = json.loads(line)
            rows += 1
            text = row["messages"][0]["content"]
            if isinstance(text, list):
                text = "".join(c.get("text", "") for c in text)
            start, end = float(row["start_time"]), float(row["end_time"])

            # The window line the container will emit must be the one training used.
            # This is the check that catches a one-argument window_line on a segment
            # clip: it would claim every span starts at 00:00:00.
            line_ = prompts_segpro.window_line(start, end)
            if not text.startswith(line_ + " "):
                text_bad += 1
                continue
            question = text[len(line_) + 1:]
            # Two checks, because the shipped cap is no longer the trained cap.
            # At TRAINED_N_MAX the sampler must still reproduce the export byte for
            # byte -- that is the logic check, independent of what we ship. At
            # SHIPPED_N_MAX it may only densify: same window, never shorter, even.
            want = list(row["frames_indices"])
            if list(sampling.sample_indices_5fps(start, end, question,
                                                 n_max=TRAINED_N_MAX)) != want:
                idx_bad += 1
            got = list(sampling.sample_indices_5fps(start, end, question,
                                                    n_max=SHIPPED_N_MAX))
            if not (got[0] == want[0] and got[-1] == want[-1] and len(got) >= len(want)
                    and len(got) <= SHIPPED_N_MAX and len(got) % 2 == 0):
                zoom_bad += 1
            denser += got != want
        check(f"{track}: sampler at n_max={TRAINED_N_MAX} reproduces all {rows} rows "
              f"of {track}_all.jsonl", idx_bad == 0, f"{idx_bad} mismatches")
        check(f"{track}: n_max={SHIPPED_N_MAX} only densifies ({denser}/{rows} rows denser)",
              zoom_bad == 0, f"{zoom_bad} violations")
        check(f"{track}: window_line reproduces the trained prompt on all {rows} rows",
              text_bad == 0, f"{text_bad} mismatches")


def main() -> int:
    # sampling.py is the ONE file that legitimately diverges: this adapter was trained
    # against sft_export_n576, so the vendored copy caps at 576 where the training tree
    # defaults to 768. Assert that this is the only difference, then prove the cap by
    # replaying the sampler against every row of the export it was trained on.
    sampling_differs_only_in_n_max()
    reproduces_the_joint_export()

    identical("vision_patch.py", SEGPRO / "vision_patch.py")

    sys.path.insert(0, str(RES))
    sys.path.insert(0, str(SEGPRO))
    sys.path.insert(0, str(REPO / "segment_track"))
    sys.path.insert(0, str(REPO / "orena_sft"))

    import prompts_segpro as vendored
    import importlib.util

    spec = importlib.util.spec_from_file_location("_orig_prompts", SEGPRO / "prompts_segpro.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)

    check("build_system_prompt() matches training",
          vendored.build_system_prompt() == original.build_system_prompt())
    check("window_line() matches training on both tracks",
          all(vendored.window_line(a, b) == original.window_line(a, b)
              for a, b in ((0, 1), (0, 3600), (0, 16089),          # procedure prefixes
                           (132.5, 143.5), (2860, 3160), (16971, 17090))))  # segment clips

    # extract_answer against every raw generation from the offline eval, which is the
    # only corpus that contains what this model actually emits.
    raws, mismatched = [], 0
    for f in (SEGPRO / "checkpoints").rglob("predictions.all.jsonl"):
        for line in f.open():
            raws.append(json.loads(line).get("raw_response", ""))
    for raw in raws:
        if vendored.extract_answer(raw) != original.extract_answer(raw):
            mismatched += 1
    if raws:
        check(f"extract_answer agrees on {len(raws)} real generations",
              mismatched == 0, f"{mismatched} mismatches")
    else:
        print("[SKIP] extract_answer: no local predictions.all.jsonl to replay against")

    # The input builder, against a real exported row.
    from collate_procedure import build_generation_inputs

    import model_inputs

    row = next(json.loads(l) for l in (EXPORT / "segment_all.jsonl").open())
    # Size is passed explicitly: the two modules carry different DEFAULT_FRAME_SIZE
    # (procedure 512x288, segment 640x360) and collate_procedure always passes it.
    from sampling import DEFAULT_FRAME_SIZE

    check("build_metadata matches segment_track/clip_sampling",
          model_inputs.build_metadata(row["frames_indices"], 5.0, 100, DEFAULT_FRAME_SIZE)
          == __import__("clip_sampling").build_metadata(row["frames_indices"], 5.0, 100,
                                                        DEFAULT_FRAME_SIZE))
    check("VIDEO_TOTAL_PIXEL_BUDGET matches collate_procedure",
          model_inputs.VIDEO_TOTAL_PIXEL_BUDGET
          == __import__("collate_procedure").VIDEO_TOTAL_PIXEL_BUDGET)
    check("with_system matches segment_track/collate",
          model_inputs.with_system(row["messages"], "SYS")
          == __import__("collate").with_system(row["messages"], "SYS"))
    _ = build_generation_inputs  # imported to prove the training path still loads

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("all vendored copies agree with the training tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
