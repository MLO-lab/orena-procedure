# orena-procedure

Training and inference code for the **ORena SAVE FOCUS 2026 PROCEDURE track** submission:
a LoRA SFT of `Qwen/Qwen3.6-27B` trained jointly on the procedure and segment tracks
("segpro"), served through a container that re-samples frames around its own first answer
for timestamp questions.

The shipped model is **`checkpoint-1850`** (epoch 1.9728). The training run reached its
wall-clock limit 26 steps short of the epoch-2 boundary at step 1876, so this checkpoint
is the epoch-2 model.

| configuration | pre_evaluation_score |
|:--|--:|
| single pass | 0.6320 |
| **two-stage zoom** (this repo) | **0.6379** |

## Layout

```
segpro/            training code and the SLURM entry point
segment_track/     modules the trainer imports (clip_sampling, collate, kernel checks)
orena_sft/         module the dataset builder imports (make_eval_video_split)
container/         the submission image: inference.py, Dockerfile, vendored resources/
configs/           resolved training config and package freeze
```

`segment_track/` and `orena_sft/` must keep those names: the training scripts locate them
with `sys.path.insert(PROC_DIR.parent / "segment_track")`.

`container/resources/` holds copies of five training modules so the image has no
dependency on the training tree. `check_vendored.py` asserts the copies still agree with
their originals.

## Requirements

Training needs 8×H100 80 GB. Building the image needs ~52 GB of free disk for the build
context and ~46 GB for the exported archive.

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt
```

`configs/pip-freeze-train.txt` lists every package in the environment used, for an exact
rebuild.

## 1. Data

Frames come from the FOCUS datasets, decoded to
`<DATA_ROOT>/{heico,lapchole}/frames_overlay/` at 5 fps with the clock burned into each
frame. The model is trained to read that clock, so the **overlayed** frames are required —
plain frames will not reproduce these results.

Build the training export (30,000 rows: 10,000 procedure + 20,000 segment):

```bash
python segpro/build_segpro_sft_dataset.py          # writes segpro/sft_export_segpro/
```

Frame caps come from this script's own defaults, `--procedure-frames 576` and
`--segment-frames 96` — **not** from `sampling.DEFAULT_N_MAX`, which is 768 and is never
consulted here. `frames_indices` are baked into the export, so the caps are fixed at
dataset-build time; changing them requires rebuilding the export and retraining.

## 2. Train

```bash
sbatch --export=ALL,ARM=joint-all,PY=$PWD/.venv/bin/python \
       segpro/train_segpro_27b_fsdp.slurm
```

FSDP `full_shard` across 8 GPUs, roughly 48 h to step 1850. Full hyperparameters are in
`configs/train_segpro_27b_ep2.json`; the load-bearing ones:

- LoRA r=8, α=16, dropout 0.05 on `q,k,v,o,gate,up,down`
- lr 1e-4, linear decay over `max_steps=2814`, seed 42
- batch 1 × grad-accum 4 × 8 GPUs = 32 effective, `save_steps=50`
- 576 frames per question at 512×288, sampled at 5 fps

Activation checkpointing must live in `fsdp_config`, not in
`TrainingArguments.gradient_checkpointing` — the latter runs out of memory under FSDP.

Re-submitting the same command resumes from the newest checkpoint
(`--resume-from-checkpoint auto`).

## 3. Build the inference image

```bash
cd container
BASE=/path/to/Qwen3.6-27B/snapshot ./stage_weights.sh ep1.97   # links weights into resources/
python check_vendored.py
podman build -f Dockerfile -t segpro-algorithm:ep1.97 .
podman save --format docker-archive segpro-algorithm:ep1.97 | gzip > segpro-ep1.97.tar.gz
```

Run `check_vendored.py` before every build. A vendored copy that drifts from its original
changes the prompt or the frame grid without changing anything visible; the check replays
the sampler over all 30,000 exported rows, compares the prompt builders and input
assembly against the training tree, and exits non-zero on any mismatch.

For rootless Podman on an account with no `/etc/subuid` range: add
`--storage-opt overlay.ignore_chown_errors=true --cgroup-manager=cgroupfs`, keep
`--root`/`--runroot` on local disk, and pass `--user 0` to `podman run` — the image ends
with `USER user`, which rootless cannot map. This affects local runs only.

## 4. How inference works

A question is a span of a procedure delivered as a 5 fps clip. `resources/sampling.py`
recomputes the frame grid from `(start_time, end_time, question)`; it is a pure function,
so the container needs nothing the export carried.

For **timestamp questions only**, a second pass runs. The router is the shape of the
answer the model already produced:

```python
_TS = re.compile(r"^\s*(\d+):(\d\d):(\d\d)\s*$")
```

A bare `HH:MM:SS` is a `time` answer and becomes the centre of a zoom; anything else is
returned untouched and costs nothing. Router and pointer are the same computation, so
there is no separate classifier that can disagree with the model's own output.

Stage 2 re-asks the same question with the same prompt over a new grid: 65% of a
384-frame budget inside ±5 minutes of the guess, the rest uniform over the whole span.
Frames fuse in pairs, so `n` frames yield `n/2` timestamp markers; the budget is capped so
markers stay at least 4.4 s apart — inside the 5 s scoring tolerance, and no denser than
the ~5 s spacing seen during training.

Two guards:

- **Budget-aware.** Before each zoom the container projects whether spending it still
  leaves enough to finish the remaining questions within the batch allowance, and skips it
  otherwise. On slower hardware the batch degrades to single-pass instead of overrunning.
- **SEGMENT-exempt.** A segment clip is at most 5 minutes, so a ±5 min band covers the
  whole clip and there is nothing to zoom into.

## Not included

Model weights, base model, datasets and built images. Adapters and submission images are
published at `Machine-Learning-Oncology/segpro-qwen3.6-27B`.

Training used `orena-focus==0.3.4`; the container installs `0.3.5`
(`container/requirements.txt`). That package supplies the Request/Response types and the
scoring tolerance.
