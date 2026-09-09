# orena-procedure

Training and inference code for our **ORena SAVE FOCUS 2026 PROCEDURE track** submission:
a LoRA SFT of `Qwen/Qwen3.6-27B` trained jointly on the procedure and segment tracks
("segpro"), served through a two-stage container that re-samples frames around its own
first answer for timestamp questions.

The shipped model is **`checkpoint-1850`, epoch 1.9728**. The run hit its 48-hour wall
26 steps short of the epoch-2 boundary at step 1876, so this *is* the epoch-2 model.

| submission | pre_evaluation_score |
|:--|--:|
| segpro, single pass | 0.6320 |
| **segpro, two-stage zoom** (this repo) | **0.6379** |

## Layout

```
segpro/            training code and the SLURM entry point
segment_track/     modules the trainer imports (clip_sampling, collate, kernel checks)
orena_sft/         module the dataset builder imports (make_eval_video_split)
container/         the submission image: inference.py, Dockerfile, vendored resources/
configs/           resolved training config + full package freeze
```

`segment_track/` and `orena_sft/` keep those exact names because the training scripts
locate them with `sys.path.insert(PROC_DIR.parent / "segment_track")`. Every file is a
byte-for-byte copy of what produced the result; only `container/check_vendored.py` and
`container/stage_weights.sh` had path constants rebased for this layout.

## 1. Environment

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt
```

`configs/pip-freeze-train.txt` is the complete 207-package list of the venv that trained
the checkpoint, if you need an exact rebuild.

## 2. Data

Frames come from the FOCUS datasets, decoded to `<DATA_ROOT>/{heico,lapchole}/frames_overlay/`
at 5 fps with the clock burned in. The model reads that clock, so the **overlayed** frames
are required — plain frames will not reproduce these numbers.

Build the training export (30,000 rows: 10,000 procedure + 20,000 segment):

```bash
python segpro/build_segpro_sft_dataset.py          # writes segpro/sft_export_segpro/
```

The frame caps come from this script's own defaults — `--procedure-frames 576` and
`--segment-frames 96` — **not** from `sampling.DEFAULT_N_MAX`, which is 768 in the
training tree and is never consulted here. `frames_indices` are baked into the export, so
these caps are fixed at dataset-build time; changing them means rebuilding the export and
retraining. Do not "fix" the 768 to match.

## 3. Train

```bash
sbatch --export=ALL,ARM=joint-all,PY=$PWD/.venv/bin/python \
       segpro/train_segpro_27b_fsdp.slurm
```

8×H100, FSDP `full_shard`, ~48 h to step 1850. Resolved hyperparameters are in
`configs/train_segpro_27b_ep2.json`; the ones that matter:

- LoRA r=8, α=16, dropout 0.05 on `q,k,v,o,gate,up,down`
- lr 1e-4, linear decay over `max_steps=2814`, **seed 42**
- batch 1 × grad-accum 4 × 8 GPUs = **32 effective**, `save_steps=50`
- 576 frames per question at 512×288, sampled at 5 fps

Activation checkpointing lives in `fsdp_config`, **not** `TrainingArguments.gradient_checkpointing`
— the latter OOMs under FSDP.

To resume (the original run was resumed after its wall-clock timeout):

```bash
sbatch --export=ALL,ARM=joint-all segpro/train_segpro_27b_fsdp.slurm   # --resume-from-checkpoint auto
```

## 4. Build the inference image

```bash
cd container
BASE=/path/to/Qwen3.6-27B/snapshot ./stage_weights.sh ep1.97   # hard-links weights into resources/
python check_vendored.py                                       # integrity gate, see below
podman build -f Dockerfile -t segpro-algorithm:ep1.97 .
podman save --format docker-archive segpro-algorithm:ep1.97 | gzip > segpro-ep1.97.tar.gz
```

Verified end to end on 2026-09-09: `stage_weights.sh` -> `check_vendored.py` (14 checks,
all pass) -> `podman build` (25 layers) -> `Successfully tagged`.

Two notes for rootless Podman without a `/etc/subuid` range for your account:

- add `--storage-opt overlay.ignore_chown_errors=true --cgroup-manager=cgroupfs`, and put
  `--root`/`--runroot` on local disk — the build context is ~52 GB
- the image ends with `USER user`, so `podman run` needs `--user 0` or it fails with
  `crun: cannot setresgid to '999'`. This affects running the image locally only; the
  evaluation platform maps users properly.

`check_vendored.py` is worth running before every build. `container/resources/` holds
copies of five training modules, and a copy that silently drifts from its original changes
the prompt or the frame grid without changing anything visible. The check replays the
sampler over all 30,000 exported rows and asserts the container reproduces the training
grid exactly, then compares the prompt builders and input assembly against the training
tree. It exits non-zero on any mismatch.

## 5. How inference works

One question is a span of a procedure, delivered as a 5 fps clip. `resources/sampling.py`
recomputes the frame grid from `(start_time, end_time, question)` — a pure function, so
the container needs nothing the export carried.

For **timestamp questions only**, a second pass runs. The router is the shape of the
answer the model already gave:

```python
_TS = re.compile(r"^\s*(\d+):(\d\d):(\d\d)\s*$")
```

A bare `HH:MM:SS` is a `time` answer and becomes the centre of a zoom; anything else is
returned untouched and costs nothing. Measured on 3,127 real generations: 97.2% of `time`
questions route to the zoom, and 1 of 1,959 non-time questions does.

Stage 2 re-asks the same question with the same prompt over a regrid — 65% of a 384-frame
budget inside ±5 minutes of the guess, the rest uniform over the whole span. Frames fuse
in pairs, so `n` frames give `n/2` timestamp markers; the budget is capped so those
markers stay ≥4.4 s apart, which is inside the 5 s scoring tolerance without going below
the 5.1 s minimum spacing the model ever saw in training.

Two guards:

- **Budget-aware.** Before each zoom the container projects whether spending it still
  leaves enough to finish every remaining question under the batch allowance, and skips
  it otherwise. On slower hardware the batch degrades to single-pass rather than paying
  the graded overrun penalty. Measured 79.4% of budget, 0 forfeited, 0 unanswered.
- **SEGMENT-exempt.** A segment clip is at most 5 minutes, so a ±5 min band is the whole
  clip and there is nothing to zoom into.

Because the router only fires on timestamp answers, the two-stage and single-pass runs
returned **bit-identical scores in 8 of the 10 evaluation buckets**; only the two
temporal-grounding buckets moved (+6 and +6 questions).

## Not included

Model weights, base model, datasets and built images. Adapters and submission images are
at `Machine-Learning-Oncology/segpro-qwen3.6-27B`.

## Note on `orena-focus`

Training ran against `orena-focus==0.3.4`; the container installs `0.3.5`
(`container/requirements.txt`). That package supplies the Request/Response types and the
scoring tolerance. The mismatch is recorded here rather than silently normalised.
