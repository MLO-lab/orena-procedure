"""LoRA SFT for Qwen3.6-27B on the FOCUS segment track, DDP across 2x H200.

Consumes the JSONL written by `build_segment_sft_dataset.py`. Each record is a
clip: 64 frame indices into an already-extracted JPEG directory, plus one question
and a bare answer. `collate.py` turns that into video tensors and masks everything
but the answer out of the loss.

Data-parallel via torchrun (one process per GPU); each GPU holds a full 27B+LoRA
replica and processes a data shard, gradients averaged every step. Keep the
EFFECTIVE batch at 32 (batch-size x grad-accum x n_gpu) so the convergence
estimate carried over from the frame track holds.

Usage (see the .slurm wrapper):
    torchrun --standalone --nproc_per_node=2 segment_track/sft_train_qwen_segment_ddp.py \\
        --run-name segment-27b-8r-direct --epochs 1.5
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    EarlyStoppingCallback,
    Qwen3_5ForConditionalGeneration,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

SEG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SEG_DIR))
sys.path.insert(0, str(SEG_DIR.parent / "orena_sft"))

from focus.config import TRACK_MAX_LATENCY  # noqa: E402
from focus.enums import Track  # noqa: E402

from clip_sampling import DEFAULT_FRAME_SIZE  # noqa: E402
from collate import build_collate_fn, build_generation_inputs, with_system  # noqa: E402
from prompts import build_system_prompt  # noqa: E402


class SampleGenerationCallback(TrainerCallback):
    """After every eval, generates real answers for a few fixed eval clips (one per
    answer format) and logs them locally and to wandb -- qualitative progress
    alongside the eval_loss curve. For the segment track the thing to watch is
    `time`: if predictions come back as raw seconds or as clip-relative offsets, the
    timestamp plumbing is broken and no amount of training will fix it.

    Latency is measured the same way `evaluate_qwen_segment.py` measures it, so the
    numbers here are directly comparable to the challenge's 15 s SEGMENT ceiling:
    `prep` (sample the clip, load 80 JPEGs, run the processor) + `generate`. The
    Evaluator scores anything slower than 15 s as WRONG rather than erroring, so a
    latency regression would otherwise only surface as an unexplained accuracy drop.
    """

    def __init__(self, processor, samples: list[dict], output_dir: str,
                 system_prompt: str | None, frame_size, max_new_tokens: int = 32):
        self.processor = processor
        self.samples = samples
        self.output_path = Path(output_dir) / "eval_samples.jsonl"
        self.system_prompt = system_prompt
        self.frame_size = frame_size
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # DDP: rank 0 only -- a forward-only pass DDP doesn't expect would deadlock
        # the other rank. Unwrap the DDP module to call .generate().
        if not state.is_world_process_zero:
            return
        model = getattr(model, "module", model)
        was_training = model.training
        model.eval()

        records = []
        for sample in self.samples:
            t0 = time.monotonic()
            inputs = build_generation_inputs(self.processor, sample, self.system_prompt,
                                             self.frame_size).to(model.device)
            torch.cuda.synchronize()
            prep_t = time.monotonic() - t0

            t1 = time.monotonic()
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                           do_sample=False)
            torch.cuda.synchronize()
            gen_t = time.monotonic() - t1
            total_t = prep_t + gen_t

            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            pred = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            records.append({
                "step": state.global_step,
                "uid": sample.get("uid"),
                "format": sample.get("format"),
                "duration": sample.get("duration"),
                "question": sample["messages"][0]["content"][1]["text"],
                "gt_answer": sample["messages"][1]["content"][0]["text"],
                "pred_answer": pred,
                "prep_time": round(prep_t, 3),
                "generate_time": round(gen_t, 3),
                "total_time": round(total_t, 3),
                "over_segment_budget": total_t > TRACK_MAX_LATENCY[Track.SEGMENT],
            })

        model.train(was_training)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        slowest = max(r["total_time"] for r in records)
        n_over = sum(r["over_segment_budget"] for r in records)
        print(f"[eval-sample step {state.global_step}] "
              + " | ".join(f"{r['format']}: gt={r['gt_answer']!r} pred={r['pred_answer']!r} "
                           f"({r['total_time']:.1f}s)" for r in records)
              + (f"  !! {n_over} OVER the {TRACK_MAX_LATENCY[Track.SEGMENT]:.0f}s budget"
                 if n_over else ""), flush=True)

        if wandb.run is not None:
            wandb.log({
                "eval_sample/latency_max_s": slowest,
                "eval_sample/latency_mean_s": sum(r["total_time"] for r in records) / len(records),
                "eval_sample/n_over_budget": n_over,
                **{f"eval_sample/{r['format']}_pred": r["pred_answer"] for r in records},
            }, step=state.global_step)


def check_fast_kernels(strict: bool = True) -> None:
    """Fail fast if the fused Gated DeltaNet kernels aren't usable.

    Both failure modes are silent and expensive:
      * no `fla` -> transformers falls back to a Python loop over the 48
        linear-attention layers, which measured 4x slower (261 s/step vs 66 s).
      * Triton 3.4-3.7.0 on Hopper computes WRONG gradients for
        `chunk_bwd_dqkwg` (fla issue #640). fla raises rather than corrupt the
        run, but only once the first backward pass is reached -- after the model
        has loaded. Better to catch it here.
    Any pip/uv install can silently pull Triton back to 3.7.0, because torch pins it.
    """
    from importlib.util import find_spec

    problems = []
    if find_spec("fla") is None:
        problems.append("flash-linear-attention is not installed (4x slower fallback); "
                        "use venv3.12, not .venv (Python 3.13 has no working build)")
    try:
        import triton

        parts = tuple(int(x) for x in triton.__version__.split(".")[:3])
        if parts < (3, 7, 1):
            problems.append(f"triton {triton.__version__} < 3.7.1 produces incorrect "
                            "gradients on Hopper; run: uv pip install triton==3.7.1")
    except Exception as e:                                     # noqa: BLE001
        problems.append(f"could not read the triton version: {e}")

    if problems:
        msg = "fast-kernel preflight failed:\n  - " + "\n  - ".join(problems)
        if strict:
            raise SystemExit(msg + "\n(override with --allow-slow-kernels)")
        print(f"WARNING: {msg}", flush=True)


def stratified_eval_subset(dataset, n: int, seed: int):
    """A fixed, balanced subset of the eval split for in-training early stopping.

    Two properties matter here, and plain random sampling gets neither:

    * **Video balance.** The eval split is 1360 clips from only 9 videos, and two of
      them contribute 400 clips each -- 59% of the set. A random subset inherits
      that skew, so `eval_loss` would largely track two videos.
    * **Format coverage.** `percentage` has 3 clips in the whole eval split; a
      random draw usually loses it entirely.

    Round-robining over (video, format) groups fixes both. The subset is chosen
    ONCE and reused at every eval, so there is no between-eval sampling noise --
    the curve is directly comparable step to step, which is what early stopping
    needs.
    """
    import random

    groups: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(dataset):
        groups.setdefault((row["videoID"], row["format"]), []).append(i)

    rng = random.Random(seed)
    for idxs in groups.values():
        rng.shuffle(idxs)

    keys = sorted(groups)                      # deterministic order before round-robin
    rng.shuffle(keys)
    picked: list[int] = []
    depth = 0
    while len(picked) < n:
        added = False
        for k in keys:
            if depth < len(groups[k]):
                picked.append(groups[k][depth])
                added = True
                if len(picked) == n:
                    break
        if not added:                          # every group exhausted
            break
        depth += 1
    return dataset.select(sorted(picked))


def pick_eval_samples(dataset, n: int) -> list[dict]:
    """One eval clip per answer format, `time` first.

    A single fixed sample tells you almost nothing here: `time` is the format whose
    plumbing can break silently, and latency varies with clip duration. Spreading
    the picks across formats makes both visible every eval.
    """
    by_format: dict[str, dict] = {}
    for row in dataset:
        by_format.setdefault(row["format"], row)
    ordered = sorted(by_format, key=lambda f: (f != "time", f))
    return [by_format[f] for f in ordered[:n]]


def preview_example(processor, model, sample: dict, system_prompt: str | None,
                    frame_size, max_new_tokens: int = 48) -> None:
    """Print one concrete example before training: the rendered prompt (vision
    blocks collapsed so the timestamp markers are readable), the target, and a live
    generation from the untrained adapter."""
    import re

    full_text = processor.apply_chat_template(
        with_system(sample["messages"], system_prompt), tokenize=False)
    readable = re.sub(r"(<\|vision_start\|>)(<\|video_pad\|>)+(<\|vision_end\|>)",
                      r"\1[…video…]\3", full_text)

    inputs = build_generation_inputs(processor, sample, system_prompt, frame_size).to(model.device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    model.train(was_training)
    pred = processor.tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    bar = "=" * 80
    print(f"\n{bar}\nEXAMPLE PREVIEW (before training)\n{bar}")
    print(f"--- CLIP: {sample['duration']:.0f}s, {len(sample['frames_indices'])} frames, "
          f"format={sample['format']} ---")
    print("--- RENDERED CHAT (vision blocks collapsed) ---")
    print(readable[:4000])
    print("--- GROUND-TRUTH ANSWER (training target) ---")
    print(repr(sample["messages"][1]["content"][0]["text"]))
    print("--- GENERATION FROM CURRENT (untrained-adapter) WEIGHTS ---")
    print(repr(pred))
    print(f"--- SEQUENCE LENGTH: {inputs['input_ids'].shape[1]} prompt tokens ---")
    print(f"{bar}\n", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Separate from main() so tests can assert on defaults without loading 54 GB
    of weights."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-file", default=str(SEG_DIR / "sft_export" / "train.jsonl"))
    ap.add_argument("--eval-file", default=str(SEG_DIR / "sft_export" / "eval.jsonl"))
    ap.add_argument("--output-dir", default=None,
                    help="defaults to <script-dir>/checkpoints/<run_name>")
    ap.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--epochs", type=float, default=3.0,
                    help="upper bound only -- early stopping on eval_loss is what "
                         "actually ends the run (see --early-stopping-patience)")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="cap optimizer steps (overrides --epochs when >0); use for smoke runs")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-lora", action="store_true", help="full fine-tune instead of LoRA")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--no-gradient-checkpointing", action="store_true",
                    help="27B at 64 frames needs checkpointing on a 141 GB H200; only "
                         "disable to buy back ~25% speed if the memory headroom is real")
    ap.add_argument("--allow-slow-kernels", action="store_true",
                    help="proceed even if flash-linear-attention is missing or triton is "
                         "older than 3.7.1; only for debugging -- the former is 4x slower "
                         "and the latter computes wrong gradients on Hopper")
    ap.add_argument("--ddp-find-unused-parameters", action="store_true",
                    help="set if DDP errors with 'parameters didn't receive grad'; "
                         "costs throughput, so left off until the smoke run says otherwise")
    ap.add_argument("--logging-steps", type=int, default=10,
                    help="set to 1 for a timing run: per-step entries separate the "
                         "expensive first step (allocator + cold dataloader) from steady state")
    # HF requires save_steps to be a multiple of eval_steps when
    # load_best_model_at_end is on, so a 50-step checkpoint cadence forces eval
    # every 50 steps too. A full 1360-clip eval costs ~33 min at N=80, which at
    # that cadence would be ~50% overhead -- hence --eval-subset.
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--eval-subset", type=int, default=512,
                    help="early-stopping eval on this many clips (0 = all 1360), chosen "
                         "ONCE and stratified over (video, format) -- see "
                         "stratified_eval_subset. Fixed across evals, so the eval_loss "
                         "curve carries no sampling noise. The full split is for final "
                         "evaluation via evaluate_qwen_segment.py.")
    ap.add_argument("--save-total-limit", type=int, default=None,
                    help="prune older checkpoints beyond this count; default keeps ALL. "
                         "Each is 464 MB (153 adapter + 305 optimizer), so a 3-epoch run "
                         "at 50-step saves is ~23 x 464 MB = 11 GB -- cheap enough to keep "
                         "every one for later evaluation.")
    ap.add_argument("--eval-sample-count", type=int, default=4,
                    help="clips generated and timed at every eval, one per answer format "
                         "(`time` first); written to eval_samples.jsonl with latencies")
    ap.add_argument("--early-stopping-patience", type=int, default=3,
                    help="stop when eval_loss hasn't improved for this many evals; "
                         "requires --save-steps to be a multiple of --eval-steps")
    ap.add_argument("--prompt-style", choices=["plain", "direct"], default="direct",
                    help="'direct' prepends the segment format-control prompt (answer "
                         "shape rules + timestamp convention). Evaluate with the SAME style.")
    ap.add_argument("--fo-definitions", action="store_true")
    ap.add_argument("--system-prompt-file", type=Path, default=None,
                    help="use a prompt read verbatim from this file (e.g. a GEPA-evolved "
                         "best_prompt.txt); overrides --prompt-style")
    ap.add_argument("--frame-size", default=f"{DEFAULT_FRAME_SIZE[0]}x{DEFAULT_FRAME_SIZE[1]}",
                    help="WxH each frame is resized to; MUST match evaluation")
    ap.add_argument("--dataloader-workers", type=int, default=16,
                    help="64 JPEG reads per sample over NFS are latency-bound and "
                         "parallelise well; see segment_track/plan.md §2.6")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb-project", default="orena-segment-sft")
    ap.add_argument("--run-name", default=None)
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.early_stopping_patience is not None and args.save_steps % args.eval_steps:
        ap.error("--save-steps must be a multiple of --eval-steps for early stopping "
                 "(load_best_model_at_end needs a checkpoint at every eval point).")
    if args.fo_definitions and args.prompt_style == "plain" and args.system_prompt_file is None:
        ap.error("--fo-definitions has no effect with --prompt-style plain.")

    w, h = (int(x) for x in args.frame_size.lower().split("x"))
    frame_size = (w, h)

    if args.system_prompt_file is not None:
        system_prompt = args.system_prompt_file.read_text()
    elif args.prompt_style != "plain":
        system_prompt = build_system_prompt(args.fo_definitions, style=args.prompt_style,
                                            track="segment")
    else:
        system_prompt = None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name or (
        f"{args.model_id.split('/')[-1]}-{'lora' if not args.no_lora else 'full'}"
        f"-r{args.lora_r}-{args.prompt_style}-{timestamp}")
    args.output_dir = args.output_dir or str(SEG_DIR / "checkpoints" / run_name)

    check_fast_kernels(strict=not args.allow_slow_kernels)

    processor = AutoProcessor.from_pretrained(args.model_id)
    # DDP: each process loads a FULL replica on its own GPU (torchrun sets
    # LOCAL_RANK). Falls back to "auto" for a plain single-GPU launch.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = {"": local_rank} if local_rank >= 0 else "auto"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map=device_map)

    if not args.no_lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ))
        model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={"train": args.train_file, "eval": args.eval_file})

    # Samples for the generation callback are drawn from the FULL eval split before
    # any subsetting, so they stay the same clips regardless of --eval-subset.
    eval_samples = pick_eval_samples(dataset["eval"], args.eval_sample_count)
    n_eval_full = len(dataset["eval"])
    if args.eval_subset and args.eval_subset < n_eval_full:
        dataset["eval"] = stratified_eval_subset(dataset["eval"], args.eval_subset, args.seed)

    effective_batch = args.batch_size * args.grad_accum * int(os.environ.get("WORLD_SIZE", "1"))
    if int(os.environ.get("RANK", "0")) != 0:      # DDP: only rank 0 logs to wandb
        os.environ["WANDB_MODE"] = "disabled"
    wandb.init(project=args.wandb_project, name=run_name, config={
        "model_id": args.model_id,
        "track": "segment",
        "lora": not args.no_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "prompt_style": args.prompt_style,
        "fo_definitions": args.fo_definitions,
        "system_prompt_file": str(args.system_prompt_file) if args.system_prompt_file else None,
        "frame_size": args.frame_size,
        "n_frames": len(dataset["train"][0]["frames_indices"]),
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "n_train_examples": len(dataset["train"]),
        "n_eval_examples": len(dataset["eval"]),
        "n_eval_examples_full": n_eval_full,
        "eval_subset": args.eval_subset,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": effective_batch,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "learning_rate": args.lr,
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_at": timestamp,
    })

    config = SFTConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=args.dataloader_workers,
        dataloader_persistent_workers=args.dataloader_workers > 0,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="wandb",
        seed=args.seed,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        # LoRA freezes the base + vision tower, so many params never get a grad.
        # False is correct (and faster) when it works; the smoke run is what settles
        # it -- flip with --ddp-find-unused-parameters if DDP raises "parameters
        # didn't receive grad".
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        load_best_model_at_end=args.early_stopping_patience is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    callbacks = [SampleGenerationCallback(processor, eval_samples, args.output_dir,
                                          system_prompt, frame_size)]
    if args.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=build_collate_fn(processor, system_prompt, frame_size),
        # The full processor, not just the tokenizer: Trainer auto-saves
        # processing_class at every checkpoint, and a checkpoint without
        # preprocessor_config.json silently degrades to a bare tokenizer later.
        processing_class=processor,
        callbacks=callbacks,
    )

    if int(os.environ.get("RANK", "0")) == 0:
        preview_example(processor, model, dataset["eval"][0], system_prompt, frame_size)

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if int(os.environ.get("RANK", "0")) == 0 and torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
