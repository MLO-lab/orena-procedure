"""LoRA SFT for the FOCUS procedure track, DDP across N GPUs.

Consumes the JSONL written by `build_procedure_sft_dataset.py`. Each record is a
*prefix* of an operation -- up to 4 h 56 m -- represented as up to `--num-frames` JPEGs
addressed on the 5 fps grid, plus one question and a bare answer.

The DDP scaffolding, the fast-kernel gate and the eval-subset logic are imported from
the segment trainer rather than copied; only the collator, the prompt and the defaults
differ. Sequence length is VARIABLE here (the capped-5s policy gives short windows fewer
frames), so peak memory is set by the longest window in a batch, not by every batch.

Usage (see the .slurm wrapper):
    torchrun --standalone --nproc_per_node=8 \\
        segpro/sft_train_qwen_procedure_ddp.py --run-name procedure-9b-n768
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
from trl import SFTConfig

PROC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROC_DIR))
sys.path.insert(0, str(PROC_DIR.parent / "segment_track"))
sys.path.insert(0, str(PROC_DIR.parent / "orena_sft"))

from focus.config import TRACK_MAX_LATENCY  # noqa: E402
from focus.enums import Track  # noqa: E402

from sft_train_qwen_segment_ddp import (  # noqa: E402
    check_fast_kernels, pick_eval_samples, stratified_eval_subset,
)

import vision_patch  # noqa: E402

from collate_procedure import build_collate_fn, build_generation_inputs, with_system  # noqa: E402
from prompts_segpro import build_system_prompt  # noqa: E402
from sampling import DEFAULT_FRAME_SIZE, DEFAULT_N_MAX  # noqa: E402
from sparse_logits_loss import SparseLogitsSFTTrainer  # noqa: E402

vision_patch.apply()

BUDGET = TRACK_MAX_LATENCY[Track.PROCEDURE]


class SampleGenerationCallback(TrainerCallback):
    """Generates and times real answers for a few fixed eval prefixes after every eval.

    The thing to watch is `time`: if a prediction comes back as raw seconds, or as an
    offset from something quoted in the question rather than from the start of the
    operation, the timestamp plumbing is broken and no amount of training fixes it.
    Latency is measured the way `evaluate_procedure.py` measures it, so the numbers are
    comparable to the 30 s PROCEDURE ceiling.
    """

    def __init__(self, processor, samples, output_dir, system_prompt, frame_size,
                 frames_root: str | None = None, max_new_tokens: int = 160):
        self.processor = processor
        self.samples = samples
        self.output_path = Path(output_dir) / "eval_samples.jsonl"
        self.system_prompt = system_prompt
        self.frame_size = frame_size
        self.frames_root = frames_root
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # DDP: rank 0 only -- a forward-only pass the other rank does not expect
        # would deadlock it.
        if not state.is_world_process_zero:
            return
        model = getattr(model, "module", model)
        was_training = model.training
        model.eval()

        records = []
        for sample in self.samples:
            t0 = time.monotonic()
            inputs = build_generation_inputs(self.processor, sample, self.system_prompt,
                                             self.frame_size,
                                             self.frames_root).to(model.device)
            torch.cuda.synchronize()
            prep_t = time.monotonic() - t0

            t1 = time.monotonic()
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                           do_sample=False)
            torch.cuda.synchronize()
            gen_t = time.monotonic() - t1
            total_t = prep_t + gen_t

            pred = self.processor.tokenizer.decode(
                generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            records.append({
                "step": state.global_step,
                "uid": sample.get("uid"),
                "format": sample.get("format"),
                "duration_min": round(sample.get("duration", 0) / 60, 1),
                "n_frames": sample.get("n_frames"),
                "question": sample["messages"][0]["content"][1]["text"],
                "gt_answer": sample["messages"][1]["content"][0]["text"],
                "pred_answer": pred,
                "prep_time": round(prep_t, 3),
                "generate_time": round(gen_t, 3),
                "total_time": round(total_t, 3),
                "over_budget": total_t > BUDGET,
            })

        model.train(was_training)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        n_over = sum(r["over_budget"] for r in records)
        print(f"[eval-sample step {state.global_step}] "
              + " | ".join(f"{r['format']}: gt={r['gt_answer']!r} pred={r['pred_answer']!r} "
                           f"({r['duration_min']:.0f}min, {r['n_frames']}f, {r['total_time']:.1f}s)"
                           for r in records)
              + (f"  !! {n_over} OVER the {BUDGET:.0f}s budget" if n_over else ""), flush=True)

        if wandb.run is not None:
            wandb.log({
                "eval_sample/latency_max_s": max(r["total_time"] for r in records),
                "eval_sample/latency_mean_s": sum(r["total_time"] for r in records) / len(records),
                "eval_sample/n_over_budget": n_over,
                **{f"eval_sample/{r['format']}_pred": r["pred_answer"] for r in records},
            }, step=state.global_step)


def preview_example(processor, model, sample, system_prompt, frame_size,
                    frames_root: str | None = None, max_new_tokens: int = 48) -> None:
    """One concrete example before training: the rendered prompt with vision blocks
    collapsed so the timestamp markers are readable, the target, and a live generation
    from the untrained adapter."""
    import re

    full_text = processor.apply_chat_template(
        with_system(sample["messages"], system_prompt), tokenize=False)
    readable = re.sub(r"(<\|vision_start\|>)(<\|video_pad\|>)+(<\|vision_end\|>)",
                      r"\1[…video…]\3", full_text)

    inputs = build_generation_inputs(processor, sample, system_prompt, frame_size,
                                     frames_root).to(model.device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    model.train(was_training)
    pred = processor.tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    bar = "=" * 80
    print(f"\n{bar}\nEXAMPLE PREVIEW (before training)\n{bar}")
    print(f"--- PREFIX: {sample['duration']/60:.0f} min, {sample['n_frames']} frames, "
          f"format={sample['format']}, gaps <= {sample.get('gap_max_gap_s', 0):.0f}s ---")
    print("--- RENDERED CHAT (vision blocks collapsed) ---")
    print(readable[:4000])
    print("--- GROUND-TRUTH ANSWER (training target) ---")
    print(repr(sample["messages"][1]["content"][0]["text"]))
    print("--- GENERATION FROM CURRENT (untrained-adapter) WEIGHTS ---")
    print(repr(pred))
    print(f"--- SEQUENCE LENGTH: {inputs['input_ids'].shape[1]} prompt tokens ---")
    print(f"{bar}\n", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Separate from main() so tests can assert on defaults without loading weights."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-file", default=str(PROC_DIR / "sft_export" / "train.jsonl"))
    ap.add_argument("--eval-file", default=str(PROC_DIR / "sft_export" / "eval.jsonl"))
    ap.add_argument("--output-dir", default=None,
                    help="defaults to <script-dir>/checkpoints/<run_name>")
    # 9B, not the 27B: 96 GB fits either at inference, but at equal training cost the
    # 9B affords N_max=768 against the 27B's 256 -- 3x the temporal resolution on the
    # one axis this track is hard along.
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--epochs", type=float, default=2.0,
                    help="upper bound; early stopping or --max-steps ends the run")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="cap optimizer steps (overrides --epochs when >0); smoke runs "
                         "and the final all-data run both use this")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-lora", action="store_true", help="full fine-tune instead of LoRA")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--no-gradient-checkpointing", action="store_true",
                    help="required at these sequence lengths; only disable to buy back "
                         "speed if the memory headroom is real")
    ap.add_argument("--allow-slow-kernels", action="store_true",
                    help="proceed without flash-linear-attention or with triton < 3.7.1; "
                         "the former is 4x slower, the latter computes wrong gradients "
                         "on Hopper")
    ap.add_argument("--ddp-find-unused-parameters", action="store_true")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--eval-subset", type=int, default=128,
                    help="early-stopping eval on this many prefixes (0 = all), chosen "
                         "ONCE and stratified over (video, format) so the eval_loss "
                         "curve carries no sampling noise. Smaller than the segment "
                         "track's because a prefix costs up to 10x more to forward.")
    ap.add_argument("--save-total-limit", type=int, default=None,
                    help="prune older checkpoints beyond this count; default keeps all")
    ap.add_argument("--eval-sample-count", type=int, default=4,
                    help="prefixes generated and timed at every eval, one per answer "
                         "format (`time` first); written to eval_samples.jsonl")
    ap.add_argument("--early-stopping-patience", type=int, default=3,
                    help="stop when eval_loss has not improved for this many evals; "
                         "pass a large value to disable it for the all-data run, where "
                         "eval.jsonl is IN-SAMPLE and the curve measures fit, not "
                         "generalisation")
    ap.add_argument("--fo-definitions", action="store_true")
    ap.add_argument("--system-prompt-file", type=Path, default=None,
                    help="use a prompt read verbatim from this file; overrides the built-in")
    ap.add_argument("--frame-size", default=f"{DEFAULT_FRAME_SIZE[0]}x{DEFAULT_FRAME_SIZE[1]}",
                    help="WxH each frame is resized to; MUST match evaluation")
    ap.add_argument("--dataloader-workers", type=int, default=16,
                    help="up to 768 JPEG reads per sample over NFS are latency-bound "
                         "and parallelise well")
    ap.add_argument("--frames-root", default=None,
                    help="re-root the exported frame_dir paths onto this filesystem, "
                         "for running an export built on another cluster "
                         "(the directory holding <dataset>/frames_overlay/)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume-from-checkpoint", default=None,
                    help="path to a checkpoint-N dir, or 'auto' for the latest in "
                         "--output-dir; restores optimizer, scheduler and step count")
    ap.add_argument("--wandb-project", default="orena-procedure-sft")
    ap.add_argument("--run-name", default=None)
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.early_stopping_patience is not None and args.save_steps % args.eval_steps:
        ap.error("--save-steps must be a multiple of --eval-steps for early stopping "
                 "(load_best_model_at_end needs a checkpoint at every eval point).")

    w, h = (int(x) for x in args.frame_size.lower().split("x"))
    frame_size = (w, h)

    if args.system_prompt_file is not None:
        system_prompt = args.system_prompt_file.read_text()
    else:
        system_prompt = build_system_prompt(args.fo_definitions, style="direct")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name or (
        f"{args.model_id.split('/')[-1]}-procedure-"
        f"{'lora' if not args.no_lora else 'full'}-r{args.lora_r}-{timestamp}")
    args.output_dir = args.output_dir or str(PROC_DIR / "checkpoints" / run_name)

    check_fast_kernels(strict=not args.allow_slow_kernels)

    processor = AutoProcessor.from_pretrained(args.model_id)
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

    eval_samples = pick_eval_samples(dataset["eval"], args.eval_sample_count)
    n_eval_full = len(dataset["eval"])
    if args.eval_subset and args.eval_subset < n_eval_full:
        dataset["eval"] = stratified_eval_subset(dataset["eval"], args.eval_subset, args.seed)

    n_frames = [r for r in dataset["train"]["n_frames"]]
    effective_batch = args.batch_size * args.grad_accum * int(os.environ.get("WORLD_SIZE", "1"))
    if int(os.environ.get("RANK", "0")) != 0:      # DDP: only rank 0 logs to wandb
        os.environ["WANDB_MODE"] = "disabled"
    wandb.init(project=args.wandb_project, name=run_name, config={
        "model_id": args.model_id,
        "track": "procedure",
        "lora": not args.no_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "fo_definitions": args.fo_definitions,
        "system_prompt_file": str(args.system_prompt_file) if args.system_prompt_file else None,
        "frame_size": args.frame_size,
        "n_frames_max": max(n_frames),
        "n_frames_mean": sum(n_frames) / len(n_frames),
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "n_train_examples": len(dataset["train"]),
        "n_eval_examples": len(dataset["eval"]),
        "n_eval_examples_full": n_eval_full,
        "eval_subset": args.eval_subset,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
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
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        load_best_model_at_end=args.early_stopping_patience is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    callbacks = [SampleGenerationCallback(processor, eval_samples, args.output_dir,
                                          system_prompt, frame_size, args.frames_root)]
    if args.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = SparseLogitsSFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=build_collate_fn(processor, system_prompt, frame_size,
                                       args.frames_root),
        # The full processor, not just the tokenizer: a checkpoint without
        # preprocessor_config.json silently degrades to a bare tokenizer later.
        processing_class=processor,
        callbacks=callbacks,
    )

    if int(os.environ.get("RANK", "0")) == 0:
        preview_example(processor, model, dataset["eval"][0], system_prompt, frame_size,
                        frames_root=args.frames_root)

    resume = args.resume_from_checkpoint
    if resume == "auto":
        ckpts = sorted(Path(args.output_dir).glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        resume = str(ckpts[-1]) if ckpts else None
        print(f"resuming from {resume or 'scratch (no checkpoint found)'}", flush=True)

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if int(os.environ.get("RANK", "0")) == 0 and torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
