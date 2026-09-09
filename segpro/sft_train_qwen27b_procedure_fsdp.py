"""LoRA SFT of Qwen3.6-27B for the procedure track, FSDP across 8 GPUs.

Separate from the 9B trainer because the 27B needs two things the 9B does not:

  * FSDP. 52 GB of frozen weights plus ~36 GB of checkpointed activations does not fit
    in 80 GB, and DDP replicates the weights on every rank. Full sharding puts 6.5 GB of
    weights on each GPU instead.
  * Fewer frames. Sharding only helps the weights; activations scale with sequence
    length, so the export is rebuilt at --num-frames 512 (36,864 visual tokens).

Trains on ALL data with no held-out split, so there is no eval loss and no early
stopping -- checkpoints are saved on a fixed step interval and chosen afterwards.

    sbatch procedure_track/train_qwen27b_procedure_fsdp.slurm
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from trl import SFTConfig

PROC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROC_DIR))
sys.path.insert(0, str(PROC_DIR.parent / "segment_track"))
sys.path.insert(0, str(PROC_DIR.parent / "orena_sft"))

from sft_train_qwen_segment_ddp import check_fast_kernels, pick_eval_samples  # noqa: E402

import vision_patch  # noqa: E402

from collate_procedure import build_collate_fn  # noqa: E402
from prompts_segpro import build_system_prompt  # noqa: E402
from sft_train_qwen_procedure_ddp import SampleGenerationCallback  # noqa: E402
from sparse_logits_loss import SparseLogitsSFTTrainer  # noqa: E402

vision_patch.apply()

FSDP_WRAP_CLASSES = ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"]

LM_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]

# The vision tower names its projections differently and none of these collide with a
# language-model module: the 27 blocks' attn.qkv / attn.proj / mlp.linear_fc{1,2}, plus
# the merger's two. patch_embed.proj also ends in ".proj" and is excluded on purpose --
# vision_patch replaces that module's forward with a matmul reading self.proj.weight
# directly, so a LoRA wrapper there would train parameters no forward ever uses.
VISION_TARGET_MODULES = ["qkv", "proj", "linear_fc1", "linear_fc2"]
VISION_EXCLUDE_MODULES = ["visual.patch_embed.proj"]


class SampleGenerationOnSave(SampleGenerationCallback):
    """Same probe as the 9B run, fired on save: there is no eval to hang it off."""

    def on_save(self, args, state, control, model=None, **kwargs):
        return super().on_evaluate(args, state, control, model=model, **kwargs)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-file", default=str(PROC_DIR / "sft_export_n512" / "all.jsonl"),
                    help="all.jsonl from a --num-frames 512 --include-test export")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--vision-lora", action="store_true",
                    help="adapt the vision tower too (blocks + merger): +4.0 M "
                         "trainable params, and the tower now needs a backward graph")
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--no-fsdp", action="store_true",
                    help="plain DDP; will OOM at 27B, kept for smoke tests at 9B")
    ap.add_argument("--allow-slow-kernels", action="store_true")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--save-total-limit", type=int, default=None)
    ap.add_argument("--eval-sample-count", type=int, default=4,
                    help="in-sample prefixes generated at every save, as a plumbing "
                         "check only -- every row is training data here")
    ap.add_argument("--frame-size", default="512x288",
                    help="WxH each frame is resized to; MUST match evaluation")
    ap.add_argument("--dataloader-workers", type=int, default=4)
    ap.add_argument("--frames-root", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume-from-checkpoint", default=None,
                    help="path to a checkpoint-N dir, or 'auto' for the latest")
    ap.add_argument("--wandb-project", default="orena-procedure-sft")
    ap.add_argument("--run-name", default=None)
    return ap


def main():
    args = build_parser().parse_args()

    w, h = (int(x) for x in args.frame_size.lower().split("x"))
    frame_size = (w, h)
    system_prompt = build_system_prompt(style="direct")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name or f"{args.model_id.split('/')[-1]}-procedure-all-{timestamp}"
    args.output_dir = args.output_dir or str(PROC_DIR / "checkpoints" / run_name)

    check_fast_kernels(strict=not args.allow_slow_kernels)

    if not args.no_fsdp:
        # Under FSDP, SDPA falls back to the math backend during the backward recompute
        # and materialises the whole attention matrix (~292 GB at these lengths). Deny
        # the backend: flash/mem-efficient serve the shape, or it raises instead.
        torch.backends.cuda.enable_math_sdp(False)

    processor = AutoProcessor.from_pretrained(args.model_id)
    # No device_map under FSDP: accelerate places and shards the parameters itself.
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16,
        **({} if not args.no_fsdp else {"device_map": {"": int(os.environ.get("LOCAL_RANK", 0))}}))

    from peft import LoraConfig, get_peft_model

    # Transformers only disables the KV cache when it sees TrainingArguments'
    # gradient_checkpointing; under FSDP the checkpointing lives in fsdp_config, so the
    # cache stays on and the recompute appends to it -- K/V come back twice as long.
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=LM_TARGET_MODULES + (VISION_TARGET_MODULES if args.vision_lora else []),
        exclude_modules=VISION_EXCLUDE_MODULES if args.vision_lora else None,
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={"train": args.train_file})
    probe_samples = pick_eval_samples(dataset["train"], args.eval_sample_count)

    n_frames = dataset["train"]["n_frames"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if int(os.environ.get("RANK", "0")) != 0:
        os.environ["WANDB_MODE"] = "disabled"
    wandb.init(project=args.wandb_project, name=run_name, config={
        "model_id": args.model_id,
        "track": "procedure",
        "sharding": "ddp" if args.no_fsdp else "fsdp_full_shard",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "vision_lora": args.vision_lora,
        "frame_size": args.frame_size,
        "n_frames_max": max(n_frames),
        "n_frames_mean": sum(n_frames) / len(n_frames),
        "train_file": args.train_file,
        "n_train_examples": len(dataset["train"]),
        "held_out": None,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum * world_size,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "learning_rate": args.lr,
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_at": timestamp,
    })

    checkpointing = not args.no_gradient_checkpointing
    fsdp_args = {} if args.no_fsdp else {
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {
            "transformer_layer_cls_to_wrap": FSDP_WRAP_CLASSES,
            # LoRA leaves frozen and trainable parameters in the same wrapped module;
            # FSDP refuses to flatten that mix without this.
            "use_orig_params": True,
            "sync_module_states": True,
            "limit_all_gathers": True,
            # Must be FSDP's own, not TrainingArguments.gradient_checkpointing: the
            # latter re-gathers the full shard in backward and OOMs.
            "activation_checkpointing": checkpointing,
        },
    }

    config = SFTConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=checkpointing and args.no_fsdp,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=args.dataloader_workers,
        dataloader_persistent_workers=args.dataloader_workers > 0,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="wandb",
        seed=args.seed,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        **fsdp_args,
    )

    # generate() needs unsharded weights: under FSDP every parameter is a flat 1-D
    # shard, so the probe's embedding lookup raises "'weight' must be 2-D".
    probe_callbacks = [] if not args.no_fsdp else [
        SampleGenerationOnSave(processor, probe_samples, args.output_dir,
                               system_prompt, frame_size, args.frames_root)]

    trainer = SparseLogitsSFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        data_collator=build_collate_fn(processor, system_prompt, frame_size,
                                       args.frames_root),
        processing_class=processor,
        callbacks=probe_callbacks,
    )

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
