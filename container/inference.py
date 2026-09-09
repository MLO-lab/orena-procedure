"""ORena SAVE FOCUS -- joint SEGMENT + PROCEDURE submission.

Qwen3.6-27B with a LoRA SFT adapter trained on both tracks at once, merged in at load.
One question is a contiguous span of a procedure -- the whole prefix on PROCEDURE, a
clip of at most 5 minutes on SEGMENT -- delivered at 5 fps with a clock burned into
every frame. The clock stays absolute on a trimmed segment clip, so both tracks are
read the same way; `window_line(start, end)` tells the model which span it is seeing.

The frame grid is recomputed rather than shipped: `resources/sampling.py` is the same
module that chose the frames at training-export time, and it is a pure function of
`(start_time, end_time, question)` -- all three of which arrive in the Request.
`check_vendored.py` verifies this against every exported row.

Clips are read from `overlayed/`, not `plain/`: the model was trained on frames with the
clock burned in, and the system prompt tells it to read that clock.
"""

import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import torch

RESOURCES_PATH = Path(__file__).parent / "resources"
sys.path.insert(0, str(RESOURCES_PATH))

from focus import Response, load_requests, save_items  # noqa: E402

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
VIDEO_DIR = INPUT_PATH / "overlayed"
MODEL_PATH = RESOURCES_PATH / "base_model"
ADAPTER_PATH = RESOURCES_PATH / "adapter"

MAX_NEW_TOKENS = 160          # 32 truncated long `time` and multi-class answers
MAX_DECODE_WORKERS = 16
SETUP_BUDGET_S = 120.0
PER_QUESTION_BUDGET_S = 30.0          # TRACK_MAX_LATENCY[Track.PROCEDURE]
SEGMENT_QUESTION_BUDGET_S = 15.0      # TRACK_MAX_LATENCY[Track.SEGMENT]
FORFEIT_CLIFF = 1.2           # overrunning the budget by this much forfeits the batch
BUDGET_TARGET = 0.97          # stage 2 only runs while the batch still projects under this

# --- Two-stage zoom (PROCEDURE only) ----------------------------------------------
# Stage 1's `time` answer is a pointer, not an answer: on the 9B, 62.5% of heico
# predictions land within 5 min of the truth but only 18.5% within the scored 5 s.
# Stage 2 re-asks over a grid concentrated on that guess -- at 384 frames over a +-5 min
# band the fused-pair markers fall 4.4 s apart, inside the tolerance. Swept on the 9B
# (two_stage_eval.py, jobs 288862-288865): heico 0.179 -> 0.291, lapchole 0.283 -> 0.366.
# Wider bands and a denser band both scored worse.
STAGE2_FRAMES = 384
ZOOM_HALF_WIDTH_S = 300.0
ZOOM_DENSE_FRAC = 0.65
# Floor on marker spacing inside the band. Training never showed the model markers
# closer than 5.1 s and the sweep ran at 4.4-4.7 s, so the zoom is allowed down to the
# swept density and no further -- on a short window that means fewer frames, not more.
ZOOM_MIN_MARKER_GAP_S = 4.4


def usable_cpus() -> int:
    """How many CPUs this container may actually use.

    `os.cpu_count()` reports the HOST's cores -- 112 on the test cluster's nodes,
    against the 4 (L40S) or 8 (RTX PRO 6000) the platform allocates. Forking a decoder
    per host core would thrash the box it actually runs on.
    """
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 2
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.is_file():
        parts = quota.read_text().split()
        if len(parts) == 2 and parts[0] != "max":
            n = min(n, max(1, round(int(parts[0]) / int(parts[1]))))
    return max(1, min(n, MAX_DECODE_WORKERS))


def log_environment(device: torch.device) -> None:
    log.info("--- Environment ---")
    log.info("  torch          : %s (built against CUDA %s)", torch.__version__, torch.version.cuda)
    arch = torch.cuda.get_arch_list() or (torch._C._cuda_getArchFlags() or "").split()
    log.info("  torch kernels  : %s", " ".join(arch) or "unknown")
    driver_file = Path("/proc/driver/nvidia/version")
    log.info("  host driver    : %s",
             driver_file.read_text().strip().splitlines()[0]
             if driver_file.exists() else "no NVIDIA driver visible")
    log.info("  cpus           : %s usable of %s reported", usable_cpus(), os.cpu_count())
    if device.type != "cuda":
        log.warning("  No GPU visible -- running on CPU. The platform always provides one.")
        return
    capability = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info(0)
    log.info("  GPU            : %s", torch.cuda.get_device_name(0))
    log.info("  capability     : %s (sm_%d%d)", capability, *capability)
    log.info("  VRAM           : %.1f GiB free of %.1f GiB", free / 1024**3, total / 1024**3)


def log_fused_kernels() -> None:
    """Report which linear-attention path is live, and expect the torch one.

    48 of this model's 64 layers are Gated DeltaNet. transformers uses fla's fused
    kernels when fla imports and falls back silently otherwise
    (`chunk_gated_delta_rule or torch_chunk_gated_delta_rule`), so the path taken is
    invisible without this line.

    fla is deliberately not installed: its kernels are ~2x faster once compiled, but
    triton compiles per shape with nothing cached between container runs, and PROCEDURE
    frame counts vary per question -- measured 136 s against 80 s over the same batch.
    See requirements.txt. So "torch fallback" here is the intended state; the line exists
    so that a future change of mind is visible in the job log rather than only in the
    latency.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    fused = m.chunk_gated_delta_rule is not None and m.fused_recurrent_gated_delta_rule is not None
    log.info("  linear attn    : %s", "fla fused kernels" if fused else "torch fallback (expected)")


def load_model(device: torch.device):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    import vision_patch

    # The vision tower's Conv3d patch embedding degenerates at these frame counts -- 1,487 s
    # for one sample against 5 ms for the identical matmul. Must run before any forward.
    vision_patch.apply()

    log_fused_kernels()
    processor = AutoProcessor.from_pretrained(str(ADAPTER_PATH))
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), dtype=torch.bfloat16, device_map={"": 0 if device.type == "cuda" else "cpu"})

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    return processor, model.merge_and_unload().eval()


def warm_up(processor, model) -> None:
    """One tiny generation, so first-call costs land in setup rather than on question 1."""
    import numpy as np

    import model_inputs
    from prompts_segpro import build_system_prompt

    idx5 = [0, 5]
    video = np.zeros((2, 288, 512, 3), dtype=np.uint8)
    meta = model_inputs.build_metadata(idx5, 5.0, 6, (512, 288))
    messages = [{"role": "user",
                 "content": [{"type": "video"}, {"type": "text", "text": "warm up"}]}]
    text = processor.apply_chat_template(
        model_inputs.with_system(messages, build_system_prompt()),
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = model_inputs.encode(processor, text, video, meta).to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=4, do_sample=False)


def run() -> int:
    t_start = time.monotonic()
    log.info("=== ORena SAVE FOCUS -- joint SEGMENT+PROCEDURE inference start ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_environment(device)

    log.info("--- Loading inputs ---")
    requests = load_requests(INPUT_PATH / "request.json")
    if not requests:
        log.error("request.json contains no requests")
        return 1
    n = len(requests)
    # Which track this batch is: every PROCEDURE window starts at 00:00:00 and no
    # SEGMENT clip does (0 of 20,000 training rows), so the requests say it themselves.
    # It matters because the per-question allowance is half as long on SEGMENT, and the
    # skip logic below is driven by it.
    is_segment = any(getattr(r, "start_time", 0) > 0 for r in requests)
    per_question = SEGMENT_QUESTION_BUDGET_S if is_segment else PER_QUESTION_BUDGET_S
    log.info("Track looks like %s (per-question budget %.0f s)",
             "SEGMENT" if is_segment else "PROCEDURE", per_question)
    budget = SETUP_BUDGET_S + per_question * n
    hard_deadline = t_start + budget * FORFEIT_CLIFF
    log.info("Batch of %d question(s); budget %.0f s, forfeit cliff at %.0f s",
             n, budget, budget * FORFEIT_CLIFF)
    log.info("  clips: %d in %s",
             len(list(VIDEO_DIR.glob("*.mp4"))) if VIDEO_DIR.is_dir() else -1, VIDEO_DIR)

    # Forked before the model is loaded: fork copies the parent's address space, and
    # the workers only ever touch PyAV and numpy -- never CUDA.
    workers = usable_cpus()
    pool = Pool(workers)
    pool.map(int, range(workers))
    log.info("Decode pool: %d worker(s)", workers)

    log.info("--- Loading model (once for the batch) ---")
    processor, model = load_model(device)
    import model_inputs
    import zoom
    from prompts_segpro import build_system_prompt, extract_answer

    system_prompt = build_system_prompt()

    warm_up(processor, model)
    if device.type == "cuda":
        log.info("  VRAM after load+warmup: %.1f GiB allocated",
                 torch.cuda.max_memory_allocated() / 1024**3)
    log.info("Setup complete in %.1f s", time.monotonic() - t_start)

    log.info("--- Running inference over %d question(s) ---", n)
    responses, answered, failed, skipped = [], 0, 0, 0
    elapsed, s2_costs, zoomed, zoom_skipped = [], [], 0, 0

    # Stage 2 is PROCEDURE-only. A SEGMENT clip is at most 5 minutes, so a +-5 min band
    # is the whole clip and there is nothing to zoom into -- and its per-question
    # allowance is half as long, which a second pass would blow.
    two_stage = STAGE2_FRAMES > 0 and not is_segment
    log.info("Two-stage zoom: %s",
             f"ON for `time` answers ({STAGE2_FRAMES} frames, +-{ZOOM_HALF_WIDTH_S:.0f} s "
             f"band, dense_frac {ZOOM_DENSE_FRAC})" if two_stage
             else f"OFF ({'SEGMENT track' if is_segment else 'STAGE2_FRAMES=0'})")

    for i, req in enumerate(requests, start=1):
        t0 = time.monotonic()

        # Skip only when starting this question would cross the cliff AND we are not
        # already past it: once past, skipping cannot refund the time already spent,
        # so answering is strictly better than writing an empty response.
        if elapsed and t0 < hard_deadline and t0 + sum(elapsed) / len(elapsed) > hard_deadline:
            skipped += 1
            log.warning("[%d/%d] qID=%s skipped: another %.0f s would cross the %.0f s cliff",
                        i, n, req.qID, sum(elapsed) / len(elapsed), budget * FORFEIT_CLIFF)
            responses.append(Response(qID=req.qID, content="", latency=0.0))
            continue

        clip = VIDEO_DIR / f"{req.qID}.mp4"
        log.info("[%d/%d] qID=%s  window=[%.1f, %.1f] s",
                 i, n, req.qID, req.start_time, req.end_time)
        try:
            inputs, idx5, timings = model_inputs.build_inputs(
                processor, clip, req.start_time, req.end_time, req.question,
                system_prompt=system_prompt, pool=pool)
            inputs = inputs.to(model.device)
            t_gen = time.monotonic()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            raw = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            answer = extract_answer(raw)
            answered += 1
            log.info("    %d frames, %d tokens | decode %.2fs encode %.2fs generate %.2fs",
                     len(idx5), inputs["input_ids"].shape[1], timings["decode"],
                     timings["encode"], time.monotonic() - t_gen)

            # The router is the answer's own shape: a bare HH:MM:SS is a `time` answer
            # and the only kind a zoom can improve. Anything else -- a count, a class,
            # a percentage -- is returned untouched and costs nothing.
            centre = zoom.to_seconds(answer) if two_stage else None
            if centre is not None:
                s1_cost = time.monotonic() - t0
                # Would this zoom, plus every remaining question at the rate observed so
                # far, still land under budget? Projected rather than assumed, so on
                # slower hardware the batch degrades to single-pass instead of paying
                # the graded overrun penalty.
                est_s2 = (sum(s2_costs) / len(s2_costs)) if s2_costs else 0.9 * s1_cost
                per_q = max(sum(elapsed) / len(elapsed), s1_cost) if elapsed else s1_cost
                projected = (time.monotonic() - t_start) + est_s2 + (n - i) * per_q
                if projected <= budget * BUDGET_TARGET:
                    t_s2 = time.monotonic()
                    n2 = zoom.stage2_budget(req.start_time, req.end_time, STAGE2_FRAMES,
                                            ZOOM_HALF_WIDTH_S, ZOOM_DENSE_FRAC,
                                            ZOOM_MIN_MARKER_GAP_S)
                    idx2 = zoom.zoom_indices(req.start_time, req.end_time,
                                             min(max(centre, 0.0), req.end_time),
                                             n2, ZOOM_HALF_WIDTH_S, ZOOM_DENSE_FRAC)
                    inputs2, _, _ = model_inputs.build_inputs(
                        processor, clip, req.start_time, req.end_time, req.question,
                        system_prompt=system_prompt, pool=pool, frames_indices=idx2)
                    inputs2 = inputs2.to(model.device)
                    with torch.no_grad():
                        out2 = model.generate(**inputs2, max_new_tokens=MAX_NEW_TOKENS,
                                              do_sample=False)
                    raw2 = processor.tokenizer.decode(
                        out2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    answer2 = extract_answer(raw2)
                    s2_costs.append(time.monotonic() - t_s2)
                    zoomed += 1
                    # An unparseable second pass is a regression, not a refinement.
                    keep = zoom.to_seconds(answer2) is not None
                    log.info("    zoom %d frames over %.0f-%.0f s | %.2fs | %r -> %r%s",
                             len(idx2), idx2[0] / 5.0, idx2[-1] / 5.0, s2_costs[-1],
                             answer, answer2, "" if keep else "  (unparseable, kept stage 1)")
                    if keep:
                        answer = answer2
                else:
                    zoom_skipped += 1
                    log.warning("    zoom skipped: batch projects to %.0f s of the "
                                "%.0f s budget (%.0f%%)", projected, budget,
                                100 * projected / budget)
        except Exception:
            # One bad question must not cost the batch; an empty answer is scored wrong.
            failed += 1
            log.exception("[%d/%d] qID=%s failed; emitting an empty answer", i, n, req.qID)
            answer = ""

        latency = time.monotonic() - t0
        elapsed.append(latency)
        responses.append(Response(qID=req.qID, content=answer, latency=latency))
        log.info("[%d/%d] qID=%s answered in %.2f s: %r", i, n, req.qID, latency, answer)

    pool.close()
    pool.join()

    total = time.monotonic() - t_start
    log.info("Inference complete: %d answered, %d failed, %d skipped in %.1f s "
             "(%.0f%% of the %.0f s budget)",
             answered, failed, skipped, total, 100 * total / budget, budget)
    if two_stage:
        log.info("Two-stage: %d question(s) zoomed, %d skipped for budget"
                 "%s", zoomed, zoom_skipped,
                 f"; mean zoom cost {sum(s2_costs) / len(s2_costs):.2f} s" if s2_costs else "")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    save_items(responses, OUTPUT_PATH / "answer.json")
    log.info("Wrote %d response(s) to %s", len(responses), OUTPUT_PATH / "answer.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
