#!/bin/bash
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1400G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=segpro-27b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# The same A/B at 27B, FSDP across 8 GPUs. Run this only once the 9B arms have said the
# mix is worth the compute: the joint arm is 3x the steps of the procedure-only one.
#
#   sbatch --export=ALL,ARM=joint-all segpro/train_segpro_27b_fsdp.slurm   # ship it
#   sbatch --export=ALL,ARM=joint      segpro/train_segpro_27b_fsdp.slurm   # measurable
#   sbatch --export=ALL,ARM=joint-all,EXTRA_ARGS="--vision-lora" segpro/train_segpro_27b_fsdp.slurm
#
# joint-all is the shipped procedure recipe with segment added: its 10,000 procedure
# rows are byte-identical to the all.jsonl that produced qwen3.6-27b-fsdp-n576-3ep.
#
# There is no eval arm here: this trainer holds nothing out (eval_strategy="no") and
# saves on a fixed step interval, so checkpoints are picked afterwards with
# eval_segpro.slurm. That is the shipped procedure recipe, unchanged.
set -euo pipefail

# Under sbatch, BASH_SOURCE points at Slurm's spool copy of this script (job 249617 died
# trying to mkdir logs/ in /var/spool/slurmd), so the submit directory is the anchor;
# the script's own location is the fallback when it is run directly.
if [[ -z "${EXP_ROOT:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/segpro" ]]; then
        EXP_ROOT="$SLURM_SUBMIT_DIR"
    else
        EXP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
    fi
fi
[[ -d "$EXP_ROOT/segpro" ]] || {
    echo "ERROR: no segpro/ under EXP_ROOT=$EXP_ROOT -- submit from the repo root or set EXP_ROOT=" >&2
    exit 1
}
SEGPRO="$EXP_ROOT/segpro"
cd "$EXP_ROOT"
mkdir -p logs

PY="${PY:-$EXP_ROOT/../.venv/bin/python}"
[[ -x "$PY" ]] || { echo "ERROR: no interpreter at $PY -- set PY=" >&2; exit 1; }
"$PY" -c "import importlib.util as u; assert u.find_spec('fla')" 2>/dev/null \
    || { echo "ERROR: $PY has no flash-linear-attention" >&2; exit 1; }

# *-all arms train on train+eval+test, which is what the shipped procedure 27B did:
# maximum data, nothing held out, and the leaderboard as the only signal.
ARM="${ARM:-baseline}"
case "$ARM" in
    baseline)     TRAIN_FILE="$SEGPRO/sft_export_segpro/procedure_train.jsonl" ;;
    joint)        TRAIN_FILE="$SEGPRO/sft_export_segpro/joint_train.jsonl" ;;
    procedure-all) TRAIN_FILE="$SEGPRO/sft_export_segpro/procedure_all.jsonl" ;;
    joint-all)    TRAIN_FILE="$SEGPRO/sft_export_segpro/joint_all.jsonl" ;;
    *) echo "ARM must be baseline|joint|procedure-all|joint-all, got '$ARM'" >&2; exit 1 ;;
esac

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
DATA_ROOT="${DATA_ROOT:-/mnt/vast/workspaces/VL_LeJepa/data/orena}"
RUN_NAME="${RUN_NAME:-segpro-27b-$ARM}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

[[ -s "$TRAIN_FILE" ]] || {
    echo "MISSING export: $TRAIN_FILE" >&2
    echo "Build it with:  $PY $SEGPRO/build_segpro_sft_dataset.py" >&2
    exit 1
}
for d in heico lapchole; do
    [[ -d "$DATA_ROOT/$d/frames_overlay" ]] \
        || { echo "MISSING: $DATA_ROOT/$d/frames_overlay" >&2; exit 1; }
done

export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export ACCELERATE_USE_FSDP=1

if [[ -f "$EXP_ROOT/../.env" ]]; then set -a; source "$EXP_ROOT/../.env"; set +a; fi
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    export WANDB_MODE=offline
fi

"$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_files_only=True)" 2>/dev/null \
    && export HF_HUB_OFFLINE=1 \
    || { echo "ERROR: $MODEL_ID is not in the HF cache; fetch it from a login node." >&2; exit 1; }

echo "arm: $ARM | run: $RUN_NAME | train: $TRAIN_FILE"
echo "rows: $(wc -l < "$TRAIN_FILE") | node: $(hostname) | batch: 8 GPUs x 1 x accum 4 = 32"

"${PY%/*}/torchrun" --standalone --nproc_per_node="${NGPU:-8}" \
    "$SEGPRO/sft_train_qwen27b_procedure_fsdp.py" \
    --model-id "$MODEL_ID" \
    --train-file "$TRAIN_FILE" \
    --frames-root "$DATA_ROOT" \
    --output-dir "$SEGPRO/checkpoints/$RUN_NAME" \
    --run-name "$RUN_NAME" \
    --batch-size 1 --grad-accum 4 \
    --lora-r 8 --lora-alpha 16 \
    --lr 1e-4 --seed 42 \
    --epochs 3 \
    --save-steps 50 \
    --dataloader-workers 4 \
    --resume-from-checkpoint auto \
    $EXTRA_ARGS
