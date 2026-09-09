#!/usr/bin/env bash
# Put the weights into resources/, and select which epoch this image will carry.
#
#     ./stage_weights.sh ep3        # base + checkpoint-939 as resources/adapter/
#
# Everything is HARDLINKED, so the ~52 GB of base weights exist once on disk no matter
# how many times this runs. The three adapters are staged side by side under
# resources/adapters/ (excluded from the build context) and the selected one is
# hardlinked to resources/adapter/, which is what the Dockerfile copies.

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)

RUN="${RUN:-$REPO/segpro/checkpoints/segpro-27b-joint-all}"
# Point BASE at a local snapshot of Qwen/Qwen3.6-27B:
#   hf download Qwen/Qwen3.6-27B
BASE="${BASE:?set BASE to a local Qwen/Qwen3.6-27B snapshot directory}"

# The joint run is 2,814 steps over 3 epochs (938 steps/epoch) on 30,000 rows, saved
# every 50, so no save lands exactly on an epoch boundary. These are the nearest.
# ep1.4 and ep1.97 are not epoch boundaries. ep1.4 was the newest save while the run
# was still training; ep1.97 (checkpoint-1850) is where job 250469 hit its 48 h wall,
# 26 steps short of the epoch-2 boundary at 1876. ep2/ep3 exist only if the run is
# resumed.
# ep2.45 is the newest save from the resumed run (job 290989), taken while it is still
# training toward 3 epochs. ep3 is the final save, which lands in the run ROOT rather
# than a checkpoint dir -- max_steps 2814 is not a multiple of save_steps 50, so
# checkpoint-2814 never exists; the last checkpoint dir is 2800.
declare -A EPOCHS=([ep1]=checkpoint-950 [ep1.4]=checkpoint-1300
                   [ep1.97]=checkpoint-1850
                   [ep2]=checkpoint-1900 [ep2.45]=checkpoint-2300
                   [ep2.98]=checkpoint-2800 [ep3]=.)
declare -A FRACTION=([ep1]=1.01 [ep1.4]=1.39 [ep1.97]=1.97 [ep2]=2.03
                     [ep2.45]=2.45 [ep2.98]=2.99 [ep3]=3.00)

want="${1:-}"
[[ -n "${EPOCHS[$want]:-}" ]] || {
    echo "usage: $0 {$(printf '%s\n' "${!EPOCHS[@]}" | sort | paste -sd'|')}" >&2; exit 1; }

link_dir() {  # link_dir <src> <dst> <find-args...>
    local src="$1" dst="$2"; shift 2
    mkdir -p "$dst"
    find "$src" -maxdepth 1 -type f -o -maxdepth 1 -type l | while read -r f; do
        ln -f "$(readlink -f "$f")" "$dst/$(basename "$f")"
    done
}

echo "=+= base weights -> resources/base_model/"
mkdir -p "$SCRIPT_DIR/resources/base_model"
for f in "$BASE"/*; do
    ln -f "$(readlink -f "$f")" "$SCRIPT_DIR/resources/base_model/$(basename "$f")"
done

for ep in $(printf '%s\n' "${!EPOCHS[@]}" | sort); do
    ck="$RUN/${EPOCHS[$ep]}"
    # The run may still be training: stage whatever exists and fail only if the epoch
    # actually asked for is the missing one.
    if [[ ! -f "$ck/adapter_model.safetensors" ]]; then
        [[ "$ep" == "$want" ]] && { echo "MISSING $ck -- not written yet" >&2; exit 1; }
        echo "=+= ${ep} (${EPOCHS[$ep]}) not written yet -- skipping"
        continue
    fi
    echo "=+= ${ep} (${EPOCHS[$ep]}, epoch ${FRACTION[$ep]}) -> resources/adapters/$ep/"
    mkdir -p "$SCRIPT_DIR/resources/adapters/$ep"
    for f in "$ck"/adapter_config.json "$ck"/adapter_model.safetensors "$ck"/chat_template.jinja \
             "$ck"/processor_config.json "$ck"/tokenizer_config.json "$ck"/tokenizer.json; do
        ln -f "$f" "$SCRIPT_DIR/resources/adapters/$ep/$(basename "$f")"
    done
done

echo "=+= selecting $want -> resources/adapter/"
rm -rf "$SCRIPT_DIR/resources/adapter"
mkdir -p "$SCRIPT_DIR/resources/adapter"
for f in "$SCRIPT_DIR/resources/adapters/$want"/*; do
    ln -f "$f" "$SCRIPT_DIR/resources/adapter/$(basename "$f")"
done
echo "$want ${EPOCHS[$want]} epoch=${FRACTION[$want]}" > "$SCRIPT_DIR/resources/adapter/EPOCH"

du -sh "$SCRIPT_DIR/resources/base_model" "$SCRIPT_DIR/resources/adapter"
echo "=+= staged. Build context is now $(du -sh --exclude=adapters "$SCRIPT_DIR" | cut -f1)"
