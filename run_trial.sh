#!/usr/bin/env bash
# Train the same Diffusion Policy on different subsets, then evaluate each in simulation.
# Usage: [REPO_ID=...] [DATA_ROOT=...] [OUT=...] [SEED=...] ./run_trial.sh STEPS SUBSET [SUBSET ...]
set -euo pipefail

STEPS="${1:-30000}"
shift || true
if [ "$#" -gt 0 ]; then SUBSETS=("$@"); else SUBSETS=(curated random full); fi

REPO_ID="${REPO_ID:-lerobot/pusht}"
DATA_ROOT="${DATA_ROOT:-}"
OUT="${OUT:-trial_outputs}"
SEED="${SEED:-1000}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"

ROOT_ARG=()
[ -n "$DATA_ROOT" ] && ROOT_ARG=(--dataset.root="$DATA_ROOT")

for NAME in "${SUBSETS[@]}"; do
  EPS=$(python -c "import json; print(json.load(open('$OUT/subsets.json'))['$NAME'])" | tr -d ' ')
  TAG="${NAME}_${STEPS}"
  [ "$SEED" != "1000" ] && TAG="${TAG}_s${SEED}"
  RUN_DIR="$OUT/train_${TAG}"
  echo "=== Training '$NAME' ($STEPS steps, seed $SEED) ==="
  if [ ! -d "$RUN_DIR/checkpoints/last" ]; then
    lerobot-train \
      --policy.type=diffusion \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --dataset.repo_id="$REPO_ID" \
      "${ROOT_ARG[@]}" \
      --dataset.episodes="$EPS" \
      --env.type=pusht \
      --steps="$STEPS" \
      --batch_size=64 \
      --env_eval_freq=0 \
      --save_freq="$STEPS" \
      --seed="$SEED" \
      --wandb.enable=false \
      --output_dir="$RUN_DIR"
  else
    echo "Found existing checkpoint, skipping training."
  fi

  echo "=== Evaluating '$NAME' on $EVAL_EPISODES simulated episodes ==="
  lerobot-eval \
    --policy.path="$RUN_DIR/checkpoints/last/pretrained_model" \
    --policy.device=cuda \
    --env.type=pusht \
    --eval.n_episodes="$EVAL_EPISODES" \
    --eval.batch_size=50 \
    --eval.use_async_envs=false \
    --seed=12345 \
    --output_dir="$OUT/eval_${TAG}"
done

python compare_results.py --out_dir "$OUT"
