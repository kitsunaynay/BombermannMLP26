#!/usr/bin/env bash
# tools/sweep_ppo_curriculum.sh
# SEEDS='0 1' ARMS=hard tools/sweep_ppo_curriculum.sh
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-$HOME/miniconda3/envs/.venv/bin/python}"
WORKERS="${WORKERS:-16}"
SEEDS="${SEEDS:-0 1 2}"
SELECT_SEEDS="${SELECT_SEEDS:-40}"

# Steps per stage. Stage 2 is cheap (no opponents); stage 4 pays ~1 ms of Python
# per opponent per step, so it is by far the longest.
STEPS_S2="${STEPS_S2:-2000000}"
STEPS_S3="${STEPS_S3:-5000000}"
STEPS_S4="${STEPS_S4:-10000000}"

run_stage () {           # seed stage steps device model_file run_id extra...
  local seed="$1" stage="$2" steps="$3" device="$4" model="$5" run_id="$6"; shift 6
  mkdir -p "results/$run_id"
  { echo "=== $run_id | stage $stage | seed $seed | safety hard | $steps steps ==="; date; } \
      > "results/$run_id/driver.log"
  AOT_PPO_MODEL_FILE="$model" "$PY" tools/train_ppo.py \
      --stage "$stage" --safety-mode hard \
      --workers "$WORKERS" --device "$device" --seed "$seed" \
      --total-steps "$steps" --run-id "$run_id" \
      --select-seeds "$SELECT_SEEDS" --promote --checkpoint-every 50 \
      "$@" >> "results/$run_id/driver.log" 2>&1
}

# One background chain per seed: stage 2, then 3 resuming from it, then 4.
chain () {
  local seed="$1" device="$2"
  # Under checkpoints/, not the agent directory: AOT_PPO_MODEL_FILE resolves
  # relative to AGENT_DIR, and three 19 MB chain files sitting next to
  # policy.pt push the submission archive to 77 MB and trip the 50 MB guard in
  # package_submission.py. checkpoints/ is excluded from the archive and is a
  # symlink to scratch, so the chain state stays off the home disk too.
  local model="checkpoints/ppo-chain-s${seed}.pt"
  run_stage "$seed" 2 "$STEPS_S2" "$device" "$model" "ppo-s2-hard-s${seed}" \
    || { echo "seed $seed failed at stage 2" >&2; return 1; }
  run_stage "$seed" 3 "$STEPS_S3" "$device" "$model" "ppo-s3-hard-s${seed}" --resume \
    || { echo "seed $seed failed at stage 3" >&2; return 1; }
  run_stage "$seed" 4 "$STEPS_S4" "$device" "$model" "ppo-s4-hard-s${seed}" --resume \
    || { echo "seed $seed failed at stage 4" >&2; return 1; }
  echo "seed $seed finished the curriculum"
}

device=0
for seed in $SEEDS; do
  chain "$seed" "cuda:${device}" &
  echo "launched chain seed $seed on cuda:${device} (pid $!)"
  device=$(( (device + 1) % 8 ))
done

# Control: stage 2 without the filter, same budget, same seeds.
for seed in $SEEDS; do
  run_id="ppo-s2-soft-s${seed}"
  mkdir -p "results/$run_id"
  { echo "=== $run_id | stage 2 | seed $seed | safety soft (CONTROL) | $STEPS_S2 steps ==="; date; } \
      > "results/$run_id/driver.log"
  AOT_PPO_MODEL_FILE="checkpoints/ppo-control-s${seed}.pt" nohup "$PY" tools/train_ppo.py \
      --stage 2 --safety-mode soft \
      --workers "$WORKERS" --device "cuda:${device}" --seed "$seed" \
      --total-steps "$STEPS_S2" --run-id "$run_id" \
      --select-seeds "$SELECT_SEEDS" --checkpoint-every 50 \
      >> "results/$run_id/driver.log" 2>&1 &
  echo "launched control seed $seed on cuda:${device} (pid $!)"
  device=$(( (device + 1) % 8 ))
done

echo
echo "Watch:    tail -f results/ppo-s*/driver.log"
echo "Progress: grep -h 'it ' results/ppo-s2-hard-s0/driver.log | tail -1"
