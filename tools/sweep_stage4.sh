#!/usr/bin/env bash
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-$HOME/miniconda3/envs/.venv/bin/python}"
EPISODES="${1:-10000}"
CKPT="agent_code/attackontensor_ql/checkpoints"

# Selected stage-3 table per arm and seed, from
# results/s3-<arm>-s<seed>/task3-hunt/checkpoint_selection.json
declare -A SOURCE=(
  [hard-0]="s3-hard-s0/q_table_r004950.pkl"
  [hard-1]="s3-hard-s1/q_table_r004350.pkl"
  [hard-2]="s3-hard-s2/q_table_r002250.pkl"
  [soft-0]="s3-soft-s0/q_table_r002850.pkl"
  [soft-1]="s3-soft-s1/q_table_r005100.pkl"
  [soft-2]="s3-soft-s2/q_table_r005250.pkl"
)

launch() {
  local arm="$1" seed="$2"
  local run_id="s4-${arm}-s${seed}"
  local work="$CKPT/$run_id"

  # `warm` is the hard lineage with a different exploration schedule.
  local mode="$arm" lineage="$arm" extra=()
  if [[ "$arm" == "warm" ]]; then
    mode="hard"; lineage="hard"; extra=(--epsilon-start 0.3)
  fi

  local source_table="$CKPT/${SOURCE[${lineage}-${seed}]}"

  if [[ ! -f "$source_table" ]]; then
    echo "MISSING stage-3 table for ${arm} seed ${seed}: $source_table" >&2
    return 1
  fi

  mkdir -p "$work" "results/$run_id"
  cp "$source_table" "$work/q_table.pkl"

  {
    echo "=== $run_id | stage 4 classic vs rule_based x3 | seed $seed | arm $arm | safety $mode ==="
    date
    echo "resumed from $source_table (md5 $(md5sum "$source_table" | cut -d' ' -f1))"
  } > "results/$run_id/driver.log"

  nohup "$PY" tools/train_ql.py \
      --stage 4 --resume --episodes "$EPISODES" \
      --safety-mode "$mode" "${extra[@]+"${extra[@]}"}" \
      --seed "$seed" --run-id "$run_id" \
      --checkpoint-every 250 \
      >> "results/$run_id/driver.log" 2>&1 &

  echo "launched $run_id (pid $!)"
}

ARMS="${ARMS:-hard warm}"
for seed in 0 1 2; do
  for arm in $ARMS; do
    launch "$arm" "$seed"
  done
done

echo
echo "runs launched for arms [$ARMS], $EPISODES episodes each."
echo "Watch:   tail -f results/s4-*/driver.log"
echo "Done?:   grep -c 'gate:' results/s4-*/driver.log"
