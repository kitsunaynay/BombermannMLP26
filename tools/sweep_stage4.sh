#!/usr/bin/env bash
# Stage-4 sweep: the agent against three rule_based_agents. This is the matchup
# the tournament is, and the one the project has never trained on.
#
# Two arms x three seeds, the arms differing only in the safety mode. The mask
# comparison was measured at stage 3 with n=3 seeds x 30 arenas and could not
# resolve the 0.70-point difference between the arms (PIPELINE_REVIEW D1);
# stage 4 evaluates on 60 arenas, which is the powered version of that question.
#
# Each arm resumes from its OWN lineage's selected stage-3 table. A table
# trained behind `hard` and replayed under `soft` suicides in every round
# (DEVLOG Phase 5), so cross-seeding the arms would not compare mask modes, it
# would compare matched against mismatched training. The cost is that the arms
# differ in initialisation as well as in mask, which the writeup must say.
#
# run.json records each run's flags, derived seeds, git sha and the md5 of the
# table it resumed from, so the pairing is verifiable afterwards.
#
# Usage:  tools/sweep_stage4.sh [episodes]        (default 10000)
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
  local mode="$1" seed="$2"
  local run_id="s4-${mode}-s${seed}"
  local work="$CKPT/$run_id"
  local source_table="$CKPT/${SOURCE[${mode}-${seed}]}"

  if [[ ! -f "$source_table" ]]; then
    echo "MISSING stage-3 table for ${mode} seed ${seed}: $source_table" >&2
    return 1
  fi

  mkdir -p "$work" "results/$run_id"
  cp "$source_table" "$work/q_table.pkl"

  {
    echo "=== $run_id | stage 4 classic vs rule_based x3 | seed $seed | safety $mode ==="
    date
    echo "resumed from $source_table (md5 $(md5sum "$source_table" | cut -d' ' -f1))"
  } > "results/$run_id/driver.log"

  nohup "$PY" tools/train_ql.py \
      --stage 4 --resume --episodes "$EPISODES" \
      --safety-mode "$mode" \
      --seed "$seed" --run-id "$run_id" \
      --checkpoint-every 250 \
      >> "results/$run_id/driver.log" 2>&1 &

  echo "launched $run_id (pid $!)"
}

for seed in 0 1 2; do
  launch hard "$seed"
  launch soft "$seed"
done

echo
echo "6 runs launched, $EPISODES episodes each."
echo "Watch:   tail -f results/s4-*/driver.log"
echo "Done?:   grep -c 'gate:' results/s4-*/driver.log"
