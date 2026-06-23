#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ZHISA_ROOT:-/mnt/zhisa/project}"
RUN_DIR="${1:-/home/ec2-user/s2_durable/12markets_20260621}"
PYTHON="${ZHISA_PYTHON:-/opt/pytorch/bin/python3}"
CHAMPION="${S1_CHAMPION:-$RUN_DIR/s1_champion_epoch14.pt}"

cd "$ROOT"
mkdir -p "$RUN_DIR"

export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export ZHISA_FAST_RENDER=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > >(tee -a "$RUN_DIR/console.log") 2>&1

if [[ -f "$RUN_DIR/COMPLETED" ]]; then
  echo "S2 already completed: $RUN_DIR/COMPLETED"
  cat "$RUN_DIR/COMPLETED"
  exit 0
fi
if [[ ! -f "$CHAMPION" ]]; then
  echo "S1 champion not found: $CHAMPION" >&2
  exit 1
fi

echo "run_started_utc=$(date -u +%FT%TZ)"
echo "root=$ROOT"
echo "run_dir=$RUN_DIR"
echo "champion=$CHAMPION"
sha256sum "$CHAMPION"
nvidia-smi
free -h
df -h / "$ROOT"
cp configs/s2_supervised_15m_12markets.yaml "$RUN_DIR/"
cp data/prepared/s1_15m_12m_v2/manifest.json "$RUN_DIR/manifest_15m.json"

metrics="$RUN_DIR/system_metrics.csv"
if [[ ! -f "$metrics" ]]; then
  echo "timestamp_utc,gpu_util_pct,gpu_mem_mib,gpu_temp_c,gpu_power_w,ram_available_kib,load_1m" > "$metrics"
fi
parent_pid=$$
monitor_system() {
  while kill -0 "$parent_pid" 2>/dev/null; do
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits | tr -d ' ')
    ram=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    load=$(awk '{print $1}' /proc/loadavg)
    echo "$(date -u +%FT%TZ),$gpu,$ram,$load" >> "$metrics"
    sleep 30
  done
}
monitor_system &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

resume_args=()
if [[ -f "$RUN_DIR/s2_last.pt" ]]; then
  resume_args=(--resume-from "$RUN_DIR/s2_last.pt")
  echo "resuming_from=$RUN_DIR/s2_last.pt"
fi

"$PYTHON" -m zhisa.scripts.train_s2 \
  --config configs/s2_supervised_15m_12markets.yaml \
  --prepared-root data/prepared/s1_15m_12m_v2 \
  --s1-checkpoint "$CHAMPION" \
  --checkpoint "$RUN_DIR/s2_last.pt" \
  --best-checkpoint "$RUN_DIR/s2_best.pt" \
  --batch-size 256 --workers 4 --fast-render \
  "${resume_args[@]}"

"$PYTHON" - "$RUN_DIR" <<'PY' | tee "$RUN_DIR/COMPLETED"
from pathlib import Path
import sys, torch
run = Path(sys.argv[1])
last = torch.load(run / "s2_last.pt", map_location="cpu", weights_only=False)
best = torch.load(run / "s2_best.pt", map_location="cpu", weights_only=False)
state = last["trainer_state"]
print(f"run_completed_epochs={state['completed_epochs']}")
print(f"run_final_step={state['step']}")
print(f"run_best_val_total={best['trainer_state']['best_val_total']}")
print(f"run_completed_utc={__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}")
PY
