#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ZHISA_ROOT:-/mnt/zhisa/project}"
RUN_DIR="${1:-/mnt/zhisa/runs/s2_finetune_h4h16_20260624_from_adaptive_best}"
PYTHON="${ZHISA_PYTHON:-/opt/pytorch/bin/python3}"
WARM_START="${S2_WARM_START:-/mnt/zhisa/runs/s2_retrain_20260624_adaptive_multihorizon_from_s1_phase2_best/s2_best.pt}"

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
  echo "S2 fine-tune already completed: $RUN_DIR/COMPLETED"
  cat "$RUN_DIR/COMPLETED"
  exit 0
fi
if [[ ! -f "$WARM_START" ]]; then
  echo "S2 warm-start checkpoint not found: $WARM_START" >&2
  exit 1
fi

echo "run_started_utc=$(date -u +%FT%TZ)"
echo "root=$ROOT"
echo "run_dir=$RUN_DIR"
echo "warm_start=$WARM_START"
sha256sum "$WARM_START"
nvidia-smi
free -h
df -h / "$ROOT"
cp configs/s2_finetune_15m_12markets_h4h16.yaml "$RUN_DIR/"
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

"$PYTHON" -m zhisa.scripts.train_s2 \
  --config configs/s2_finetune_15m_12markets_h4h16.yaml \
  --prepared-root data/prepared/s1_15m_12m_v2 \
  --warm-start-checkpoint "$WARM_START" \
  --checkpoint "$RUN_DIR/s2_finetune_last.pt" \
  --best-checkpoint "$RUN_DIR/s2_finetune_best.pt" \
  --batch-size 256 --workers 4 --fast-render

"$PYTHON" - "$RUN_DIR" <<'PY' | tee "$RUN_DIR/COMPLETED"
from pathlib import Path
import datetime as dt
import sys, torch
run = Path(sys.argv[1])
last = torch.load(run / "s2_finetune_last.pt", map_location="cpu", weights_only=False)
best = torch.load(run / "s2_finetune_best.pt", map_location="cpu", weights_only=False)
state = last["trainer_state"]
best_state = best["trainer_state"]
print(f"run_completed_epochs={state['completed_epochs']}")
print(f"run_final_step={state['step']}")
print(f"run_best_val_metric={best_state['best_val_metric']}")
print(f"run_best_val_total={best_state['best_val_total']}")
print(f"run_completed_utc={dt.datetime.now(dt.timezone.utc).isoformat()}")
PY
