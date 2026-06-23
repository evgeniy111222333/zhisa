#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ZHISA_ROOT:-/mnt/zhisa/project}"
RUN_DIR="${1:-/home/ec2-user/s1_durable/12m_clean_20260621}"
PYTHON="${ZHISA_PYTHON:-/opt/pytorch/bin/python3}"
PHASE1_TARGET=5
TOTAL_TARGET=20

cd "$ROOT"
mkdir -p "$RUN_DIR"

export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export ZHISA_FAST_RENDER=1
export ZHISA_SSL_WORKERS=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > >(tee -a "$RUN_DIR/console.log") 2>&1

echo "run_started_utc=$(date -u +%FT%TZ)"
echo "root=$ROOT"
echo "run_dir=$RUN_DIR"
nvidia-smi
free -h
df -h "$ROOT"
cp configs/s1_ssl_1h_12m.yaml "$RUN_DIR/"
cp configs/s1_ssl_15m_12m.yaml "$RUN_DIR/"
cp data/prepared/s1_1h_12m_v2/manifest.json "$RUN_DIR/manifest_1h.json"
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

checkpoint_epochs() {
  "$PYTHON" - "$1" <<'PY'
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint.get("trainer_state", {}).get("completed_epochs", 0)))
PY
}

run_phase1() {
  local resume_args=()
  local completed=0
  if [[ -f "$RUN_DIR/phase1_last.pt" ]]; then
    completed=$(checkpoint_epochs "$RUN_DIR/phase1_last.pt")
    resume_args=(--resume-from "$RUN_DIR/phase1_last.pt")
  fi
  local remaining=$((PHASE1_TARGET - completed))
  if (( remaining <= 0 )); then
    echo "phase1 already complete: completed_epochs=$completed"
    return
  fi
  echo "phase1 starting: completed=$completed remaining=$remaining"
  "$PYTHON" -m zhisa.scripts.train_s1 \
    --config configs/s1_ssl_1h_12m.yaml \
    --prepared-root data/prepared/s1_1h_12m_v2 \
    --epochs "$PHASE1_TARGET" --batch-size 128 --workers 4 --fast-render \
    --checkpoint "$RUN_DIR/phase1_last.pt" \
    --best-checkpoint "$RUN_DIR/phase1_best.pt" \
    "${resume_args[@]}"
}

run_phase2() {
  local resume_path reset_args completed remaining
  reset_args=()
  if [[ -f "$RUN_DIR/phase2_last.pt" ]]; then
    resume_path="$RUN_DIR/phase2_last.pt"
    completed=$(checkpoint_epochs "$resume_path")
  else
    resume_path="$RUN_DIR/phase1_last.pt"
    completed=$(checkpoint_epochs "$resume_path")
    reset_args=(--reset-best-on-resume)
  fi
  remaining=$((TOTAL_TARGET - completed))
  if (( remaining <= 0 )); then
    echo "phase2 already complete: completed_epochs=$completed"
    return
  fi
  echo "phase2 starting: completed=$completed remaining=$remaining resume=$resume_path"
  "$PYTHON" -m zhisa.scripts.train_s1 \
    --config configs/s1_ssl_15m_12m.yaml \
    --prepared-root data/prepared/s1_15m_12m_v2 \
    --epochs "$TOTAL_TARGET" --batch-size 128 --workers 4 --fast-render \
    --resume-from "$resume_path" "${reset_args[@]}" \
    --checkpoint "$RUN_DIR/phase2_last.pt" \
    --best-checkpoint "$RUN_DIR/phase2_best.pt"
}

if [[ -f "$RUN_DIR/phase2_last.pt" ]]; then
  run_phase2
else
  run_phase1
  run_phase2
fi

final_completed=$(checkpoint_epochs "$RUN_DIR/phase2_last.pt")
if (( final_completed < TOTAL_TARGET )); then
  echo "training incomplete: completed_epochs=$final_completed target=$TOTAL_TARGET" >&2
  exit 1
fi
echo "run_completed_utc=$(date -u +%FT%TZ)" | tee "$RUN_DIR/COMPLETED"
