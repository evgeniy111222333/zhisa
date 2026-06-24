#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ZHISA_ROOT:-/mnt/zhisa/project}"
RUN_DIR="${1:-/mnt/zhisa/runs/s2_retrain_20260624_adaptive_multihorizon_from_s1_phase2_best}"
PYTHON="${ZHISA_PYTHON:-/opt/pytorch/bin/python3}"
S1_SOURCE="${S1_CHAMPION:-/mnt/zhisa/checkpoints/S1_PHASE2_BEST_ACTUAL.pt}"
RUN_S1="$RUN_DIR/S1_CHAMPION_PHASE2_BEST_ACTUAL.pt"

cd "$ROOT"
if [[ ! -e data ]]; then
  ln -s /mnt/zhisa/data data
fi

mkdir -p "$RUN_DIR"
if [[ ! -f "$S1_SOURCE" ]]; then
  echo "S1 checkpoint not found: $S1_SOURCE" >&2
  exit 1
fi
cp "$S1_SOURCE" "$RUN_S1"

if pgrep -af 'zhisa.scripts.train_s2|aws_train_s2_12markets' >/dev/null; then
  echo "S2 training already appears to be running:" >&2
  pgrep -af 'zhisa.scripts.train_s2|aws_train_s2_12markets' >&2
  exit 2
fi

nohup env \
  ZHISA_ROOT="$ROOT" \
  ZHISA_PYTHON="$PYTHON" \
  S1_CHAMPION="$RUN_S1" \
  bash scripts/aws_train_s2_12markets.sh "$RUN_DIR" \
  > "$RUN_DIR/nohup.out" 2>&1 &

pid="$!"
echo "$pid" > "$RUN_DIR/pid"
sleep "${STARTUP_WAIT_SECONDS:-25}"
echo "pid=$pid"
ps -p "$pid" -o pid,etime,pcpu,pmem,cmd || true
echo "--- log tail ---"
tail -120 "$RUN_DIR/nohup.out" || true
