#!/usr/bin/env bash
set -euo pipefail

: "${S2B_CHECKPOINT:?Set S2B_CHECKPOINT to the selected S2b champion}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/zhisa/project}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/zhisa/venv/bin/python}"
PREPARED_ROOT="${PREPARED_ROOT:-$PROJECT_ROOT/data/prepared/s1_15m_12m_v2}"
RUN_ROOT="${RUN_ROOT:-/home/ec2-user/s4_durable/12markets_$(date -u +%Y%m%d)}"
CONTROL_RESUME="${CONTROL_RESUME:-}"
CVAR_RESUME="${CVAR_RESUME:-}"

mkdir -p "$RUN_ROOT/control" "$RUN_ROOT/cvar"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 ZHISA_FAST_RENDER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

test -x "$PYTHON_BIN"
test -f "$S2B_CHECKPOINT"
test -f "$PREPARED_ROOT/manifest.json"
test -f configs/s4_ppo_15m_12markets_control.yaml
test -f configs/s4_cvar_15m_12markets.yaml

sha256sum "$S2B_CHECKPOINT" | tee "$RUN_ROOT/s2b_input.sha256"
sha256sum \
  src/zhisa/env/trading_env.py \
  src/zhisa/env/rewards.py \
  src/zhisa/risk/guard.py \
  src/zhisa/training/s4_rl.py \
  src/zhisa/training/cvar_ppo.py \
  src/zhisa/scripts/_rl_training.py \
  src/zhisa/scripts/train_s4.py \
  src/zhisa/scripts/train_s4_cvar.py \
  > "$RUN_ROOT/code_sha256.txt"
cp configs/s4_ppo_15m_12markets_control.yaml "$RUN_ROOT/control/config.yaml"
cp configs/s4_cvar_15m_12markets.yaml "$RUN_ROOT/cvar/config.yaml"
nvidia-smi > "$RUN_ROOT/nvidia-smi.txt"
"$PYTHON_BIN" -m pip freeze > "$RUN_ROOT/pip-freeze.txt"
"$PYTHON_BIN" -m zhisa.scripts.preflight_s4 \
  --config configs/s4_cvar_15m_12markets.yaml \
  --checkpoint "$S2B_CHECKPOINT" \
  --prepared-root "$PREPARED_ROOT" | tee "$RUN_ROOT/preflight.json"

CONTROL_RESUME_ARGS=()
if [[ -n "$CONTROL_RESUME" ]]; then
  CONTROL_RESUME_ARGS=(--resume-from "$CONTROL_RESUME")
fi
CVAR_RESUME_ARGS=()
if [[ -n "$CVAR_RESUME" ]]; then
  CVAR_RESUME_ARGS=(--resume-from "$CVAR_RESUME")
fi

# Control run establishes whether the CVaR constraint adds value. It is not
# used to initialise the main run; both branches start from the same S2b model.
"$PYTHON_BIN" -m zhisa.scripts.train_s4 \
  --config configs/s4_ppo_15m_12markets_control.yaml \
  --init-checkpoint "$S2B_CHECKPOINT" \
  --prepared-root "$PREPARED_ROOT" \
  --checkpoint "$RUN_ROOT/control/s4_control_last.pt" \
  "${CONTROL_RESUME_ARGS[@]}" \
  --fast-render 2>&1 | tee "$RUN_ROOT/control/train.log"

"$PYTHON_BIN" -m zhisa.scripts.train_s4_cvar \
  --config configs/s4_cvar_15m_12markets.yaml \
  --init-checkpoint "$S2B_CHECKPOINT" \
  --prepared-root "$PREPARED_ROOT" \
  --checkpoint "$RUN_ROOT/cvar/s4_cvar_last.pt" \
  "${CVAR_RESUME_ARGS[@]}" \
  --fast-render 2>&1 | tee "$RUN_ROOT/cvar/train.log"
