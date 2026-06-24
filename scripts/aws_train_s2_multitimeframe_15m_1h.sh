#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/zhisa/project}"
DATA_ROOT="${DATA_ROOT:-/mnt/zhisa/data/prepared/s1_15m_12m_v2}"
MACRO_DATA_ROOT="${MACRO_DATA_ROOT:-/mnt/zhisa/data/prepared/s1_1h_12m_v2}"
S1_CHECKPOINT="${S1_CHECKPOINT:-/mnt/zhisa/artifacts/s1/S1_CHAMPION_PHASE2_BEST_ACTUAL.pt}"
RUN_DIR="${RUN_DIR:-/mnt/zhisa/runs/s2_mtf_15m1h_20260624_from_s1_champion}"
PYTHON_BIN="${PYTHON_BIN:-/opt/pytorch/bin/python3}"

mkdir -p "${RUN_DIR}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export ZHISA_FAST_RENDER=1

cp configs/s2_multitimeframe_15m_1h_context.yaml "${RUN_DIR}/s2_multitimeframe_15m_1h_context.yaml"
cp "${DATA_ROOT}/manifest.json" "${RUN_DIR}/manifest_15m.json"
cp "${MACRO_DATA_ROOT}/manifest.json" "${RUN_DIR}/manifest_1h.json"

"${PYTHON_BIN}" -m zhisa.scripts.train_s2 \
  --config configs/s2_multitimeframe_15m_1h_context.yaml \
  --prepared-root "${DATA_ROOT}" \
  --macro-prepared-root "${MACRO_DATA_ROOT}" \
  --train-split train \
  --val-split val \
  --s1-checkpoint "${S1_CHECKPOINT}" \
  --checkpoint "${RUN_DIR}/s2_mtf_last.pt" \
  --best-checkpoint "${RUN_DIR}/s2_mtf_best.pt" \
  --fast-render \
  2>&1 | tee "${RUN_DIR}/console.log"

{
  echo "run_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "run_dir=${RUN_DIR}"
  echo "s1_checkpoint=${S1_CHECKPOINT}"
} > "${RUN_DIR}/COMPLETED"
