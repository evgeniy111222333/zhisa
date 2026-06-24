#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/zhisa/project
S1_RUN=/mnt/zhisa/runs/s1_retrain_20260623_fix_contract
S2_RUN=/mnt/zhisa/runs/s2_retrain_20260623_fixed_contract_from_s1_phase2_best
CHAMPION="$S2_RUN/S1_CHAMPION_PHASE2_BEST_ACTUAL.pt"

cd "$ROOT"
mkdir -p "$S2_RUN"

if pgrep -af 'zhisa.scripts.train_s2|aws_train_s2_12markets' >/tmp/zhisa_s2_pids 2>/dev/null; then
  echo 'existing S2 training:'
  cat /tmp/zhisa_s2_pids
  exit 2
fi

cp "$S1_RUN/phase2_best.pt" "$CHAMPION"
cp "$S1_RUN/manifest_15m.json" "$S2_RUN/s1_source_manifest_15m.json"
sha256sum "$CHAMPION" | tee "$S2_RUN/S1_CHAMPION_PHASE2_BEST_ACTUAL.sha256"

export S1_CHAMPION="$CHAMPION"
nohup bash scripts/aws_train_s2_12markets.sh "$S2_RUN" > "$S2_RUN.bootstrap.log" 2>&1 &
echo "PID=$!"
echo "RUN=$S2_RUN"
echo "CHAMPION=$CHAMPION"
sleep 8
echo '---bootstrap---'
tail -100 "$S2_RUN.bootstrap.log" || true
echo '---process---'
pgrep -af 'zhisa.scripts.train_s2|aws_train_s2_12markets' || true
