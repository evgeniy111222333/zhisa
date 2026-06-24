#!/usr/bin/env bash
set -Eeuo pipefail

cd /mnt/zhisa/project
chmod +x scripts/aws_train_s1_12m.sh
mkdir -p /mnt/zhisa/runs

RUN=/mnt/zhisa/runs/s1_retrain_20260623_fix_contract
if pgrep -af 'zhisa.scripts.train_s1|aws_train_s1_12m' >/tmp/zhisa_pids 2>/dev/null; then
  echo 'existing training:'
  cat /tmp/zhisa_pids
  exit 2
fi

nohup bash scripts/aws_train_s1_12m.sh "$RUN" > "$RUN.bootstrap.log" 2>&1 &
echo "PID=$!"
echo "RUN=$RUN"
sleep 5
echo '---bootstrap---'
tail -80 "$RUN.bootstrap.log" || true
echo '---process---'
pgrep -af 'zhisa.scripts.train_s1|aws_train_s1_12m' || true
