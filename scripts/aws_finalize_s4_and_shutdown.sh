#!/usr/bin/env bash
set -uo pipefail

: "${TARGET_SESSION:?Set TARGET_SESSION to the training tmux session}"
: "${RUN_ROOT:?Set RUN_ROOT to the durable S4 run directory}"
: "${S2B_CHECKPOINT:?Set S2B_CHECKPOINT to the source champion}"

EXPORT_ROOT="${EXPORT_ROOT:-/home/ec2-user/s4_exports}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RUN_NAME="$(basename "$RUN_ROOT")"
ARCHIVE="$EXPORT_ROOT/${RUN_NAME}.tar.gz"
WATCH_LOG="$EXPORT_ROOT/${RUN_NAME}_watcher.log"

mkdir -p "$EXPORT_ROOT"
exec >>"$WATCH_LOG" 2>&1
echo "[$(date -u --iso-8601=seconds)] watcher started for $TARGET_SESSION"

if ! tmux has-session -t "$TARGET_SESSION" 2>/dev/null; then
  echo "target session does not exist; refusing to archive or shut down"
  exit 2
fi

while tmux has-session -t "$TARGET_SESSION" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done
sleep 10

echo "[$(date -u --iso-8601=seconds)] training session ended"
mkdir -p "$RUN_ROOT"

CONTROL_CHECKPOINT="$RUN_ROOT/control/s4_control_last.pt"
CVAR_CHECKPOINT="$RUN_ROOT/cvar/s4_cvar_last.pt"
STATUS="incomplete"
if [[ -s "$CONTROL_CHECKPOINT" && -s "$CVAR_CHECKPOINT" ]]; then
  STATUS="complete"
fi

{
  echo "status=$STATUS"
  echo "finished_utc=$(date -u --iso-8601=seconds)"
  echo "run_root=$RUN_ROOT"
  echo "source_checkpoint=$S2B_CHECKPOINT"
  echo "control_checkpoint_present=$([[ -s "$CONTROL_CHECKPOINT" ]] && echo yes || echo no)"
  echo "cvar_checkpoint_present=$([[ -s "$CVAR_CHECKPOINT" ]] && echo yes || echo no)"
} > "$RUN_ROOT/final_status.txt"

find "$RUN_ROOT" -type f ! -name final_file_sha256.txt -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$RUN_ROOT/final_file_sha256.txt"
sha256sum "$S2B_CHECKPOINT" > "$RUN_ROOT/source_checkpoint_final.sha256"

TMP_ARCHIVE="$EXPORT_ROOT/.${RUN_NAME}.tar.gz.tmp-$$"
rm -f "$TMP_ARCHIVE"
tar -czf "$TMP_ARCHIVE" \
  -C "$(dirname "$RUN_ROOT")" "$(basename "$RUN_ROOT")" \
  -C "$(dirname "$S2B_CHECKPOINT")" "$(basename "$S2B_CHECKPOINT")"
mv -f "$TMP_ARCHIVE" "$ARCHIVE"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
sync

echo "[$(date -u --iso-8601=seconds)] export_status=$STATUS"
echo "archive=$ARCHIVE"
cat "$ARCHIVE.sha256"
echo "shutdown requested"
sudo shutdown -h now
