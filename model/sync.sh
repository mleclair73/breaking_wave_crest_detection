#!/usr/bin/env bash
# Push the runnable ablation tree or pull its outputs from the current RunPod.
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$MODEL_DIR/.." && pwd)"
REMOTE_DIR="${RUNPOD_REMOTE_DIR:-/workspace/breaking_wave_crest_detection}"
RUNPOD_TARGET="${RUNPOD_HOST:-}"
RUNPOD_PORT="${RUNPOD_PORT:-}"
SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"
STRICT_HOST_KEY_CHECKING="${RUNPOD_STRICT_HOST_KEY_CHECKING:-yes}"
KNOWN_HOSTS_FILE="${RUNPOD_KNOWN_HOSTS_FILE:-$HOME/.ssh/known_hosts}"
ACTION="${1:-pull}"

if [[ -z "$RUNPOD_TARGET" || -z "$RUNPOD_PORT" ]]; then
    cat >&2 <<'EOF'
Set the current ephemeral RunPod SSH endpoint first, for example:
  RUNPOD_HOST=root@203.0.113.10 RUNPOD_PORT=12345 bash sync.sh pull
EOF
    exit 2
fi

SSH_TRANSPORT="ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=$STRICT_HOST_KEY_CHECKING -o UserKnownHostsFile=$KNOWN_HOSTS_FILE -p $RUNPOD_PORT -i $SSH_KEY"

case "$ACTION" in
    push)
        # --relative preserves paths while avoiding the 40+ GB predictions and
        # historical outputs trees. No remote files are deleted.
        cd "$REPO_DIR"
        rsync --relative --recursive --links --times --compress --progress \
            -e "$SSH_TRANSPORT" \
            pyproject.toml requirements.txt uv.lock utils \
            model/common model/calculate_wave_speeds.py model/dunex_paths.py \
            model/data/full_split model/data/ood_test \
            model/ablation model/pretrained_weights model/segmentation \
            model/setup.sh model/train.sh model/sync.sh \
            "$RUNPOD_TARGET:$REMOTE_DIR/"
        ;;
    pull)
        mkdir -p "$MODEL_DIR/outputs"
        # Checkpoints are already compressed. Keep partial files so an
        # interrupted RunPod connection can resume instead of restarting a
        # multi-hundred-megabyte checkpoint from byte zero.
        rsync --recursive --links --times --partial --append-verify --info=progress2 \
            --exclude='checkpoint_epoch_*.pth' \
            -e "$SSH_TRANSPORT" \
            "$RUNPOD_TARGET:$REMOTE_DIR/model/outputs/" \
            "$MODEL_DIR/outputs/"
        ;;
    verify)
        ssh -o BatchMode=yes -o StrictHostKeyChecking="$STRICT_HOST_KEY_CHECKING" \
            -o UserKnownHostsFile="$KNOWN_HOSTS_FILE" \
            -p "$RUNPOD_PORT" -i "$SSH_KEY" "$RUNPOD_TARGET" \
            "'$REMOTE_DIR/model/.venv-ablation/bin/python' -m ablation.verify"
        ;;
    *)
        echo "Usage: $0 {push|pull|verify}" >&2
        exit 2
        ;;
esac
