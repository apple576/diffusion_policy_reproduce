#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. 檢查權重是否齊全，若缺少則自動執行下載腳本
if [ ! -f "$PROJECT_ROOT/checkpoints/epoch=0550-test_mean_score=0.841.ckpt" ] && \
   [ ! -f "$PROJECT_ROOT/checkpoints/pusht_vision_cnn.ckpt" ]; then
    echo "[Auto-Setup] 未檢測到模型權重，自動啟動下載程序..."
    bash "$PROJECT_ROOT/scripts/download_assets.sh"
fi

# 2. 若在宿主機執行，自動喚起 Docker 容器
if [ ! -f /.dockerenv ]; then
    # 防呆檢查：若 Docker 映像檔尚未建置，自動建置
    if ! sudo docker image inspect diff_policy:latest >/dev/null 2>&1; then
        echo "[Host] 檢測到尚未建置 diff_policy:latest 映像檔，正在自動建置 Docker 環境..."
        sudo docker build -t diff_policy:latest "$PROJECT_ROOT"
    fi

    echo "[Host] 啟動 Docker 容器執行復現評測..."
    sudo docker run --gpus all --rm \
        --user $(id -u):$(id -g) \
        --shm-size=16g \
        -e HOME=/tmp \
        -e WANDB_MODE=offline \
        -e MPLCONFIGDIR=/tmp/matplotlib \
        -v "$PROJECT_ROOT":/workspace \
        -w /workspace \
        diff_policy:latest \
        bash /workspace/scripts/run_reproduce.sh "$@"
    exit 0
fi

# ==============================================================================
# Docker 容器內核心評估流程
# ==============================================================================
export MPLCONFIGDIR=/tmp/matplotlib
CKPT_DIR="/workspace/checkpoints"
CKPT_PATH="${1:-$CKPT_DIR/epoch=0550-test_mean_score=0.841.ckpt}"

if [ ! -f "$CKPT_PATH" ] && [ -f "$CKPT_DIR/pusht_vision_cnn.ckpt" ]; then
    CKPT_PATH="$CKPT_DIR/pusht_vision_cnn.ckpt"
fi

echo "=== 執行 Push-T 視覺閉環評測 (50 回合) ==="
echo "使用權重: $CKPT_PATH"
python /workspace/scripts/eval_with_log.py "$CKPT_PATH" 42

if [ "$(id -u)" = "0" ]; then
    HOST_UID=$(stat -c '%u' /workspace)
    HOST_GID=$(stat -c '%g' /workspace)
    chown -R ${HOST_UID}:${HOST_GID} /workspace/outputs 2>/dev/null || true
    chmod -R 775 /workspace/outputs 2>/dev/null || true
fi

echo "=== 復現完成！日誌摘要 ==="
cat /workspace/outputs/eval_log_seed42.json
