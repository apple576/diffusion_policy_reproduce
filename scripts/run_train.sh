#!/bin/bash
set -e

# ==============================================================================
# 🎯 [模式設定區]：在此切換「冒煙測試 (smoke)」或「論文完整復現 (full)」
# ==============================================================================
# 預設為 smoke。也可以在執行指令時傳入參數：
#   bash scripts/run_train.sh smoke  --> 跑冒煙測試 (2 epochs, 10 steps)
#   bash scripts/run_train.sh full   --> 跑論文完整復現 (3050 epochs, 全資料集)
# ==============================================================================
MODE="${1:-smoke}"

if [ "$MODE" = "smoke" ]; then
    echo "=================================================="
    echo "⚡ 正在執行：【冒煙測試 (Smoke Test)】"
    echo "說明：僅跑 2 個 Epochs (抽樣 10 Steps)，用於驗證管線"
    echo "=================================================="
    EPOCHS=2
    VAL_EVERY=1
    EXTRA_ARGS="training.max_train_steps=10 training.max_val_steps=5"

elif [ "$MODE" = "full" ]; then
    echo "=================================================="
    echo "🔥 正在執行：【論文完整復現 (Full Paper Reproduction)】"
    echo "說明：跑滿官方標準 3050 Epochs，完整遍歷示範資料庫"
    echo "=================================================="
    EPOCHS=3050
    VAL_EVERY=50
    EXTRA_ARGS="" # 不設步數限制，跑滿所有訓練批次
else
    echo "錯誤：未知的模式 '$MODE'。請使用 'smoke' 或 'full'。"
    exit 1
fi

# 檢查是否在 Docker 容器內，若不在容器內則自動啟動 Docker 執行自身
if [ ! -f /.dockerenv ]; then
    echo "[Host] 啟動 Docker 容器並指派 RTX 4080 GPU..."
    sudo docker run --gpus all --rm \
        --user $(id -u):$(id -g) \
        --shm-size=16g \
        -e HOME=/tmp \
        -e WANDB_MODE=offline \
        -e MPLCONFIGDIR=/tmp/matplotlib \
        -v $(pwd):/workspace \
        -w /workspace/diffusion_policy \
        diff_policy:latest \
        bash /workspace/scripts/run_train.sh "$MODE"
    exit 0
fi

# ==============================================================================
# 以下為 Docker 容器內部執行的核心訓練指令
# ==============================================================================
echo "=== [1/2] 檢查 Zarr 資料集 ==="
if [ ! -d "data/pusht/pusht_cchi_v7_replay.zarr" ]; then
    echo "Error: 找不到 data/pusht/pusht_cchi_v7_replay.zarr"
    exit 1
fi

echo "=== [2/2] 啟動 Diffusion Policy 訓練並記錄算力遙測 ==="
python train_telemetry.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
    task=pusht_image \
    training.seed=42 \
    training.device=cuda:0 \
    training.num_epochs=${EPOCHS} \
    training.val_every=${VAL_EVERY} \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
    logging.mode=offline \
    ${EXTRA_ARGS}

echo "=== 訓練與遙測記錄全數完成！==="
