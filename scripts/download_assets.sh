#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "📦 正在從 Hugging Face 自動下載專案資源 (apple1233/diffusion_policy_reproduce)"
echo "=========================================================="

mkdir -p checkpoints
mkdir -p diffusion_policy/data/pusht

# 1. 官方預訓練權重
OFFICIAL_CKPT="$PROJECT_ROOT/checkpoints/pusht_vision_cnn.ckpt"
if [ -f "$OFFICIAL_CKPT" ]; then
    echo "✔ [1/3] 官方權重已存在: $OFFICIAL_CKPT"
else
    echo "⬇ [1/3] 正在下載官方預訓練權重..."
    wget -q --show-progress -O "$OFFICIAL_CKPT" \
        "https://huggingface.co/apple1233/diffusion_policy_reproduce/resolve/main/pusht_vision_cnn.ckpt"
fi

# 2. 自行訓練之最佳權重 (Epoch 550, IoU 0.841)
MY_CKPT="$PROJECT_ROOT/checkpoints/epoch=0550-test_mean_score=0.841.ckpt"
if [ -f "$MY_CKPT" ]; then
    echo "✔ [2/3] 自訓最佳權重已存在: $MY_CKPT"
else
    echo "⬇ [2/3] 正在下載自訓最佳權重..."
    wget -q --show-progress -O "$MY_CKPT" \
        "https://huggingface.co/apple1233/diffusion_policy_reproduce/resolve/main/epoch=0550-test_mean_score=0.841.ckpt"
fi

# 3. Push-T 示範資料集 (Zarr)
DATASET_DIR="$PROJECT_ROOT/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
if [ -d "$DATASET_DIR" ]; then
    echo "✔ [3/3] 訓練資料集已存在: $DATASET_DIR"
else
    echo "⬇ [3/3] 正在下載 Push-T 資料集壓縮檔..."
    ZIP_PATH="$PROJECT_ROOT/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr.zip"
    wget -q --show-progress -O "$ZIP_PATH" \
        "https://huggingface.co/apple1233/diffusion_policy_reproduce/resolve/main/pusht_cchi_v7_replay.zarr.zip"
    
    echo "解壓縮資料集中..."
    unzip -q -o "$ZIP_PATH" -d "$PROJECT_ROOT/diffusion_policy/data/pusht/"
    rm -f "$ZIP_PATH"
    echo "資料集解壓縮完成！"
fi

echo "=========================================================="
echo "🎉 所有資源下載完畢並已配置至標準目錄！"
echo "=========================================================="
