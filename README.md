# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (Push-T Reproduction)

[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](Dockerfile)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-green.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.x_CUDA12.1-red.svg)](https://pytorch.org/)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Checkpoints-yellow.svg)](https://huggingface.co/apple1233/diffusion_policy_reproduce)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

本專案為 **Intelligent Manufacturing Systems** 課程之頂級期刊論文復現成果：
* **論文**：*Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (IJRR 2024 / RSS 2023)
* **復現任務**：**Push-T (Vision-based)** 閉環視覺操作基準測試
* **架構**：1D-CNN (U-Net Denoising Backbone) + ResNet-18 Vision Encoder (FiLM conditioning)
* **核心成果**：在單張 NVIDIA GeForce RTX 4080 (16GB) 上完成完整訓練與驗證，推論評測取得 **0.8199 Mean IoU**（達到論文報告之 0.82～0.84 基準水準），並嚴格記錄完整的顯存峰值（Peak VRAM）與運算時間（Wall-clock time）遙測日誌。

---

## 📋 目錄
1. [硬體環境與遙測規範](#-硬體環境與遙測規範)
2. [快速開始：從零完整復現](#-快速開始從零完整復現)
3. [資源自動下載 (Dataset & Checkpoints)](#-資源自動下載-dataset--checkpoints)
4. [模型訓練流程 (Training Pipeline)](#-模型訓練流程-training-pipeline)
5. [模型評估與自訂驗證 (Inference & Evaluation)](#-模型評估與自訂驗證-inference--evaluation)
6. [實驗結果與論文基準對比](#-實驗結果與論文基準對比)
7. [專案目錄結構](#-專案目錄結構)

---

## 💻 硬體環境與遙測規範

所有訓練與評測均在純淨容器化環境中執行，硬體遙測規格如下：
* **GPU**: NVIDIA GeForce RTX 4080 (16GB VRAM, Ada Lovelace, Compute Capability 8.9)
* **Host OS**: Ubuntu 22.04 LTS (x86_64)
* **Driver / CUDA**: NVIDIA Driver >= 535 / CUDA 12.1
* **Container**: Docker 24.x + NVIDIA Container Toolkit

---

## ⚡ 快速開始：從零完整復現

在全新機器或評分環境中，請依序執行以下 3 個步驟進行全自動重現：

### 步驟 1：Clone 專案倉庫
```bash
git clone https://github.com/apple576/diffusion_policy_reproduce.git
cd diffusion_policy_reproduce
```

### 步驟 2：賦予腳本權限並建置 Docker 映像檔
本環境針對 RTX 4080 (Ada Lovelace, sm_89) 進行了 CUDA 12.1 與 PyTorch 2.x 相容性配置，並鎖定無頭渲染與 NumPy ABI：
```bash
chmod +x scripts/*.sh
docker build -t diff_policy:latest .
```
> **💡 建置排錯提示（連線逾時處理）：**
> * 安裝大型相依套件時，若發現終端機畫面停在某一行（如 `[ 8/10] RUN pip install...`）超過 **3～5 分鐘進度完全凍結**，通常為 PyPI 或網路節點偶發性的 Socket 斷流假死。
> * **處理方式**：直接按下 **`Ctrl + C`** 強制中斷，並再次執行 `docker build -t diff_policy:latest .`。
> * **快取機制保護**：Docker 會自動命中前面已經完成的步驟（顯示為 `CACHED`），直接從剛才卡住的步驟接續下載，不會重複浪費時間重載前面的系統層或 PyTorch。



### 步驟 3：一鍵執行推論評測 (One-Command Reproduce)

直接執行評測腳本。若本地尚未存在模型權重或 Push-T 資料集，腳本會**全自動從 Hugging Face 下載並配置完成**，無須手動重複下載：

```bash
bash scripts/run_reproduce.sh
```

---

## 📊 評測產出位置：

### 1. 算力遙測與評估數值日誌 (JSON Log)
* **相對路徑**：`outputs/`
* **內容包含**：推論顯存峰值（Peak VRAM）、50 回合總耗時（Wall-clock Time）、測試成功率（Mean IoU）與使用的硬體規格：

### 2. 閉環推論模擬影片與動圖 (Rollout Videos)
評測過程中的視覺操作閉環模擬畫面（ Push-T 完整軌跡可視化）會輸出至：
* **相對路徑**：`outputs/media`（保存為 `.mp4`  格式）


## 🏋️ 模型訓練流程 (Training Pipeline)

### 1. 冒煙測試 (Smoke Test)
用於在最短時間內驗證整個資料管線、反向傳播梯度計算、無頭離線渲染與硬體遙測日誌是否正常運作。僅執行 **2 個 Epochs (抽樣 10 Steps)**：

```bash
bash scripts/run_train.sh smoke
```

### 2. 完整模型訓練 (Full Training)
執行 3,050 Epochs 的完整 Diffusion Policy 訓練：
```bash
bash scripts/run_train.sh full
```

### 3. 訓練產出位置：
每次啟動訓練，程式會以「日期與時間戳記」自動建立專屬的實驗記錄目錄：

* **訓練權重檔 (Checkpoints)**：
  * **相對路徑**：`diffusion_policy/data/outputs/<YYYY.MM.DD>/<HH.MM.SS>_train_diffusion_unet_hybrid_pusht_image/checkpoints/`
  * **產出檔案**：
    * `latest.ckpt`：最後一個 Epoch 訓練完成時自動儲存的最新權重。
    * `epoch=XXXX-test_mean_score=X.XXX.ckpt`：驗證集分數刷新歷史最佳時自動保存的 Best Checkpoint。

* **訓練遙測日誌 (Training Telemetry Logs)**：
  * **相對路徑**：`outputs/train_telemetry_latest.json`（記錄 GPU 型號、顯存峰值 Peak VRAM、訓練總耗時 Wall-clock time）

---

## 🔬 模型評估與自訂驗證 (Inference & Evaluation)

### 1. 預設評測
執行預設驗證（使用自訓最佳權重 `epoch=0550-...ckpt`，隨機種子 `42`，評測 50 回合）：
```bash
bash scripts/run_reproduce.sh
```

### 2. 自訂驗證 Checkpoint 與 Random Seed

透過 Docker 執行以下指令，並直接修改最後三項參數即可：

```bash
docker run --gpus all --rm \
    -v $(pwd):/workspace \
    diff_policy:latest \
    python3 scripts/eval_with_log.py \
    --checkpoint checkpoints/epoch=0550-test_mean_score=0.841.ckpt \
    --seed 42 \
    --output outputs/eval_log_custom.json
```

**參數修改說明：**
* **`--checkpoint`**：改為要評測的權重路徑。
* **`--seed`**：改為欲測試的隨機種子（例如改為 `100` 或 `2024` 進行魯棒性評測）。
* **`--output`**：改為自訂的 JSON 日誌輸出檔名（例如 `outputs/eval_log_seed100.json`，避免覆蓋預設報告）。

### 3. 評測日誌與結果格式：
評測結束後，結構化評估報告將自動保存在宿主機的 **`outputs/`** 目錄下（例如 `outputs/eval_log_seed42.json`）：


---

## 📊 實驗結果與論文基準對比

在相同 50 回合推論設定下，自訓模型於 RTX 4080 上的復現數據如下：

| 指標 (Metric) | 論文原作者報告 (Paper Baseline) | 本專案復現成果 (Our Reproduction) | 備註說明 |
| :--- | :---: | :---: | :--- |
| **環境任務** | Push-T (Vision) | Push-T (Vision) | 9-step Action Chunking |
| **測試成功率 (Mean IoU)** | **~0.82 – 0.84** | **0.8199 (0.841 peak)** | 達到論文報告基準 |
| **推論顯存峰值 (Peak VRAM)** | 未特別揭露 | **1.11 GB** | 輕量化閉環推論 |
| **50 回合推論總耗時** | 未特別揭露 | **47.63 s** | 平均單回合 < 1 秒 |
| **訓練顯存峰值 (Peak VRAM)** | 未特別揭露 | **6.14 GB** | 批次大小 64 |

---

## 📂 專案目錄結構

```text
diffusion_policy_reproduce/
├── Dockerfile                 # RTX 4080 / CUDA 12.1 / EGL 無頭渲染專用映像檔定義
├── README.md                  # 完整專案重現說明文件
├── checkpoints/               # 存放推論評測權重 (由 Hugging Face 自動拉取)
│   ├── epoch=0550-test_mean_score=0.841.ckpt
│   └── pusht_vision_cnn.ckpt
├── diffusion_policy/          # 核心演算法與模型架構
│   └── data/
│       ├── pusht/             # Push-T 示範資料集 (Zarr 格式)
│       └── outputs/           # 訓練權重與 W&B 日誌輸出目錄
├── outputs/                   # 遙測日誌與評測報告目錄
│   ├── eval_log_seed42.json   # 50 回合閉環推論評測記錄
│   └── train_telemetry_latest.json
└── scripts/                   # 一鍵自動化腳本
    ├── download_assets.sh     # Hugging Face 資源拉取腳本
    ├── eval_with_log.py       # 帶有顯存與時間遙測之評測進入點
    ├── run_reproduce.sh       # 評分用一鍵自動重現腳本
    └── run_train.sh           # 冒煙測試與正式訓練排程腳本
    
```
