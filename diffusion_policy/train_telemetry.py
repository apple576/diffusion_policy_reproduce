import sys
import time
import json
import os
import pathlib
import torch

# 1. 注入 Robomimic 相容性補丁
import robomimic.models.base_nets as rmbn
import robomimic.models.obs_core as rmoc
if not hasattr(rmbn, 'CropRandomizer'):
    rmbn.CropRandomizer = rmoc.CropRandomizer

# 設定無緩衝輸出，確保進度條即時刷新
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

# 2. 自動鎖定官方嵌套的 config 目錄路徑
_base_dir = pathlib.Path(__file__).parent.resolve()
if (_base_dir / "diffusion_policy" / "config").exists():
    _cfg_path = str(_base_dir / "diffusion_policy" / "config")
else:
    _cfg_path = str(_base_dir / "config")

@hydra.main(
    version_base=None,
    config_path=_cfg_path
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    # 3. 初始化算力監控 (Telemetry Initialization)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    start_time = time.perf_counter()

    print("\n" + "=" * 60)
    print(f"🚀 [Training Telemetry] 硬體設備: {gpu_name}")
    print(f"🚀 [Training Telemetry] 開始訓練任務: {cfg.task_name} (Seed: {cfg.training.seed})")
    print("=" * 60 + "\n")

    # 4. 實例化 Workspace 並執行訓練
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

    # 5. 結算算力開銷數據 (Telemetry Finalization)
    end_time = time.perf_counter()
    total_sec = end_time - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

    telemetry_data = {
        "benchmark": "Push-T (Vision-based)",
        "task_name": str(cfg.task_name),
        "hardware": {
            "gpu_device": gpu_name,
            "cuda_available": torch.cuda.is_available()
        },
        "compute_logs": {
            "peak_vram_gb": round(peak_vram_gb, 4),
            "wall_clock_time_sec": round(total_sec, 2),
            "wall_clock_time_human": f"{int(total_sec // 3600)}h {int((total_sec % 3600) // 60)}m {int(total_sec % 60)}s"
        },
        "hyperparameters": {
            "num_epochs": int(cfg.training.num_epochs),
            "seed": int(cfg.training.seed),
            "device": str(cfg.training.device)
        },
        "output_directory": os.getcwd()
    }

    # 6. 保存日誌至實驗資料夾與 outputs 根目錄
    exp_log_path = pathlib.Path(os.getcwd()) / "compute_telemetry.json"
    with open(exp_log_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_data, f, indent=4, ensure_ascii=False)

    global_outputs = pathlib.Path("/workspace/outputs")
    global_outputs.mkdir(parents=True, exist_ok=True)
    latest_log_path = global_outputs / "train_telemetry_latest.json"
    with open(latest_log_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_data, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("📊 [Training Telemetry Summary]")
    print(f"-> 顯示卡型號:     {gpu_name}")
    print(f"-> 峰值顯存 (VRAM): {peak_vram_gb:.4f} GB")
    print(f"-> 訓練總耗時:     {total_sec:.2f} 秒 ({int(total_sec // 60)}分 {int(total_sec % 60)}秒)")
    print(f"-> 實驗遙測日誌:   {exp_log_path}")
    print(f"-> 全域最新日誌:   {latest_log_path}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
