import os
import sys
import time
import json
import pathlib
import random
import numpy as np

# 1. 無頭渲染設定：避免 Pygame 在容器內報錯
os.environ["SDL_VIDEODRIVER"] = "dummy"

# 2. 修復 robomimic 0.3+ 命名空間相容性 (將 CropRandomizer 映射回 base_nets)
import robomimic.models.base_nets as rmbn
import robomimic.models.obs_core as rmoc
if not hasattr(rmbn, "CropRandomizer"):
    rmbn.CropRandomizer = rmoc.CropRandomizer

import torch
import dill
import hydra

# 3. 將 diffusion_policy 專案目錄加入模組搜尋路徑
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "diffusion_policy"))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_evaluation(checkpoint_path, output_dir, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    
    # 4. 啟動硬體與顯存監控
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    print(f"==================================================")
    print(f"Running Evaluation on: {gpu_name}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Seed: {seed}")
    print(f"==================================================")

    # 5. 載入 Checkpoint
    try:
        payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill, weights_only=False)
    except (TypeError, ValueError):
        try:
            payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        except Exception:
            payload = torch.load(checkpoint_path, map_location=device)
            
    cfg = payload['cfg']
    
    # 6. 動態反射載入 Workspace 並還原模型權重
    cls = hydra.utils.get_class(cfg._target_)
    try:
        workspace = cls(cfg, output_dir=output_dir)
    except TypeError:
        workspace = cls(cfg)
        workspace.output_dir = output_dir
        
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # 7. 取得策略模型（優先使用 EMA 平滑權重）
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.to(device)
    policy.eval()

    # 8. 依據官方標準實例化環境評測器 (Push-T Env Runner)
    print("Executing simulation rollouts (Push-T Env)...")
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir
    )
    runner_log = env_runner.run(policy)
    
    # 9. 結算運算耗時與 VRAM 峰值
    wall_clock = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    
    # 10. 產出符合作業要求的標準 eval JSON
    eval_results = {
        "benchmark": "Push-T (Vision-based)",
        "workspace_class": cfg._target_,
        "seed": seed,
        "compute_logs": {
            "gpu_hardware": gpu_name,
            "peak_vram_gb": round(peak_vram, 3),
            "wall_clock_time_sec": round(wall_clock, 2)
        },
        "metrics": {
            "test_mean_score": round(float(runner_log.get('test/mean_score', 0.0)), 4),
            "sim_max_reward": round(float(runner_log.get('test/sim_max_reward', 0.0)), 4)
        },
        "checkpoint": os.path.basename(checkpoint_path)
    }

    log_path = os.path.join(output_dir, f"eval_log_seed{seed}.json")
    with open(log_path, "w") as f:
        json.dump(eval_results, f, indent=4)
        
    print(f"\n[Success] Evaluation complete!")
    print(f"-> Peak VRAM: {peak_vram:.2f} GB")
    print(f"-> Wall-Clock Time: {wall_clock:.2f} s")
    print(f"-> Test Mean Score (IoU): {eval_results['metrics']['test_mean_score']}")
    print(f"-> Log saved to: {log_path}\n")

if __name__ == "__main__":
    import argparse

    default_ckpt = "/workspace/checkpoints/epoch=0550-test_mean_score=0.841.ckpt"
    fallback_ckpt = "/workspace/checkpoints/pusht_vision_cnn.ckpt"
    local_train_ckpt = "/workspace/diffusion_policy/data/outputs/2026.09.26/14.18.30_train_diffusion_unet_hybrid_pusht_image/checkpoints/epoch=0550-test_mean_score=0.841.ckpt"

    # 自動判斷預設權重
    if os.path.exists(default_ckpt):
        resolved_default_ckpt = default_ckpt
    elif os.path.exists(fallback_ckpt):
        resolved_default_ckpt = fallback_ckpt
    elif os.path.exists(local_train_ckpt):
        resolved_default_ckpt = local_train_ckpt
    else:
        resolved_default_ckpt = default_ckpt

    parser = argparse.ArgumentParser(description="Push-T Evaluation with Telemetry")
    parser.add_argument("--checkpoint", type=str, default=resolved_default_ckpt, help="Path to checkpoint .ckpt")
    parser.add_argument("--output", type=str, default="/workspace/outputs", help="Output directory or JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # 若傳入的是 .json 檔案路徑，取出其目錄作為 output_dir
    if args.output.endswith(".json"):
        out_dir = os.path.dirname(args.output) or "/workspace/outputs"
    else:
        out_dir = args.output

    run_evaluation(args.checkpoint, out_dir, seed=args.seed)
