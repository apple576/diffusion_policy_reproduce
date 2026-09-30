import os
import sys
import time
import json
import pathlib
import random
import argparse
import numpy as np

# 1. 無頭渲染設定：避免 Pygame 在容器內報錯
os.environ["SDL_VIDEODRIVER"] = "dummy"

# 2. 修復 robomimic 0.3+ 命名空間相容性
import robomimic.models.base_nets as rmbn
import robomimic.models.obs_core as rmoc
if not hasattr(rmbn, "CropRandomizer"):
    rmbn.CropRandomizer = rmoc.CropRandomizer

import torch
import dill
import hydra

# 3. 模組搜尋路徑
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "diffusion_policy"))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_evaluation(checkpoint_path, output_dir, seed=42, custom_json_name=None):
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    print(f"==================================================")
    print(f"Running Evaluation on: {gpu_name}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Random Seed (Env & Model): {seed}")
    print(f"==================================================")

    # 4. 載入 Checkpoint
    try:
        payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill, weights_only=False)
    except (TypeError, ValueError):
        try:
            payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        except Exception:
            payload = torch.load(checkpoint_path, map_location=device)
            
    cfg = payload['cfg']
    
    # 5. 還原 Workspace 與策略模型
    cls = hydra.utils.get_class(cfg._target_)
    try:
        workspace = cls(cfg, output_dir=output_dir)
    except TypeError:
        workspace = cls(cfg)
        workspace.output_dir = output_dir
        
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.to(device)
    policy.eval()

    # 6. 動態綁定 test_start_seed，使每次測試的關卡環境真正產生變化
    # 預設 seed=42 時嚴格使用官方基準關卡 100000；自訂其他種子時才變更關卡
    MAX_GYM_SEED = 2**32 - 10000
    safe_seed = abs(int(seed)) % MAX_GYM_SEED
    env_start_seed = 100000 if seed == 42 else safe_seed

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
        test_start_seed=env_start_seed
    )
    runner_log = env_runner.run(policy)
    
    wall_clock = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    
    eval_results = {
        "benchmark": "Push-T (Vision-based)",
        "workspace_class": cfg._target_,
        "seed": seed,
        "env_start_seed": seed * 1000,
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

    # 7. 檔名判定：若指定特定檔名則使用，否則依 seed 命名
    if custom_json_name:
        log_path = os.path.join(output_dir, custom_json_name) if not os.path.isabs(custom_json_name) else custom_json_name
    else:
        log_path = os.path.join(output_dir, f"eval_log_seed{seed}.json")

    with open(log_path, "w") as f:
        json.dump(eval_results, f, indent=4)
        
    print(f"\n[Success] Evaluation complete!")
    print(f"-> Peak VRAM: {peak_vram:.2f} GB")
    print(f"-> Wall-Clock Time: {wall_clock:.2f} s")
    print(f"-> Test Mean Score (IoU): {eval_results['metrics']['test_mean_score']}")
    print(f"-> Log saved to: {log_path}\n")

if __name__ == "__main__":
    default_ckpt = "/workspace/checkpoints/epoch=0550-test_mean_score=0.841.ckpt"
    fallback_ckpt = "/workspace/checkpoints/pusht_vision_cnn.ckpt"
    resolved_default_ckpt = default_ckpt if os.path.exists(default_ckpt) else fallback_ckpt

    parser = argparse.ArgumentParser(description="Push-T Evaluation with Telemetry")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .ckpt")
    parser.add_argument("--output", type=str, default="/workspace/outputs", help="Output directory or JSON filename")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    # 支援位置參數相容（相容 run_reproduce.sh 的舊調用）
    parser.add_argument("pos_checkpoint", nargs="?", type=str, default=None)
    parser.add_argument("pos_seed", nargs="?", type=int, default=None)

    args = parser.parse_args()

    ckpt = args.checkpoint or args.pos_checkpoint or resolved_default_ckpt
    seed = args.seed if args.seed is not None else (args.pos_seed if args.pos_seed is not None else 42)

    custom_name = None
    if args.output.endswith(".json"):
        custom_name = os.path.basename(args.output)
        out_dir = os.path.dirname(args.output) or "/workspace/outputs"
    else:
        out_dir = args.output

    run_evaluation(ckpt, out_dir, seed=seed, custom_json_name=custom_name)
