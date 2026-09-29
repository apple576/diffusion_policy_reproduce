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
* **核心成果**：在單張 NVIDIA GeForce RTX 4080 (16GB) 上完成 3,050 Epochs 完整訓練，推論評測取得 **0.8199 Mean IoU**（成功達到論文報告之 0.82～0.84 基準水準），並嚴格記錄完整的顯存峰值（Peak VRAM）與運算時間（Wall-clock time）遙測日誌。

---

## 📋 目錄
1. [硬體環境與遙測規範](#硬體環境與遙測規範)
2. [快速開始：One-Command Docker 復現](#快速開始one-command-docker-復現)
3. [資源自動下載 (Dataset & Checkpoints)](#資源自動下載-dataset--checkpoints)
4. [模型訓練流程 (Training Pipeline)](#模型訓練流程-training-pipeline)
5. [模型評估 (Inference & Evaluation)](#模型評估-inference--evaluation)
6. [實驗結果與論文基準對比](#實驗結果與論文基準對比)
7. [專案目錄結構](#專案目錄結構)

---

## 💻 硬體環境與遙測規範

所有訓練與評測均在純淨容器化環境中執行，硬體遙測規格如下：
* **GPU**: NVIDIA GeForce RTX 4080 (16GB VRAM, Ada Lovelace, Compute Capability 8.9)
* **Host OS**: Ubuntu 22.04 LTS (x86_64)
* **Driver / CUDA**: NVIDIA Driver >= 535 / CUDA 12.1
* **Container**: Docker 24.x + NVIDIA Container Toolkit

---

## ⚡ 快速開始：One-Command Docker 復現

為符合課堂評測要求，本專案提供**一鍵自動化復現**。若本地尚未建置 Docker 映像檔或缺少模型權重，腳本將自動觸發 Hugging Face 下載與映像檔建置：

```bash
# 賦予腳本執行權限並啟動一鍵復現
chmod +x scripts/*.sh
bash scripts/run_reproduce.sh
