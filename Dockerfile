# ==============================================================================
# 1. 基礎映像檔：使用 Ubuntu 22.04 + CUDA 12.1 Devel（支援 RTX 4080 sm_89）
# ==============================================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ==============================================================================
# 2. 核心無頭渲染環境變數（支援 MuJoCo / robosuite / EGL 離線渲染評測）
# ==============================================================================
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# ==============================================================================
# 3. 安裝系統層級依賴與編譯函式庫
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    tar \
    software-properties-common \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libglew-dev \
    libglfw3 \
    libglfw3-dev \
    libgles2-mesa-dev \
    patchelf \
    libegl1 \
    libegl-dev \
    ffmpeg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# 4. 配置 Python 3.9 與 pip，並鎖定打包基礎套件
# ==============================================================================
RUN curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | python3.9 \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3

WORKDIR /workspace

# 升級核心打包套件（鎖定 setuptools<=65.5.0 避免舊版 gym/robosuite 編譯錯誤）
RUN pip install --no-cache-dir --upgrade pip "setuptools<=65.5.0" wheel

# ==============================================================================
# 5. 安裝相容 RTX 4080 的 PyTorch 2.x (CUDA 12.1)
# ==============================================================================
RUN pip install --no-cache-dir --default-timeout=100 --retries 5 \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ==============================================================================
# 6. 分層安裝 Python 相依套件
# ==============================================================================

# 6-1. 基礎科學計算庫與修復版 gym
RUN pip install --no-cache-dir --default-timeout=100 --retries 5 "numpy>=1.23.5,<1.25.0" Cython \
    && git clone --depth 1 -b v0.21.0 https://github.com/openai/gym.git /tmp/gym \
    && sed -i 's/opencv-python>=3\./opencv-python>=3/g' /tmp/gym/setup.py \
    && pip install --no-cache-dir --no-build-isolation --no-deps /tmp/gym \
    && rm -rf /tmp/gym

# 6-2. 視覺與影像處理庫（升級 scikit-image 解決 ABI 衝突）
RUN pip install --no-cache-dir --default-timeout=100 --retries 5 \
    opencv-python-headless \
    av \
    moviepy \
    imagecodecs \
    scikit-image>=0.20.0 \
    scikit-video==1.1.11

# 6-3. 物理模擬器與資料庫（MuJoCo / robosuite / Zarr）
RUN pip install --no-cache-dir --default-timeout=100 --retries 5 \
    mujoco==2.3.7 \
    robosuite==1.4.1 \
    pygame==2.1.2 \
    pymunk==6.2.1 \
    shapely==1.8.5.post1 \
    zarr==2.12.0 \
    numcodecs==0.10.2 \
    h5py

# 6-4. 擴散模型核心與訓練管理庫（並強制鎖定 NumPy ABI）
RUN pip install --no-cache-dir --default-timeout=100 --retries 5 \
    wandb \
    diffusers==0.11.1 \
    hydra-core==1.2.0 \
    einops \
    numba \
    dill==0.3.5.1 \
    robomimic \
    pandas \
    "huggingface_hub<0.20.0" \
    && pip install --no-cache-dir --force-reinstall "numpy>=1.23.5,<1.25.0"

# ==============================================================================
# 7. 環境路徑與進入點
# ==============================================================================
ENV PYTHONPATH="/workspace/diffusion_policy"

CMD ["/bin/bash"]
