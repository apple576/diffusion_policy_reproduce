# 1. 基礎映像檔：使用 Ubuntu 22.04 + CUDA 12.1 Devel（支援 RTX 4080 sm_89）
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 2. 核心無頭渲染環境變數（支援 MuJoCo / robosuite / EGL 離線渲染評測）
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# 3. 安裝系統層級依賴與編譯函式庫
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
    && apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    && rm -rf /var/lib/apt/lists/*

# 4. 配置 Python 3.9 與 pip
RUN curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | python3.9 \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3

WORKDIR /workspace

# 升級核心打包套件（鎖定 setuptools<=65.5.0）
RUN pip install --no-cache-dir --upgrade pip "setuptools<=65.5.0" wheel

# 5. 安裝相容 RTX 4080 的 PyTorch 2.x (CUDA 12.1)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 6. 安裝修復版 gym、鎖定 mujoco、robosuite 與所有核心依賴（含 huggingface_hub 舊版相容與 pandas）
RUN pip install --no-cache-dir "numpy<2.0.0,>=1.23.5" Cython \
    && git clone --depth 1 -b v0.21.0 https://github.com/openai/gym.git /tmp/gym \
    && sed -i 's/opencv-python>=3\./opencv-python>=3/g' /tmp/gym/setup.py \
    && pip install --no-cache-dir --no-build-isolation --no-deps /tmp/gym \
    && rm -rf /tmp/gym \
    && pip install --no-cache-dir \
        "numpy<2.0.0,>=1.23.5" \
        mujoco==2.3.7 \
        robosuite==1.4.1 \
        pygame==2.1.2 \
        pymunk==6.2.1 \
        shapely==1.8.5.post1 \
        scikit-image==0.19.3 \
        scikit-video==1.1.11 \
        zarr==2.12.0 \
        numcodecs==0.10.2 \
        h5py \
        wandb \
        diffusers==0.11.1 \
        hydra-core==1.2.0 \
        einops \
        numba \
        dill==0.3.5.1 \
        opencv-python-headless \
        av \
        moviepy \
        imagecodecs \
        robomimic \
        pandas \
        "huggingface_hub<0.20.0"

ENV PYTHONPATH="/workspace/diffusion_policy"

CMD ["/bin/bash"]
