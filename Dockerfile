FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch \
    transformers \
    huggingface_hub \
    runpod \
    pillow

# HF OFFLINE CACHE
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV HF_HUB_CACHE=/models/hf
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV HF_DATASETS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Pre-download model
RUN python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="utrobinmv/t5_summary_en_ru_zh_large_2048",
    local_dir="/models/hf/utrobinmv/t5",
    local_dir_use_symlinks=False
)
EOF

WORKDIR /app
COPY handler.py .

CMD ["python3", "handler.py"]
