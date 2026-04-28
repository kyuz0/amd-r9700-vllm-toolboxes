# Multi-stage vLLM Toolbox Build for gfx1201 (Ubuntu-based)

ARG BASE_IMAGE="docker.io/kyuz0/pytorch-toolbox-gfx1201:v2.11.0-rocm-7.2.1"
ARG MAX_JOBS=""

############# rocm_base #############
FROM ${BASE_IMAGE} AS rocm_base

ENV DEBIAN_FRONTEND=noninteractive

# Add minimal apt packages for source fetching
RUN apt-get update && apt-get install -y --no-install-recommends \
    git aria2 curl jq tzdata ca-certificates numactl && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade --ignore-installed pyjwt

# Emulate original Dockerfile Environment Settings
ENV ROCM_PATH=/opt/rocm
ENV HIP_PLATFORM=amd
ENV HIP_PATH=/opt/rocm
ENV HIP_CLANG_PATH=/opt/rocm/llvm/bin
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib:/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
ENV ROCBLAS_USE_HIPBLASLT=1
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
ENV VLLM_TARGET_DEVICE=rocm
ENV HIP_FORCE_DEV_KERNARG=1
ENV RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
ENV HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode
ENV VLLM_HOST_IP=127.0.0.1
# LD_PRELOAD for ubuntu packaging of tcmalloc is moved to final stage

############# Fetch Sources Phase #############
FROM rocm_base AS files_fa
WORKDIR /app/flash-attention
RUN git clone https://github.com/ROCm/flash-attention.git . && \
    git checkout main_perf

FROM rocm_base AS files_vllm
WORKDIR /app/vllm
RUN git clone https://github.com/vllm-project/vllm.git .

# Apply R9700 / gfx1201 patches to vLLM. patch_r9700.py is a port of
# kyuz0/amd-strix-halo-vllm-toolboxes/scripts/patch_strix.py with the
# arch swapped to gfx1201 and the APU-only Patch 10 dropped. The
# original 4-patch heredoc this replaced predates the gfx1x AITER
# patches that current vLLM main needs to load Qwen3.5/3.6 model
# classes — that's why the published kyuz0 image fails on Qwen3.6.
COPY scripts/patch_r9700.py /app/vllm/patch_r9700.py
RUN python3 patch_r9700.py

FROM rocm_base AS files_bnb
WORKDIR /app/bitsandbytes
RUN git clone -b rocm_enabled_multi_backend https://github.com/ROCm/bitsandbytes.git .

############# Build Base Phase #############
FROM rocm_base AS build_base
RUN pip3 install --upgrade --ignore-installed build cmake ninja packaging wheel numpy "setuptools-scm>=8" "setuptools<80.0.0" scikit-build-core pybind11

############# Build wheels Phase #############
FROM build_base AS build_fa
COPY --from=files_fa /app/flash-attention /app/flash-attention
WORKDIR /app/flash-attention
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
ENV PYTORCH_ROCM_ARCH="gfx1201"
ARG MAX_JOBS
RUN MAX_JOBS=${MAX_JOBS:-$(nproc)} \
    python3 -m pip wheel --no-build-isolation --no-deps -w /dist -v .

FROM build_base AS build_vllm
COPY --from=files_vllm /app/vllm /app/vllm
WORKDIR /app/vllm
RUN pip3 install -r requirements/rocm.txt
ENV PYTORCH_ROCM_ARCH="gfx1201"
ENV HIP_ARCHITECTURES="gfx1201"
ENV AMDGPU_TARGETS="gfx1201"
ARG MAX_JOBS
RUN MAX_JOBS=${MAX_JOBS:-$(nproc)} \
    export HIP_DEVICE_LIB_PATH=$(find /opt/rocm -type d -name bitcode -print -quit) && \
    export CMAKE_ARGS="-DAMDGPU_TARGETS=gfx1201 -DHIP_ARCHITECTURES=gfx1201" && \
    python3 -m pip wheel --no-build-isolation --no-deps -w /dist -v .

FROM build_base AS build_bnb
COPY --from=files_bnb /app/bitsandbytes /app/bitsandbytes
WORKDIR /app/bitsandbytes
ENV HIP_PLATFORM="amd"
ENV CMAKE_PREFIX_PATH="/opt/rocm"
# Use direct compilation similar to old Dockerfile but package it as a wheel
RUN cmake -S . \
  -DGPU_TARGETS="gfx1201" \
  -DBNB_ROCM_ARCH="gfx1201" \
  -DCOMPUTE_BACKEND=hip \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  && \
  make -j$(nproc) && \
  python3 setup.py bdist_wheel --dist-dir=/dist

############# Final Container #############
FROM rocm_base AS final

# Install minimal runtime utils
RUN apt-get update && apt-get install -y --no-install-recommends \
    dialog vim pciutils procps libtcmalloc-minimal4 && \
    rm -rf /var/lib/apt/lists/*

# Fix shutdown crashes
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# Install generated python wheels
RUN --mount=type=bind,from=build_fa,src=/dist/,target=/dist_fa \
    --mount=type=bind,from=build_vllm,src=/dist/,target=/dist_vllm \
    --mount=type=bind,from=build_vllm,src=/app/vllm/requirements/,target=/app/vllm/requirements \
    --mount=type=bind,from=build_bnb,src=/dist/,target=/dist_bnb \
    pip3 install /dist_fa/*.whl /dist_vllm/*.whl /dist_bnb/*.whl -r /app/vllm/requirements/rocm.txt && \
    pip3 install git+https://github.com/ROCm/aiter.git && \
    pip3 cache purge

# Custom RCCL injection disabled for stable ROCm 7.2.1 flow.

# Deploy scripts and configurations
COPY scripts/01-rocm-envs.sh /etc/profile.d/01-rocm-envs.sh
COPY scripts/99-toolbox-banner.sh /etc/profile.d/99-toolbox-banner.sh
COPY scripts/start_vllm.py /usr/local/bin/start-vllm
COPY benchmarks/max_context_results.json /opt/max_context_results.json
COPY benchmarks/run_vllm_bench.py /opt/run_vllm_bench.py

RUN chmod 0644 /etc/profile.d/*.sh && chmod +x /usr/local/bin/start-vllm && chmod 0644 /opt/max_context_results.json
RUN printf 'ulimit -S -c 0\n' > /etc/profile.d/90-nocoredump.sh && chmod 0644 /etc/profile.d/90-nocoredump.sh

CMD ["/bin/bash"]
