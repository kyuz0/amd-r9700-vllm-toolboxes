FROM registry.fedoraproject.org/fedora:43

# 1. System Base & Build Tools
# Added 'gperftools-libs' for tcmalloc (fixes double-free)
RUN dnf -y install --setopt=install_weak_deps=False --nodocs \
  python3.13 python3.13-devel git rsync libatomic bash ca-certificates curl \
  gcc gcc-c++ binutils make ffmpeg-free \
  cmake ninja-build aria2c tar xz vim nano \
  libdrm-devel zlib-devel openssl-devel \
  numactl-devel gperftools-libs \
  && dnf clean all && rm -rf /var/cache/dnf/*

# 2. Install "TheRock" ROCm SDK (Tarball Method)
WORKDIR /tmp
ARG ROCM_MAJOR_VER=7
ARG GFX=gfx120X-all
RUN set -euo pipefail; \
  BASE="https://therock-nightly-tarball.s3.amazonaws.com"; \
  PREFIX="therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}"; \
  KEY="$(curl -s "${BASE}?list-type=2&prefix=${PREFIX}" \
  | tr '<' '\n' \
  | grep -o "therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}\..*\.tar\.gz" \
  | sort -V | tail -n1)"; \
  echo "Downloading Latest Tarball: ${KEY}"; \
  aria2c -x 16 -s 16 -j 16 --file-allocation=none "${BASE}/${KEY}" -o therock.tar.gz; \
  mkdir -p /opt/rocm; \
  tar xzf therock.tar.gz -C /opt/rocm --strip-components=1; \
  rm therock.tar.gz

# 3. Configure Global ROCm Environment
# We add LD_PRELOAD for tcmalloc here to fix the shutdown crash
RUN export ROCM_PATH=/opt/rocm && \
  BITCODE_PATH=$(find /opt/rocm -type d -name bitcode -print -quit) && \
  printf '%s\n' \
  "export ROCM_PATH=/opt/rocm" \
  "export HIP_PLATFORM=amd" \
  "export HIP_PATH=/opt/rocm" \
  "export HIP_CLANG_PATH=/opt/rocm/llvm/bin" \
  "export HIP_DEVICE_LIB_PATH=$BITCODE_PATH" \
  "export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:\$PATH" \
  "export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:\$LD_LIBRARY_PATH" \
  "export ROCBLAS_USE_HIPBLASLT=1" \
  "export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1" \
  "export VLLM_TARGET_DEVICE=rocm" \
  "export HIP_FORCE_DEV_KERNARG=1" \
  "export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1" \
  "export LD_PRELOAD=/usr/lib64/libtcmalloc_minimal.so.4" \
  > /etc/profile.d/rocm-sdk.sh && \
  chmod 0644 /etc/profile.d/rocm-sdk.sh

# 4. Python Venv Setup
RUN /usr/bin/python3.13 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PIP_NO_CACHE_DIR=1
RUN printf 'source /opt/venv/bin/activate\n' > /etc/profile.d/venv.sh
RUN python -m pip install --upgrade pip wheel packaging "setuptools<80.0.0"

# 5. Install PyTorch (TheRock Nightly)
RUN python -m pip install \
  --index-url https://rocm.nightlies.amd.com/v2-staging/gfx120X-all/ \
  --pre torch torchaudio torchvision

WORKDIR /opt

# Flash-Attention
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"

RUN git clone https://github.com/ROCm/flash-attention.git &&\ 
  cd flash-attention &&\
  git checkout main_perf &&\
  python setup.py install && \
  cd /opt && rm -rf /opt/flash-attention

# 6. Clone vLLM
RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm
WORKDIR /opt/vllm



# 7. Build vLLM (Wheel Method) with CLANG Host Compiler
RUN python -m pip install --upgrade cmake ninja packaging wheel numpy "setuptools-scm>=8" "setuptools<80.0.0" scikit-build-core pybind11
ENV ROCM_HOME="/opt/rocm"
ENV HIP_PATH="/opt/rocm"
ENV VLLM_TARGET_DEVICE="rocm"
ENV PYTORCH_ROCM_ARCH="gfx1201"
ENV HIP_ARCHITECTURES="gfx1201"          
ENV AMDGPU_TARGETS="gfx1201"              
ENV MAX_JOBS="4"

# --- CRITICAL FIX FOR SEGFAULT ---
# We force the Host Compiler (CC/CXX) to be the ROCm Clang, not Fedora GCC.
# This aligns the ABI of the compiled vLLM extensions with PyTorch.
ENV CC="/opt/rocm/llvm/bin/clang"
ENV CXX="/opt/rocm/llvm/bin/clang++"

RUN export HIP_DEVICE_LIB_PATH=$(find /opt/rocm -type d -name bitcode -print -quit) && \
  echo "Compiling with Bitcode: $HIP_DEVICE_LIB_PATH" && \
  export CMAKE_ARGS="-DROCM_PATH=/opt/rocm -DHIP_PATH=/opt/rocm -DAMDGPU_TARGETS=gfx1201 -DHIP_ARCHITECTURES=gfx1201" && \   
  python -m pip wheel --no-build-isolation --no-deps -w /tmp/dist -v . && \
  python -m pip install /tmp/dist/*.whl

# 8. Final Cleanup & Runtime
WORKDIR /opt
RUN chmod -R a+rwX /opt && \
  find /opt/venv -type f -name "*.so" -exec strip -s {} + 2>/dev/null || true && \
  find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + && \
  rm -rf /root/.cache/pip || true && \
  dnf clean all && rm -rf /var/cache/dnf/*

COPY scripts/01-rocm-env-for-triton.sh /etc/profile.d/01-rocm-env-for-triton.sh
COPY scripts/99-toolbox-banner.sh /etc/profile.d/99-toolbox-banner.sh
COPY scripts/zz-venv-last.sh /etc/profile.d/zz-venv-last.sh
RUN chmod 0644 /etc/profile.d/*.sh
RUN printf 'ulimit -S -c 0\n' > /etc/profile.d/90-nocoredump.sh && chmod 0644 /etc/profile.d/90-nocoredump.sh

CMD ["/bin/bash"]
