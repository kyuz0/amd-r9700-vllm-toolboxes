FROM registry.fedoraproject.org/fedora:43

# 1. System Base & Build Tools
# Added 'gperftools-libs' for tcmalloc (fixes double-free)
RUN dnf -y install --setopt=install_weak_deps=False --nodocs \
  python3.13 python3.13-devel git rsync libatomic bash ca-certificates curl \
  gcc gcc-c++ binutils make ffmpeg-free \
  cmake ninja-build aria2c tar xz vim nano \
  libdrm-devel zlib-devel openssl-devel jq \
  numactl-devel gperftools-libs dialog procps-ng \
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
  --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ \
  --pre torch torchaudio torchvision

# Flash-Attention
WORKDIR /opt
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"

RUN git clone https://github.com/ROCm/flash-attention.git &&\ 
  cd flash-attention &&\
  git checkout main_perf &&\
  python setup.py install && \
  cd /opt && rm -rf /opt/flash-attention

# 6. Clone vLLM
RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm
WORKDIR /opt/vllm

# --- PATCHING ---
# vLLM relies on 'amdsmi' to detect AMD GPUs. If it's missing or fails (common in containers),
# vLLM falls back to CPU. We patch it to force ROCm detection.
RUN echo "import sys, re" > patch_vllm.py && \
  echo "from pathlib import Path" >> patch_vllm.py && \
  # Patch 1: __init__.py - Force is_rocm=True and bypass amdsmi checks
  echo "p = Path('vllm/platforms/__init__.py')" >> patch_vllm.py && \
  echo "txt = p.read_text()" >> patch_vllm.py && \
  echo "txt = txt.replace('import amdsmi', '# import amdsmi')" >> patch_vllm.py && \
  echo "txt = re.sub(r'is_rocm = .*', 'is_rocm = True', txt)" >> patch_vllm.py && \
  echo "txt = re.sub(r'if len\(amdsmi\.amdsmi_get_processor_handles\(\)\) > 0:', 'if True:', txt)" >> patch_vllm.py && \
  echo "txt = txt.replace('amdsmi.amdsmi_init()', 'pass')" >> patch_vllm.py && \
  echo "txt = txt.replace('amdsmi.amdsmi_shut_down()', 'pass')" >> patch_vllm.py && \
  echo "p.write_text(txt)" >> patch_vllm.py && \
  # Patch 2: rocm.py - Mock amdsmi and force device name
  echo "p = Path('vllm/platforms/rocm.py')" >> patch_vllm.py && \
  echo "txt = p.read_text()" >> patch_vllm.py && \
  echo "header = 'import sys\nfrom unittest.mock import MagicMock\nsys.modules[\"amdsmi\"] = MagicMock()\n'" >> patch_vllm.py && \
  echo "txt = header + txt" >> patch_vllm.py && \
  echo "txt = re.sub(r'device_type = .*', 'device_type = \"rocm\"', txt)" >> patch_vllm.py && \
  echo "txt = re.sub(r'device_name = .*', 'device_name = \"gfx1201\"', txt)" >> patch_vllm.py && \
  echo "txt += '\n    def get_device_name(self, device_id: int = 0) -> str:\n        return \"AMD-gfx1201\"\n'" >> patch_vllm.py && \
  echo "p.write_text(txt)" >> patch_vllm.py && \
  echo "print('Successfully patched vLLM for R9700')" >> patch_vllm.py && \
  python patch_vllm.py

# 7. Build vLLM (Wheel Method) with CLANG Host Compiler
RUN python -m pip install --upgrade cmake ninja packaging wheel numpy "setuptools-scm>=8" "setuptools<80.0.0" scikit-build-core pybind11
ENV ROCM_HOME="/opt/rocm"
ENV HIP_PATH="/opt/rocm"
ENV VLLM_TARGET_DEVICE="rocm"
ENV PYTORCH_ROCM_ARCH="gfx1201"
ENV HIP_ARCHITECTURES="gfx1201"          
ENV AMDGPU_TARGETS="gfx1201"              
ENV MAX_JOBS="4"

# --- FIX FOR SEGFAULT ---
# We force the Host Compiler (CC/CXX) to be the ROCm Clang, not Fedora GCC.
# This aligns the ABI of the compiled vLLM extensions with PyTorch.
ENV CC="/opt/rocm/llvm/bin/clang"
ENV CXX="/opt/rocm/llvm/bin/clang++"

RUN export HIP_DEVICE_LIB_PATH=$(find /opt/rocm -type d -name bitcode -print -quit) && \
  echo "Compiling with Bitcode: $HIP_DEVICE_LIB_PATH" && \
  export CMAKE_ARGS="-DROCM_PATH=/opt/rocm -DHIP_PATH=/opt/rocm -DAMDGPU_TARGETS=gfx1201 -DHIP_ARCHITECTURES=gfx1201" && \   
  python -m pip wheel --no-build-isolation --no-deps -w /tmp/dist -v . && \
  python -m pip install /tmp/dist/*.whl

# --- bitsandbytes (ROCm) ---
WORKDIR /opt
RUN git clone -b rocm_enabled_multi_backend https://github.com/ROCm/bitsandbytes.git
WORKDIR /opt/bitsandbytes

# Explicitly set HIP_PLATFORM (Docker ENV, not /etc/profile)
ENV HIP_PLATFORM="amd"
ENV CMAKE_PREFIX_PATH="/opt/rocm"

# Force CMake to use the System ROCm Compiler (/opt/rocm/llvm/bin/clang++)
RUN cmake -S . \
  -DGPU_TARGETS="gfx1201" \
  -DBNB_ROCM_ARCH="gfx1201" \
  -DCOMPUTE_BACKEND=hip \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  && \
  make -j$(nproc) && \
  python -m pip install --no-cache-dir . --no-build-isolation --no-deps

# 8. Final Cleanup & Runtime
WORKDIR /opt
RUN chmod -R a+rwX /opt && \
  find /opt/venv -type f -name "*.so" -exec strip -s {} + 2>/dev/null || true && \
  find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + && \
  rm -rf /root/.cache/pip || true && \
  dnf clean all && rm -rf /var/cache/dnf/*

COPY scripts/01-rocm-envs.sh /etc/profile.d/01-rocm-envs.sh
COPY scripts/99-toolbox-banner.sh /etc/profile.d/99-toolbox-banner.sh
COPY scripts/zz-venv-last.sh /etc/profile.d/zz-venv-last.sh
COPY scripts/start_vllm.py /usr/local/bin/start-vllm
COPY benchmarks/max_context_results.json /opt/max_context_results.json
COPY benchmarks/run_vllm_bench.py /opt/run_vllm_bench.py
RUN chmod 0644 /etc/profile.d/*.sh && chmod +x /usr/local/bin/start-vllm && chmod 0644 /opt/max_context_results.json
RUN printf 'ulimit -S -c 0\n' > /etc/profile.d/90-nocoredump.sh && chmod 0644 /etc/profile.d/90-nocoredump.sh

CMD ["/bin/bash"]
