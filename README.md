# AMD R9700 — vLLM Toolbox/Container (gfx1201, PyTorch)

An **fedora-based** Docker/Podman container that is **Toolbx-compatible** (usable as a Fedora toolbox) for serving LLMs with **vLLM** on **AMD Radeon R9700 (gfx1201)**. Built on the PyTorch nightly base to make ROCm on R9700 practical for day‑to‑day use.


---

## Table of Contents

* [Tested Models (Benchmarks)](#tested-models-benchmarks)
* [1) Toolbx vs Docker/Podman](#1-toolbx-vs-dockerpodman)
* [2) Quickstart — Fedora Toolbx (development)](#2-quickstart--fedora-toolbx-development)
* [3) Quickstart — Ubuntu (Distrobox)](#3-quickstart--ubuntu-distrobox)
* [4) Testing the API](#4-testing-the-api)
* [5) Quickstart — Podman/Docker](#5-quickstart--podmandocker)


## Tested Models (Benchmarks)

View full benchmarks at: [https://kyuz0.github.io/amd-r9700-vllm-toolboxes/](https://kyuz0.github.io/amd-r9700-vllm-toolboxes/)

| Model | GPUs (TP) | Mem Util | Context Capacity (1 / 4 / 8 / 16 Concurrency) |
| :--- | :--- | :--- | :--- |
| **`meta-llama/Meta-Llama-3.1-8B-Instruct`** | 1 | 0.98 | 127k / 127k / 127k / 127k |
|  | 2 | 0.98 | 105k / 105k / 105k / 105k |
| **`openai/gpt-oss-20b`** | 1 | 0.98 | 131k / 131k / 131k / 131k |
|  | 2 | 0.95 | 131k / 131k / 131k / 131k |
| **`RedHatAI/Qwen3-14B-FP8-dynamic`** | 1 | 0.98 | 41k / 41k / 41k / 41k |
|  | 2 | 0.95 | 41k / 41k / 41k / 41k |
| **`cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit`** | 1 | 0.98 | 151k / 151k / 151k / 151k |
|  | 2 | 0.98 | 262k / 262k / 262k / 262k |
| **`cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit`** | 2 | 0.98 | 145k / 156k / 156k / 156k |
| **`RedHatAI/gemma-3-12b-it-FP8-dynamic`** | 1 | 0.98 | 8k / 8k / 8k / 8k |
|  | 2 | 0.98 | 8k / 8k / 8k / 8k |
| **`RedHatAI/gemma-3-27b-it-FP8-dynamic`** | 2 | 0.98 | 60k / 60k / 60k / 60k |

### Advanced Tuning

See [TUNING.md](TUNING.md) for a guide on how to enable undervolting and raise the power limit on AMD R9700 cards on Linux to improve performance and efficiency.


---

## 1) Toolbx vs Docker/Podman

The `kyuz0/vllm-therock-gfx1201:latest` image can be used both as: 

* **Fedora Toolbx (recommended for development):** Toolbx shares your **HOME** and user, so models/configs live on the host. Great for iterating quickly while keeping the host clean. 
* **Docker/Podman (recommended for deployment/perf):** Use for running vLLM as a service (host networking, IPC tuning, etc.). Always **mount a host directory** for model weights so they stay outside the container.

---

## 2) Quickstart — Fedora Toolbx (development)

Create a toolbox that exposes the GPU and relaxes seccomp to avoid ROCm syscall issues:

```bash
toolbox create vllm-r9700 \
  --image docker.io/kyuz0/vllm-therock-gfx1201:latest \
  -- --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render --security-opt seccomp=unconfined
```

Enter it:

```bash
toolbox enter vllm-r9700
```

**Model storage:** Models are downloaded to `~/.cache/huggingface` by default. This directory is shared with the host if you created the toolbox correctly, so downloads persist.

Serve a model using the helper script **`start-vllm`** (it prints the exact `vllm serve` command and then runs it).

```bash
start-vllm
# pick a model from the menu; the script prints the serve command and launches it
```

> **Cache note:** vLLM writes compiled kernels to `~/.cache/vllm/`.

---

## 3) Quickstart — Ubuntu (Distrobox)

Ubuntu’s toolbox package still breaks GPU access, so use Distrobox instead:

```bash
distrobox create -n vllm-r9700 \
  --image docker.io/kyuz0/vllm-therock-gfx1201:latest \
  --additional-flags "--device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined"

distrobox enter vllm-r9700
```

> **Verification:** Run `rocm-smi` or simply `start-vllm` to check GPU access.

---

## 4) Testing the API

Once the server is up, hit the OpenAI‑compatible endpoint:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Hello! Test the performance."}]}'
```

You should receive a JSON response with a `choices[0].message.content` reply.

If you don't want to bother specifying the model name, you can run this which will query the currently deployed model:

```bash
MODEL=$(curl -s http://localhost:8000/v1/models | jq -r '.data[0].id') curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\":[{\"role\":\"user\",\"content\":\"Hello! Test the performance.\"}]
  }"
```

---

## 5) Quickstart — Podman/Docker

Prefer this for persistent services. We map `~/.cache/huggingface` to ensure models persist between runs.

**Llama 3.1 8B Instruct**

```bash
podman run -d --name vllm-llama3-8b \
  --ipc=host \
  --network host \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --group-add render \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  docker.io/kyuz0/vllm-therock-gfx1201:latest \
  bash -lc 'source /torch-therock/.venv/bin/activate; \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --dtype float16 \
      --max-model-len 65536 \
      --host 0.0.0.0 --port 8000'
```

**Qwen3 30B 4-bit (GPTQ)**

```bash
podman run -d --name vllm-qwen3-30b \
  --ipc=host \
  --network host \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --group-add render \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  docker.io/kyuz0/vllm-therock-gfx1201:latest \
  bash -lc 'source /torch-therock/.venv/bin/activate; \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    vllm serve cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit --dtype float16 \
      --max-model-len 24576 \
      --host 0.0.0.0 --port 8000'
```

**Qwen3 80B 4-bit (AWQ) - DUAL GPU**

```bash
podman run -d --name vllm-qwen3-80b \
  --ipc=host \
  --network host \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --group-add render \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  -e VLLM_USE_TRITON_AWQ=1 \
  docker.io/kyuz0/vllm-therock-gfx1201:latest \
  bash -lc 'source /torch-therock/.venv/bin/activate; \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    vllm serve cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit \
      --model cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit \
      --tensor-parallel-size 2 \
      --max-model-len 20480 \
      --enforce-eager \
      --host 0.0.0.0 --port 8000'
```
