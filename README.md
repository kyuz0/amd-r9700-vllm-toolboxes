# AMD Radeon 9700 AI PRO (gfx1201) — vLLM Toolbox/Container

An **fedora-based** Docker/Podman container that is **Toolbx-compatible** (usable as a Fedora toolbox) for serving LLMs with **vLLM** on **AMD Radeon R9700 (gfx1201)**. Built on the TheRock nightly builds for ROCM.

![Demo](demo.gif)

---

## Table of Contents

* [Tested Models (Benchmarks)](#tested-models-benchmarks)
* [1) Toolbx vs Docker/Podman](#1-toolbx-vs-dockerpodman)
* [2) Quickstart — Fedora Toolbx](#2-quickstart--fedora-toolbx)
* [3) Quickstart — Ubuntu (Distrobox)](#3-quickstart--ubuntu-distrobox)
* [4) Testing the API](#4-testing-the-api)
* [5) Use a Web UI for Chatting](#5-use-a-web-ui-for-chatting)


## Tested Models (Benchmarks)

View full benchmarks at: [https://kyuz0.github.io/amd-r9700-vllm-toolboxes/](https://kyuz0.github.io/amd-r9700-vllm-toolboxes/)

*Run benchmarks now include a comparison between the default Triton backend and the optional ROCm attention backend.*


**Table Key:** Cell values represent `Max Context Length (GPU Memory Utilization)`.

| Model | TP | 1 Req | 4 Reqs | 8 Reqs | 16 Reqs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`meta-llama/Meta-Llama-3.1-8B-Instruct`** | 1 | 127k (0.98) | 127k (0.98) | 127k (0.98) | 127k (0.98) |
|  | 2 | 105k (0.98) | 105k (0.98) | 105k (0.98) | 105k (0.98) |
| **`openai/gpt-oss-20b`** | 1 | 131k (0.98) | 131k (0.98) | 131k (0.98) | 131k (0.98) |
|  | 2 | 131k (0.95) | 131k (0.95) | 131k (0.95) | 131k (0.95) |
| **`RedHatAI/Qwen3-14B-FP8-dynamic`** | 1 | 41k (0.98) | 41k (0.98) | 41k (0.98) | 41k (0.98) |
|  | 2 | 41k (0.95) | 41k (0.95) | 41k (0.95) | 41k (0.95) |
| **`cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit`** | 1 | 151k (0.98) | 151k (0.98) | 151k (0.98) | 151k (0.98) |
|  | 2 | 262k (0.98) | 262k (0.98) | 262k (0.98) | 262k (0.98) |
| **`cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit`** | 2 | 156k (0.98) | 156k (0.98) | 156k (0.98) | 156k (0.98) |
| **`RedHatAI/gemma-3-12b-it-FP8-dynamic`** | 1 | 45k (0.98) | 45k (0.98) | 45k (0.98) | 45k (0.98) |
|  | 2 | 126k (0.98) | 126k (0.98) | 121k (0.95) | 121k (0.95) |
| **`RedHatAI/gemma-3-27b-it-FP8-dynamic`** | 2 | 60k (0.98) | 60k (0.98) | 60k (0.98) | 60k (0.98) |

### Advanced Tuning

See [TUNING.md](TUNING.md) for a guide on how to enable undervolting and raise the power limit on AMD R9700 cards on Linux to improve performance and efficiency.

### 🆕 Update: Comparison of Attention Backends (Triton vs ROCm)

*Added Support for ROCm Native Attention Backend*

I have added the ability to switch between the default **Triton** backend and the experimental **ROCm** native backend for attention operations. This provides you with more flexibility to optimize for stability or throughput depending on your specific model and workload.

| Backend | Stability | Throughput | Compatibility |
| :--- | :--- | :--- | :--- |
| **Triton** (Default) | ✅ **High** | 🔸 Good | Works with all tested models |
| **ROCm** | ⚠️ **Experimental** | 🚀 **Highest** | May fail with complex architectures |

**Key Differences:**
- **Triton**: The safe choice. It uses the Triton compiler to generate kernels and is the standard for vLLM on AMD.
- **ROCm**: Uses composable kernel based attention. In my benchmarks, this often yields higher throughput (tokens/sec) but can be less stable, leading to crashes or "invalid graph" errors on some newer models.

**How to Use:**
1. **Easy Mode**: Select the backend in the `start-vllm` wizard (Item 5 in the menu).
2. **Manual Mode**: Export the following environment variables before running `vllm serve`:
   ```bash
   export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
   export VLLM_USE_TRITON_FLASH_ATTN=0
   ```


---

## 1) Toolbx vs Docker/Podman

The `kyuz0/vllm-therock-gfx1201:latest` image can be used both as: 

* **Fedora Toolbx (recommended for development):** Toolbx shares your **HOME** and user, so models/configs live on the host. Great for iterating quickly while keeping the host clean. 
* **Docker/Podman (recommended for deployment/perf):** Use for running vLLM as a service (host networking, IPC tuning, etc.). Always **mount a host directory** for model weights so they stay outside the container.

---

## 2) Quickstart — Fedora Toolbx

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

### Serving a Model (Easiest Way)

The toolbox includes a TUI wizard called **`start-vllm`** which includes pre-configured models and handles launch flags. It also allows you to select the experimental **ROCm attention backend**. This is the easiest way to get started.

```bash
start-vllm
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

> **Verification:** Run `rocm-smi` to check GPU status.

### Serving a Model
Same as above, you can use the **`start-vllm`** wizard to launch models easily.

```bash
start-vllm
```

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

## 5) Use a Web UI for Chatting

If vLLM is on a remote server, expose port 8000 via SSH port forwarding:

```bash
ssh -L 0.0.0.0:8000:localhost:8000 <vllm-host>
```

Then, you can start HuggingFace ChatUI like this (on your host):

```bash
docker run -p 3000:3000 \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=dummy \
  -v chat-ui-data:/data \
  ghcr.io/huggingface/chat-ui-db
```

