# Radiance toolbox for Radeon AI PRO R9700

Install **Radiance FP8 · two R9700 cards** or **Radiance MXFP4 · two R9700 cards** from AI Toolbox Cockpit. Both pull `docker.io/kyuz0/amd-r9700-vllm-toolboxes:radiance`; use Update for the current image.

The manual `build-radiance` workflow resolves an upstream stable release when available, otherwise the maintainer's documented `release` branch. That branch currently identifies itself as 1.1.0-rc1 with vLLM 0.30.0, Torch 2.12.0 and ROCm 7.14; this channel is experimental. Its component stack comes from the selected upstream recipe. Base images use mutable tags; source revisions and image IDs are recorded in build artifacts rather than required by the installation catalogue.

To build locally on the GPU host with Podman and Git, run `python3 scripts/build_radiance.py`. Compilation runs inside containers, with two jobs and a 24 GiB limit by default. The command builds the rolling tag locally without publishing it.

The publication workflow uses a prepared Linux runner selected by its `runner_label` input (default `r9700-radiance-builder`), with Podman, Git, Python 3, at least 32 GiB RAM and 80 GiB free disk. It builds with two workers and a 24 GiB memory limit. Existing Podman build layers are reusable. For a single-use runner, register it with `--ephemeral --no-default-labels --labels r9700-radiance-builder`; the runner deregisters after its job. No host package installation or system service is performed by the workflow. Registry authentication uses a temporary job file and is removed after publication.

## Models

Use Cockpit's vLLM Models panel to download the exact catalogue artifact for **Qwen3.8 27B native FP8** or **Qwen3.8 27B MXFP4 + FP8 MTP**. For MXFP4, run Prepare to create a separate converted checkpoint, then select its directory in Server Mode. The image includes the GGZ14 checkpoint converter and records its source and hash separately from Radiance's engine identity.

The two profiles use TP2, 67,840 context tokens, FP8 KV and R4D attention. MXFP4 uses the W4A8 path with FP8 activations; native FP8 uses FP8 weights. Start with one sequence for everyday serving. The measured performance allocation is available separately; check VRAM and cache replay before selecting concurrent workloads.
