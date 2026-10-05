# GGZ14 MXFP4 toolboxes for Radeon AI PRO R9700

Install **GGZ14 MXFP4 · one R9700** or **GGZ14 MXFP4 · two R9700 cards** from AI Toolbox Cockpit. These profiles pull the rolling public channels `docker.io/kyuz0/amd-r9700-vllm-toolboxes:ggz14-tp1` and `:ggz14-tp2`. Use Update in Toolboxes to obtain the current build.

The manual `build-ggz14` workflow resolves the latest upstream release at build time. GGZ14 currently has no GitHub Releases; its maintained `main` branch supplies version 0.13.0 and the compatible published Radiance base. The build follows that upstream stack, uses mutable base tags, rebuilds its patches and RX9 single-card kernel overlay, and records resolved sources, converter hash and image IDs in the workflow artifact. The inherited base includes published binaries; the wrapper does not establish complete source provenance for every binary. Both channels are experimental.

For a local build on the GPU host, install Podman, Skopeo and Git, then run `python3 scripts/build_ggz14.py`. Compilation runs inside containers. The default two build jobs and 24 GiB memory limit can be changed with `--jobs` and `--memory`. The command builds the rolling tags locally and does not publish them.

## Models

In Cockpit's vLLM Models panel, download **Qwen3.8 27B MXFP4 + FP8 MTP**, then choose **Prepare**. This downloads the exact catalogue revision of `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` and converts a separate checkpoint using the selected engine image's converter. Keep the original snapshot; preparation needs about 20 GB additional disk space and records converter provenance. Select the prepared directory in Server Mode.

Both profiles use R4D attention, FP8 KV and aligned hybrid prefix caching. The one-card image includes RX9 narrow-state GDN kernels; the dual image retains RX6 kernels. Everyday launches start with one sequence at 67,840 context tokens. Performance profiles expose the recorded server allocation separately from client concurrency; concurrent throughput does not imply all long histories remain resident.

For DFlash2, download **Qwen3.8 27B DFlash2 FP8 draft** in Models and select **DFlash2-7** in Server Mode. This qualified profile uses one sequence. The dual profile disables allreduce/RMS compiler fusion and uses an 8,192-token prefill chunk. Use baseline decoding for concurrent serving.
