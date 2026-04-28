# Fork notes — Strix patch port for gfx1201

This fork started from `kyuz0/amd-r9700-vllm-toolboxes`. Upstream's
inline vLLM patches (in both the Fedora `Dockerfile` and the Ubuntu
`Dockerfile.rocm7.2.1`) predate the gfx1x AITER patches that current
vLLM main needs to load `Qwen3_5*` / `Qwen3_5Moe*` model classes —
which is why the published `kyuz0/vllm-therock-gfx1201:latest` image
fails to load Qwen3.5 / Qwen3.6 models.

The Strix-Halo sibling repo
(`kyuz0/amd-strix-halo-vllm-toolboxes`) tracks those patches in
`scripts/patch_strix.py`. This fork ports them to gfx1201 in
`scripts/patch_r9700.py` and wires that into the existing Ubuntu
`Dockerfile.rocm7.2.1` build pipeline (renamed to `Dockerfile`).

## What changed vs upstream

- **`Dockerfile`** — was upstream's `Dockerfile.rocm7.2.1`. The inline
  4-patch heredoc that wrote `patch_vllm.py` from echo'd lines was
  replaced with `COPY scripts/patch_r9700.py` + `RUN python3
  patch_r9700.py`. Nothing else in the multi-stage build changed.
- **Old Fedora `Dockerfile` + `refresh-toolbox.sh`** — deleted. The
  Ubuntu/multi-stage pipeline based on
  `kyuz0/pytorch-toolbox-gfx1201:v2.11.0-rocm-7.2.1` is the canonical
  build now.
- **`scripts/patch_r9700.py`** — new. Port of Strix's
  `patch_strix.py`:
  - Arch swapped `gfx1151` → `gfx1201` in vLLM's `_get_gcn_arch` and
    `device_name` overrides. Most patches gate on `on_gfx1x()` which
    already covers the entire gfx1xxx family, so they apply
    unchanged.
  - Strix's Patch 10 (ROCM-21812 APU VRAM dynamic margin via
    `/sys/class/drm` GTT scraping) is **dropped** — that workaround
    is for RDNA 3.5 APUs with shared GTT memory. R9700 has dedicated
    VRAM, no GTT split.
  - Patch 9 (Triton MoE compute-capability cap) bumped from
    `< (11, 0)` to `< (13, 0)` so RDNA 4's reported capability is
    permitted (Strix's port to `< (12, 0)` covered RDNA 3.5 only).
- **`.github/workflows/build.yml`** — new. Builds on push to main,
  weekly cron, or manual dispatch. Publishes to
  `ghcr.io/<owner>/vllm-r9700-gfx1201:<date>-<sha>` and `:latest`.

## What was NOT ported from Strix

- Cluster scripts (`start_vllm_cluster.py`, `cluster_manager.py`,
  RDMA bench scripts, etc.) — multi-node/RDMA features the homelab
  use case doesn't need.
- AITER + composable_kernel source build — upstream's
  `Dockerfile.rocm7.2.1` already pulls AITER via
  `pip3 install git+https://github.com/ROCm/aiter.git`, which is
  simpler. If we hit AITER bugs that the source build fixes, revisit.
- `patch_aiter_headers.py` — only relevant when AITER is built from
  source (it edits AITER's CK headers). Not needed with the pip
  install path.
- `build_rccl_*.sh` / `manage_rccl_install.sh` — upstream
  `Dockerfile.rocm7.2.1` has "Custom RCCL injection disabled for
  stable ROCm 7.2.1 flow" — the bundled ROCm 7.2.1 RCCL is fine.

## Long-term

If this works, a PR back to `kyuz0/amd-r9700-vllm-toolboxes`
upstreaming `patch_r9700.py` is the right end state. Until then, this
fork is the canonical R9700 image for the homelab.
