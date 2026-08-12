#!/usr/bin/env bash
# Conservative defaults for the stable multi-architecture image. Backend and
# communication experiments belong in model profiles or the launch command.

export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
export FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:-TRUE}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-rocm}"
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-0}"
export VLLM_ROCM_USE_AITER_LINEAR="${VLLM_ROCM_USE_AITER_LINEAR:-0}"
export PYTHONNOUSERSITE=1
