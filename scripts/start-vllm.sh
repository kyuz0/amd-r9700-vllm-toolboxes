#!/usr/bin/env bash
set -euo pipefail

# Defaults (override via env: HOST, PORT, DOWNLOAD_DIR, EXTRA_FLAGS)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$HOME/vllm-models}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

models=(
  "Llama 2 7B Chat|meta-llama/Llama-2-7b-chat-hf|"
  "Qwen2.5 7B Instruct|Qwen/Qwen2.5-7B-Instruct|"
  "Qwen3 30B A3B Instruct|Qwen/Qwen3-30B-A3B-Instruct-2507|"
  "Qwen3 14B AWQ|Qwen/Qwen3-14B-AWQ|--quantization awq --dtype float16 --enforce-eager"
  "Gemma 3 27B instruct|google/gemma-3-27b-it|"
  "Gemma 3 12B Instruct|google/gemma-3-12b-it|"
  "Gemma 3 4B Instruct|google/gemma-3-4b-it|"
)

echo "Select a model:"
for i in "${!models[@]}"; do
  name="${models[$i]%%|*}"
  printf "  [%d] %s\n" "$((i+1))" "$name"
done

read -rp "Enter number: " choice
[[ "$choice" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid choice."; exit 1; }
idx=$((choice-1))
(( idx >= 0 && idx < ${#models[@]} )) || { echo "Invalid choice."; exit 1; }

IFS='|' read -r label repo flags <<< "${models[$idx]}"

mkdir -p "$DOWNLOAD_DIR"

CMD=(vllm serve "$repo" --host "$HOST" --port "$PORT" --download-dir "$DOWNLOAD_DIR")

# Per-model flags
if [[ -n "${flags:-}" ]]; then
  # shellcheck disable=SC2206
  CMD+=($flags)
fi

# Optional global extras: e.g. EXTRA_FLAGS="--gpu-memory-utilization 0.8"
if [[ -n "${EXTRA_FLAGS:-}" ]]; then
  # shellcheck disable=SC2206
  CMD+=($EXTRA_FLAGS)
fi

echo -e "Running:\n\n  ${CMD[@]}\n"
echo "API test  →  curl -s http://localhost:${PORT}/v1/models | jq -r '.data[0].id'"
echo "SSH tip   →  ssh -L ${PORT}:localhost:${PORT} user@host"
echo

exec "${CMD[@]}"
