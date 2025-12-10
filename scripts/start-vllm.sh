#!/usr/bin/env bash
# start-vllm.sh - Interactive vLLM Launcher for R9700 using dialog (ncurses)
# Features: Robust TUI, Auto-GPU detection, Configurable defaults
set -u

# --- Dependencies ---
if ! command -v dialog >/dev/null 2>&1; then
    echo "Error: 'dialog' is required for this interface."
    echo "Please install it in your container (e.g., apt-get install dialog)"
    exit 1
fi

# --- Configuration ---
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CACHE_DIR="${HOME}/.cache/huggingface"
TEMP_FILE=$(mktemp)

# Cleanup on exit
trap 'rm -f "$TEMP_FILE"' EXIT

# Model Definitions (Format: "Name|Repo|MaxCtx|MaxTP|Flags|EnvVars")
MODELS=(
    "Llama 3.1 8B Instruct|meta-llama/Meta-Llama-3.1-8B-Instruct|65536|2||"
    "GPT-OSS 20B|openai/gpt-oss-20b|32768|2|--trust-remote-code|"
    "Qwen3 14B FP8|RedHatAI/Qwen3-14B-FP8-dynamic|32768|1|--trust-remote-code|"
    "Qwen3 30B 4-bit (GPTQ)|cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit|24576|2|--trust-remote-code|"
    "Qwen3 80B 4-bit (AWQ)|cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit|20480|2|--trust-remote-code --enforce-eager|VLLM_USE_TRITON_AWQ=1"
)

# --- Helpers ---
detect_gpus() {
    local count=0
    if command -v rocm-smi >/dev/null 2>&1; then
        count=$(rocm-smi --showid --csv 2>/dev/null | grep -c "GPU")
    fi
    if [[ "$count" -eq 0 ]]; then
         count=$(ls /dev/dri/renderD* 2>/dev/null | wc -l)
    fi
    echo "${count:-1}"
}

GPU_COUNT=$(detect_gpus)

# --- Functions ---

select_model() {
    local options=()
    for i in "${!MODELS[@]}"; do
        IFS='|' read -r name repo ctx tp flags env <<< "${MODELS[$i]}"
        options+=("$i" "$name")
    done

    dialog --clear --backtitle "AMD R9700 vLLM Launcher (GPUs: $GPU_COUNT)" \
           --title "Select Model" \
           --menu "Choose a model to serve:" 20 60 10 \
           "${options[@]}" 2> "$TEMP_FILE"
    
    local ret=$?
    local choice=$(cat "$TEMP_FILE")

    # Handle Cancel/Esc
    if [[ "$ret" -ne 0 ]] || [[ -z "$choice" ]]; then
        clear
        echo "Selection cancelled or no choice made."
        exit 0
    fi

    # Sanitize choice (must be integer)
    if ! [[ "$choice" =~ ^[0-9]+$ ]]; then
        clear
        echo "Error: Invalid selection received from dialog: '$choice'"
        exit 1
    fi
    
    # Do not echo choice, just return success
    return 0
}

configure_launch() {
    local idx="$1"
    
    # Verify index exists
    if [[ -z "${MODELS[$idx]+exists}" ]]; then
        echo "Error: Invalid model index '$idx'"
        exit 1
    fi

    IFS='|' read -r name repo max_ctx max_tp flags envs <<< "${MODELS[$idx]}"
    
    # Defaults
    local current_tp=$GPU_COUNT
    if [[ "$max_tp" -lt "$current_tp" ]]; then current_tp="$max_tp"; fi
    local current_ctx="$max_ctx"

    while true; do
        dialog --clear --backtitle "AMD R9700 vLLM Launcher" \
               --title "Configuration: $name" \
               --menu "Customize Launch Parameters:" 15 60 5 \
               "1" "Tensor Parallelism: $current_tp" \
               "2" "Context Length:     $current_ctx" \
               "3" "LAUNCH SERVER" 2> "$TEMP_FILE"
        
        local action=$(cat "$TEMP_FILE")
        
        case "$action" in
            1)
                dialog --title "Tensor Parallelism" \
                       --rangebox "Set TP Size (1-$max_tp)" 10 40 1 "$max_tp" "$current_tp" 2> "$TEMP_FILE"
                local new_tp=$(cat "$TEMP_FILE")
                if [[ -n "$new_tp" ]]; then current_tp=$new_tp; fi
                ;;
            2)
                dialog --title "Context Length" \
                       --inputbox "Enter Max Context Length:" 10 40 "$current_ctx" 2> "$TEMP_FILE"
                local new_ctx=$(cat "$TEMP_FILE")
                if [[ -n "$new_ctx" ]]; then current_ctx=$new_ctx; fi
                ;;
            3)
                break # Launch
                ;;
            *)
                # Escape/Cancel goes back to model selection? 
                # Or exit? Let's assume exit for safety, or return failure.
                # Returning failure to loop back to model select is user friendly.
                return 1
                ;;
        esac
    done

    # --- Execute ---
    local cmd="vllm serve $repo --host $HOST --port $PORT --tensor-parallel-size $current_tp --max-model-len $current_ctx $flags"
    
    # Confirmation / Run
    clear
    echo "============================================================"
    echo " Launching: $name"
    echo " Command:   $cmd"
    if [[ -n "$envs" ]]; then
        echo " Envs:      $envs"
        export $envs
    fi
    echo "============================================================"
    echo
    
    # shellcheck disable=SC2086
    exec $cmd
}

# --- Main Loop ---
while true; do
    # Run directly, do not capture stdout!
    select_model
    if [[ $? -ne 0 ]]; then
        exit 0 # Exit if select_model returned non-zero (cancel)
    fi
    
    # Read choice from file
    choice=$(cat "$TEMP_FILE")
    
    if ! configure_launch "$choice"; then
        continue 
    fi
    break
done
