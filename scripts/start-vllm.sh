#!/usr/bin/env bash
# start-vllm.sh - Interactive vLLM Launcher for R9700
# Features: Auto-GPU detection, TUI-like interface, Configurable defaults
set -euo pipefail

# --- Colors & Styles ---
R=$'\e[0;31m'
G=$'\e[0;32m'
Y=$'\e[1;33m'
B=$'\e[0;34m'
C=$'\e[0;36m'
W=$'\e[1;37m'
N=$'\e[0m'
DIM=$'\e[2m'
BOLD=$'\e[1m'

# --- Configuration ---
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# We rely on internal cache by default now
CACHE_DIR="${HOME}/.cache/huggingface"

# Model Definitions (Format: "Name|Repo|MaxCtx|MaxTP|Flags|EnvVars")
# Note: MaxTP=1 means strictly 1 GPU. MaxTP=2 means can scale.
MODELS=(
    "Llama 3.1 8B Instruct|meta-llama/Meta-Llama-3.1-8B-Instruct|65536|2||"
    "GPT-OSS 20B|openai/gpt-oss-20b|32768|2|--trust-remote-code|"
    "Qwen3 14B FP8|RedHatAI/Qwen3-14B-FP8-dynamic|32768|1|--trust-remote-code|"
    "Qwen3 30B 4-bit (GPTQ)|cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit|24576|2|--trust-remote-code|"
    "Qwen3 80B 4-bit (AWQ)|cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit|20480|2|--trust-remote-code --enforce-eager|VLLM_USE_TRITON_AWQ=1"
)

# --- Helpers ---
clear_screen() { printf "\e[H\e[2J"; }
cursor_to() { printf "\e[%s;%sH" "$1" "$2"; }
print_bar() { printf "${DIM}%*s${N}\n" "${COLUMNS:-$(tput cols)}" '' | tr ' ' '-'; }
center() { 
    local text="$1"
    local width="${COLUMNS:-$(tput cols)}"
    local pad=$(( (width - ${#text}) / 2 ))
    printf "%*s%s\n" $pad "" "$text"
}

# --- GPU Detection ---
detect_gpus() {
    local count=0
    if command -v rocm-smi >/dev/null 2>&1; then
        count=$(rocm-smi --showid --csv 2>/dev/null | grep -c "GPU")
    fi
    # Fallback to enumerating kfd/render devices if rocm-smi check weird
    if [[ "$count" -eq 0 ]]; then
         # rough heuristic
         count=$(ls /dev/dri/renderD* 2>/dev/null | wc -l)
    fi
    echo "${count:-1}"
}

GPU_COUNT=$(detect_gpus)

# --- Interface ---
draw_header() {
    clear_screen
    echo -e "${B}"
    center " ██    ██ ██      ██      ███    ███ "
    center " ██    ██ ██      ██      ████  ████ "
    center " ██    ██ ██      ██      ██ ████ ██ "
    center "  ██  ██  ██      ██      ██  ██  ██ "
    center "   ████   ███████ ███████ ██      ██ "
    echo -e "${N}"
    center "${DIM}AMD R9700 Launcher (Detected GPUs: ${W}${GPU_COUNT}${DIM})${N}"
    print_bar
}

select_model() {
    local selected=0
    local key=""
    
    while true; do
        draw_header
        echo -e " ${BOLD}Select a Model to Serve:${N}\n"
        
        for i in "${!MODELS[@]}"; do
            IFS='|' read -r name repo ctx tp flags env <<< "${MODELS[$i]}"
            
            prefix="   "
            color="${N}"
            if [[ "$i" -eq "$selected" ]]; then
                prefix=" ${Y}●${N} "
                color="${BOLD}${W}"
            fi
            
            # Info string
            info="${DIM}(Ctx: ${ctx}"
            [[ "$tp" -eq 1 ]] && info="${info}, ${R}Single GPU${DIM}"
            [[ "$tp" -eq 2 && "$GPU_COUNT" -ge 2 ]] && info="${info}, ${G}Dual GPU Ready${DIM}"
            info="${info})${N}"
            
            printf "${prefix}${color}%-30s ${info}\n" "$name"
        done
        
        echo -e "\n ${DIM}Use [UP/DOWN] to navigate, [ENTER] to select, [Q] to quit${N}"
        
        # Input loop
        read -rsn1 key
        if [[ "$key" == $'\x1b' ]]; then
            read -rsn2 key
            case "$key" in
                '[A') # UP
                    ((selected--))
                    [[ $selected -lt 0 ]] && selected=$(( ${#MODELS[@]} - 1 ))
                    ;;
                '[B') # DOWN
                    ((selected++))
                    [[ $selected -ge ${#MODELS[@]} ]] && selected=0
                    ;;
            esac
        elif [[ "$key" == "" ]]; then
            break # ENTER
        elif [[ "$key" == "q" || "$key" == "Q" ]]; then
            clear_screen; exit 0
        fi
    done
    
    return "$selected"
}

configure_launch() {
    local idx="$1"
    IFS='|' read -r name repo max_ctx max_tp flags envs <<< "${MODELS[$idx]}"
    
    # Defaults
    local current_tp=$GPU_COUNT
    # Clamp TP if model doesn't support it or we don't have enough GPUs
    if [[ "$max_tp" -lt "$current_tp" ]]; then current_tp="$max_tp"; fi
    
    local current_ctx="$max_ctx"
    
    local selected_setting=0
    
    while true; do
        draw_header
        echo -e " ${BOLD}Configuration: ${C}${name}${N}\n"
        
        # Menu Items
        local opts=(
            "Tensor Parallelism (TP) : ${current_tp}"
            "Context Length          : ${current_ctx}"
            "Extra Flags             : ${flags}"
            "${G}► LAUNCH SERVER${N}"
        )
        
        for i in "${!opts[@]}"; do
            prefix="   "
            if [[ "$i" -eq "$selected_setting" ]]; then
                prefix=" ${Y}●${N} "
            fi
            echo -e "${prefix}${opts[$i]}"
        done
        
        echo -e "\n ${DIM}[UP/DOWN] Navigate, [LEFT/RIGHT] Change Value, [ENTER] Confirm${N}"
        
        read -rsn1 key
        if [[ "$key" == $'\x1b' ]]; then
            read -rsn2 key
            case "$key" in
                '[A') # UP
                    ((selected_setting--))
                    [[ $selected_setting -lt 0 ]] && selected_setting=3
                    ;;
                '[B') # DOWN
                    ((selected_setting++))
                    [[ $selected_setting -gt 3 ]] && selected_setting=0
                    ;;
                '[C') # RIGHT
                    if [[ "$selected_setting" -eq 0 ]]; then
                        # Toggle TP
                        if [[ "$current_tp" -lt "$max_tp" && "$current_tp" -lt "$GPU_COUNT" ]]; then
                             ((current_tp++))
                        fi
                    fi
                    ;;
                '[D') # LEFT
                     if [[ "$selected_setting" -eq 0 ]]; then
                        # Toggle TP
                        if [[ "$current_tp" -gt 1 ]]; then
                             ((current_tp--))
                        fi
                    fi
                    ;;
            esac
        elif [[ "$key" == "" ]]; then
            if [[ "$selected_setting" -eq 3 ]]; then
                break # Launch
            fi
        fi
    done
    
    # Launch
    clear_screen
    echo -e "${G}Initializing vLLM...${N}"
    
    # Construct Command
    local cmd="vllm serve $repo --host $HOST --port $PORT --tensor-parallel-size $current_tp --max-model-len $current_ctx $flags"
    
    # Print nice summary
    print_bar
    echo -e "${BOLD}Model:${N}  $name"
    echo -e "${BOLD}Repo:${N}   $repo"
    echo -e "${BOLD}TP:${N}     $current_tp"
    echo -e "${BOLD}Ctx:${N}    $current_ctx"
    [[ -n "$envs" ]] && echo -e "${BOLD}Env:${N}    $envs"
    echo -e "${BOLD}Cmd:${N}    $cmd"
    print_bar
    echo
    
    # Execute
    if [[ -n "$envs" ]]; then
        export $envs
    fi
    
    # We do a little manual eval to handle the flag string logic safely enough for this context
    # shellcheck disable=SC2086
    exec $cmd
}

# --- Main ---
select_model
choice=$?
configure_launch "$choice"
