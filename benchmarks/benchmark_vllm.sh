#!/bin/bash
set -e

# Define models to benchmark
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "google/gemma-3-12b-it"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    "openai/gpt-oss-20b"
)

# Define scenarios
# Format: "name:input_len:output_len"
SCENARIOS=(
    "standard:512:128"
    "long_context:16384:128"
    "throughput:128:128"
)

# Create output directory
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

echo "Starting vLLM Benchmarks..."
echo "============================="

for model in "${MODELS[@]}"; do
    model_name=$(basename "$model")
    echo "Benchmarking Model: $model"
    
    for scenario in "${SCENARIOS[@]}"; do
        IFS=':' read -r scenario_name input_len output_len <<< "$scenario"
        
        echo "  Scenario: $scenario_name (Input: $input_len, Output: $output_len)"
        
        log_file="$LOG_DIR/${model_name}_${scenario_name}.log"
        
        if [ -f "$log_file" ]; then
            echo "    SKIPPING: Result file $log_file already exists."
            continue
        fi

        # Run vLLM benchmark
        # We use --trust-remote-code as some new models might require it
        # We capture both stdout and stderr to the log file
        if vllm bench throughput \
            --backend vllm \
            --model "$model" \
            --input-len "$input_len" \
            --output-len "$output_len" \
            --num-prompts 1000 \
            --trust-remote-code \
            --skip-mm-profiling > "$log_file" 2>&1; then
            echo "    SUCCESS: Log saved to $log_file"
        else
            echo "    FAILURE: Check $log_file for details"
        fi
        
    done
    echo "-----------------------------"
done

echo "All benchmarks completed."
