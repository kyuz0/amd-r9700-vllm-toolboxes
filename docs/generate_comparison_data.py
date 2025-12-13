import json
import re
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
BENCHMARKS_ROOT = PROJECT_ROOT / "benchmarks"
OUTPUT_FILE = SCRIPT_DIR / "comparison_results.json"

# GPU Mapping
GPU_MAP = {
    "benchmark_results_amd-r9700": "AMD R9700",
    "benchmark_results_amd-r9700-uv+pl": "AMD R9700 (UV+PL)",
    "benchmark_results_nvidia-3090": "NVIDIA RTX 3090",
    "benchmark_results_nvidia-4090": "NVIDIA RTX 4090",
    "benchmark_results_nvidia-5090": "NVIDIA RTX 5090",
    "benchmark_results_nvidia-ada5000": "NVIDIA RTX 5000 Ada",
    "benchmark_results_nvidia-a100": "NVIDIA A100"
}

def parse_model_name(clean_name):
    """
    Extracts a pretty display name and basic metadata from the filename-safe model string.
    """
    # Example: meta-llama_Meta-Llama-3.1-8B-Instruct -> meta-llama/Meta-Llama-3.1-8B-Instruct
    # Example: RedHatAI_Qwen3-14B-FP8-dynamic -> RedHatAI/Qwen3-14B-FP8-dynamic
    
    # Simple heuristic: replace first underscore with slash to restore Org/Model format
    if "_" in clean_name:
        display_name = clean_name.replace("_", "/", 1)
    else:
        display_name = clean_name
        
    return display_name

def analyze_benchmarks():
    results = {}

    if not BENCHMARKS_ROOT.exists():
        print(f"Error: {BENCHMARKS_ROOT} not found.")
        return

    print(f"Scanning {BENCHMARKS_ROOT}...")

    for folder_name, gpu_display_name in GPU_MAP.items():
        folder_path = BENCHMARKS_ROOT / folder_name
        if not folder_path.exists():
            print(f"Warning: {folder_path} not found, skipping.")
            continue
            
        print(f"Processing {gpu_display_name}...")
        
        # We only care about throughput.json files for "Tokens/s" comparison
        # And STRICTLY tp1 (Single GPU)
        for json_file in folder_path.glob("*_tp1_throughput.json"):
            try:
                data = json.loads(json_file.read_text())
            except json.JSONDecodeError:
                print(f"  Skipping invalid JSON: {json_file.name}")
                continue
                
            # Extract basic info from filename
            # Format: {model_clean}_tp1_throughput.json
            filename = json_file.name
            
            # Remove suffix to get model_clean
            model_clean = filename.replace("_tp1_throughput.json", "")
            
            # Get metric
            tokens_per_sec = data.get("tokens_per_second", 0)
            if not tokens_per_sec:
                continue

            # Store in results structure
            # Structure: results[model_clean] = { "display_name": "...", "gpus": { "AMD R9700": 123.4, ... } }
            
            if model_clean not in results:
                results[model_clean] = {
                    "model_name": parse_model_name(model_clean),
                    "model_clean": model_clean,
                    "gpus": {}
                }
            
            results[model_clean]["gpus"][gpu_display_name] = tokens_per_sec

    # Convert to list for easier frontend consumption
    final_output = []
    for model_key, info in results.items():
        # Only include if we have at least TWO data points for comparison
        # (User requested removal of single-GPU only models)
        if len(info["gpus"]) < 2:
            continue
            
        final_output.append(info)
            
    # Sort by model name for consistency
    final_output.sort(key=lambda x: x["model_name"])

    print(f"Found data for {len(final_output)} models.")
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_benchmarks()
