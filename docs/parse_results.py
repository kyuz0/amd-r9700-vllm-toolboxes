
import os
import json
import re
from pathlib import Path

# Config
BENCHMARK_DIR = Path("../benchmarks/benchmark_results_amd-r9700")
OUTPUT_FILE = Path("results.json")

# Regex to parse model name for quantization and parameters
# Examples: 
# "meta-llama/Meta-Llama-3.1-8B-Instruct"
# "cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit"
# "RedHatAI/Llama-3.1-8B-Instruct-FP8-block"
PARAMS_REGEX = r"(\d+(?:\.\d+)?)B"
QUANT_REGEX = r"(FP8|AWQ|GPTQ|BF16|4bit|Int4)"

def extract_meta(model_name):
    # Params
    params_match = re.search(PARAMS_REGEX, model_name, re.IGNORECASE)
    params_b = float(params_match.group(1)) if params_match else None
    
    # Quant
    quant_match = re.search(QUANT_REGEX, model_name, re.IGNORECASE)
    quant = quant_match.group(1).upper() if quant_match else "BF16" # Default assumption if no tag? Or unknown.
    # Refine quant if 4bit
    if quant == "4BIT" or quant == "INT4":
        if "GPTQ" in model_name: quant = "GPTQ-4bit"
        elif "AWQ" in model_name: quant = "AWQ-4bit"
        else: quant = "4-bit"

    return params_b, quant

def parse_logs():
    runs = []
    
    # Define directories and their tags
    # (Path, variant_tag)
    dirs = [
        (BENCHMARK_DIR, "default"),
        (Path("../benchmarks/benchmark_results_amd-r9700-rocm_atten"), "rocm")
    ]
    
    for b_dir, variant in dirs:
        if not b_dir.exists():
            print(f"Warning: {b_dir} does not exist, skipping.")
            continue

        print(f"Scanning {b_dir} [{variant}]...")
        
        files = list(b_dir.glob("*.json"))
        
        for f in files:
            fname = f.name
            try:
                data = json.loads(f.read_text())
            except:
                print(f"Skipping bad JSON: {fname}")
                continue

            # Infer metadata from filename
            parts = fname.split("_tp")
            if len(parts) < 2: continue
            
            model_part = parts[0]
            rest = parts[1] 
            
            # TP
            tp_match = re.match(r"^(\d+)", rest)
            if not tp_match: continue
            tp = int(tp_match.group(1))
            
            env = f"TP{tp}"
            
            # Model Name Restoration
            if "_" in model_part:
                model_display = model_part.replace("_", "/", 1)
            else:
                model_display = model_part
                
            params_b, quant = extract_meta(model_display)
            
            base_run = {
                "model": model_display,
                "model_clean": model_display,
                "env": env,
                "variant": variant,
                "gpu_config": "dual" if tp > 1 else "single",
                "quant": quant,
                "params_b": params_b,
                "name_params_b": params_b,
                "backend": "vLLM", 
                "error": False
            }

            if "throughput" in fname:
                tps = data.get("tokens_per_second", 0)
                run = base_run.copy()
                run["test"] = "Throughput"
                run["tps_mean"] = tps
                if tps == 0 and "error" in str(data).lower():
                    run["error"] = True
                runs.append(run)

            elif "latency" in fname:
                raw = data.get("raw_output", "")
                qps_match = re.search(r"_qps([\d\.]+)_", fname)
                qps = qps_match.group(1) if qps_match else "?"
                
                ttft_m = re.search(r"(?:Mean TTFT|TTFT).*?([\d\.]+)", raw)
                ttft = float(ttft_m.group(1)) if ttft_m else 0.0
                
                tpot_m = re.search(r"(?:Mean TPOT|TPOT).*?([\d\.]+)", raw)
                tpot = float(tpot_m.group(1)) if tpot_m else 0.0
                
                # Entry 1: TTFT
                r1 = base_run.copy()
                r1["test"] = f"TTFT @ QPS {qps}"
                r1["tps_mean"] = ttft 
                runs.append(r1)
                
                # Entry 2: TPOT
                r2 = base_run.copy()
                r2["test"] = f"TPOT @ QPS {qps}"
                r2["tps_mean"] = tpot
                runs.append(r2)

    return runs

if __name__ == "__main__":
    data = {"runs": parse_logs()}
    
    runs_count = len(data["runs"])
    print(f"Parsed {runs_count} runs.")
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Written to {OUTPUT_FILE}")
