
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
    
    if not BENCHMARK_DIR.exists():
        print(f"Error: {BENCHMARK_DIR} does not exist!")
        return []

    print(f"Scanning {BENCHMARK_DIR}...")
    
    # Files are flat in the dir: {model_safe}_tp{tp}_{type}.json
    # or latency: {model_safe}_tp{tp}_qps{q}_latency.json
    
    # We need to group by (model, tp) to form cohesive records if we want, 
    # BUT the webapp expects a list of "runs".
    # Looking at the example JSON, each "run" is a single test point (e.g. "pp2048 @ d16384" OR "tg32 @ d16384")
    # Actually, looking at the provided valid example:
    # "test": "pp512", "tps_mean": 2708.86 ...
    
    # Our data:
    # throughput.json -> tokens_per_second. This is usually "decoding" or a mix?
    # vLLM bench throughput usually streams tokens. 
    # Let's look at what run_vllm_bench.py produces.
    # Throughput: --input-len 1024 --output-len 512.
    # This is effectively a mixed batch. 
    # We'll label it "Throughput (1024/512)" or just "Throughput"
    
    # Latency: qps-based.
    
    files = list(BENCHMARK_DIR.glob("*.json"))
    
    for f in files:
        fname = f.name
        try:
            data = json.loads(f.read_text())
        except:
            print(f"Skipping bad JSON: {fname}")
            continue

        # Infer metadata from filename
        # Format: {model_safe}_tp{tp}_{suffix}
        # Suffix can be: "throughput.json" or "qps{q}_latency.json"
        
        # We need model name. The script replaces / with _ in filenames.
        # But we verify against the known models list? Or just parse string.
        # We can reconstruct roughly.
        
        # Split by "_tp" which is a strong delimiter
        parts = fname.split("_tp")
        if len(parts) < 2: continue
        
        model_part = parts[0]
        rest = parts[1] # "1_throughput.json" or "2_qps1.0_latency.json"
        
        # TP
        tp_match = re.match(r"^(\d+)", rest)
        if not tp_match: continue
        tp = int(tp_match.group(1))
        
        # Env mapping
        env = f"TP{tp}"
        
        # Model Name Restoration (best effort or matching)
        # In the script: model.replace("/", "_")
        # We can reverse this if we have the list, but for now let's just use the clean string?
        # The webapp uses "model_clean" and "model".
        # Let's assume standard "org_model" format -> "org/model"
        if "_" in model_part:
            # Heuristic: First _ is likely the slash
            model_display = model_part.replace("_", "/", 1)
        else:
            model_display = model_part
            
        params_b, quant = extract_meta(model_display)
        
        base_run = {
            "model": model_display,
            "model_clean": model_display,
            "env": env,
            "gpu_config": "dual" if tp > 1 else "single",
            "quant": quant,
            "params_b": params_b,
            "name_params_b": params_b,
            # Defaults
            "backend": "vLLM", 
            "error": False
        }

        if "throughput" in fname:
            # Throughput run
            # data has "tokens_per_second"
            tps = data.get("tokens_per_second", 0)
            
            run = base_run.copy()
            run["test"] = "Throughput"
            run["tps_mean"] = tps
            # If tps is 0 or missing, it might be an error?
            if tps == 0 and "error" in str(data).lower():
                run["error"] = True
            
            runs.append(run)

        elif "latency" in fname:
            # Latency run
            # raw_output has strings like "Mean TTFT: 12.3 ms", "Mean TPOT: 45.6 ms"
            raw = data.get("raw_output", "")
            qps_match = re.search(r"_qps([\d\.]+)_", fname)
            qps = qps_match.group(1) if qps_match else "?"
            
            # Extract metrics
            ttft = 0.0
            tpot = 0.0
            
            ttft_m = re.search(r"(?:Mean TTFT|TTFT).*?([\d\.]+)", raw)
            if ttft_m: ttft = float(ttft_m.group(1))
            
            tpot_m = re.search(r"(?:Mean TPOT|TPOT).*?([\d\.]+)", raw)
            if tpot_m: tpot = float(tpot_m.group(1))
            
            # We create TWO entries? Or how does the webapp handle multiple metrics?
            # Example webapp table columns are "Backends" showing ONE value.
            # But grouping is by "Test". 
            # So we can have a test called "TTFT (QPS 1.0)" and "TPOT (QPS 1.0)"
            
            # Entry 1: TTFT
            r1 = base_run.copy()
            r1["test"] = f"TTFT @ QPS {qps}"
            r1["tps_mean"] = ttft # Using tps_mean field for the numeric value
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
