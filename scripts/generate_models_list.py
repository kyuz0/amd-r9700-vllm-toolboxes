#!/usr/bin/env python3
import sys
import json
import os
from pathlib import Path

# Add benchmarks dir to path to import config
SCRIPT_DIR = Path(__file__).parent.resolve()
BENCH_DIR = SCRIPT_DIR.parent / "benchmarks"
sys.path.append(str(BENCH_DIR))

try:
    from run_vllm_bench import MODEL_TABLE, MODELS_TO_RUN
except ImportError:
    # Fallback if run_vllm_bench not found
    MODEL_TABLE = {}
    MODELS_TO_RUN = []

RESULTS_FILE = BENCH_DIR / "max_context_results.json"

def get_best_context(model_id, max_tp):
    """
    Finds the maximum verified context for the given model
    that fits within max_tp (system limit).
    """
    if not RESULTS_FILE.exists():
        # Fallback to configured ctx in MODEL_TABLE
        return int(MODEL_TABLE.get(model_id, {}).get("ctx", 8192))
        
    try:
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
    except:
        return 8192

    best_ctx = 0
    
    # Filter for this model
    candidates = [r for r in data if r["model"] == model_id and r["status"] == "success"]
    
    # Filter by TP <= max_tp (we can't launch TP2 on 1 GPU)
    # But we WANT the limit for the Highest Allowable TP.
    valid_candidates = [r for r in candidates if r["tp"] <= max_tp]
    
    if not valid_candidates:
         # Fallback to hardcoded table
         return int(MODEL_TABLE.get(model_id, {}).get("ctx", 8192))

    # Sort by Context Length (Descending) -> Then TP (Descending)
    # This ensures we pick the biggest context possible on the hardware.
    valid_candidates.sort(key=lambda x: (x["max_context_1_user"], x["tp"]), reverse=True)
    
    return valid_candidates[0]["max_context_1_user"]

def main():
    if len(sys.argv) > 1:
        gpu_count = int(sys.argv[1])
    else:
        gpu_count = 1

    for model_id in MODELS_TO_RUN:
        config = MODEL_TABLE.get(model_id, {})
        
        # 1. Name: Use cleaner name
        name = model_id.split("/")[-1]
        
        # 2. Repo: model_id
        
        # 3. MaxTP: Min of (Model valid tp max, System GPU Count)
        valid_tps = config.get("valid_tp", [1])
        model_max_tp = max(valid_tps) if valid_tps else 1
        
        # We cap the reported MaxTP at the system limit for the UI rangebox
        # But for finding the context, we look at what is POSSIBLY supported.
        # Actually, for the UI, we should only show what is switchable.
        ui_max_tp = min(model_max_tp, gpu_count)
        
        if ui_max_tp < 1: ui_max_tp = 1 # Safety
        
        # 4. MaxCtx: Get from Results for this UI_MAX_TP
        ctx = get_best_context(model_id, ui_max_tp)
        
        # 5. Flags
        flags = []
        if config.get("trust_remote"): flags.append("--trust-remote-code")
        if config.get("enforce_eager"): flags.append("--enforce-eager")
        flags_str = " ".join(flags)
        
        # 6. EnvVars
        env_dict = config.get("env", {})
        envs_str = " ".join([f"{k}={v}" for k,v in env_dict.items()])
        
        # Format: "Name|Repo|MaxCtx|MaxTP|Flags|EnvVars"
        print(f"{name}|{model_id}|{ctx}|{ui_max_tp}|{flags_str}|{envs_str}")

if __name__ == "__main__":
    main()
