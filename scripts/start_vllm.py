#!/usr/bin/env python3
import sys
import os
import json
import shutil
import tempfile
import subprocess
from pathlib import Path

# Add benchmarks dir to path to import config
# Add benchmarks dir to path to import config
SCRIPT_DIR = Path(__file__).parent.resolve()
BENCH_DIR = SCRIPT_DIR.parent / "benchmarks"
OPT_DIR = Path("/opt")

# Check /opt first (Container), then local fallback
if (OPT_DIR / "run_vllm_bench.py").exists():
    sys.path.append(str(OPT_DIR))
else:
    sys.path.append(str(BENCH_DIR))

try:
    from run_vllm_bench import MODEL_TABLE, MODELS_TO_RUN
except ImportError:
    print("Error: Could not import run_vllm_bench.py config.")
    sys.exit(1)

if (OPT_DIR / "max_context_results.json").exists():
    RESULTS_FILE = OPT_DIR / "max_context_results.json"
else:
    RESULTS_FILE = BENCH_DIR / "max_context_results.json"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = os.getenv("PORT", "8000")

def check_dependencies():
    if not shutil.which("dialog"):
        print("Error: 'dialog' is required. Please install it (apt-get install dialog).")
        sys.exit(1)

def detect_gpus():
    """Detects AMD GPUs via rocm-smi or /dev/dri."""
    try:
        # Try rocm-smi first
        res = subprocess.run(["rocm-smi", "--showid", "--csv"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            count = res.stdout.count("GPU")
            if count > 0: return count
    except: pass
    
    # Fallback to /dev/dri/render*
    try:
        return len(list(Path("/dev/dri").glob("renderD*")))
    except:
        return 1

def load_verified_context(model_id, tp_size):
    """
    Reads benchmarks/max_context_results.json to find the best Verified Limit.
    Returns default (from MODEL_TABLE) if no result found.
    """
    default_ctx = int(MODEL_TABLE.get(model_id, {}).get("ctx", 8192))
    
    if not RESULTS_FILE.exists():
        return default_ctx

    try:
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
            
        # Find results for this Model + exact TP (or lower TP if exact not found? No, exact is safer for verifying)
        # Actually, we want the limit for THIS hardware config.
        # Find best result where tp <= tp_size
        matches = [r for r in data if r["model"] == model_id and r["tp"] <= tp_size and r["status"] == "success"]
        
        if not matches:
            return default_ctx
            
        # Sort by verified length desc
        matches.sort(key=lambda x: (x["max_context_1_user"], x["tp"]), reverse=True)
        return matches[0]["max_context_1_user"]
        
    except Exception as e:
        return default_ctx

def run_dialog(args):
    """Runs dialog and returns stderr (selection)."""
    with tempfile.NamedTemporaryFile(mode="w+") as tf:
        cmd = ["dialog"] + args
        try:
            subprocess.run(cmd, stderr=tf, check=True)
            tf.seek(0)
            return tf.read().strip()
        except subprocess.CalledProcessError:
            return None # User cancelled

def configure_and_launch(model_idx, gpu_count):
    model_id = MODELS_TO_RUN[model_idx]
    config = MODEL_TABLE[model_id]
    
    # Static Config
    valid_tps = config.get("valid_tp", [1])
    max_tp = max(valid_tps) if valid_tps else 1
    
    # Default selection
    current_tp = min(gpu_count, max_tp)
    current_ctx = load_verified_context(model_id, current_tp)
    
    name = model_id.split("/")[-1]
    
    while True:
        menu_args = [
            "--clear", "--backtitle", f"AMD R9700 vLLM Launcher (GPUs: {gpu_count})",
            "--title", f"Configuration: {name}",
            "--menu", "Customize Launch Parameters:", "15", "60", "5",
            "1", f"Tensor Parallelism: {current_tp}",
            "2", f"Context Length:     {current_ctx}",
            "3", "LAUNCH SERVER"
        ]
        
        choice = run_dialog(menu_args)
        if not choice: return False # Back/Cancel
        
        if choice == "1":
            # TP Selection
            new_tp = run_dialog([
                "--title", "Tensor Parallelism",
                "--rangebox", f"Set TP Size (1-{max_tp})", "10", "40", "1", str(max_tp), str(current_tp)
            ])
            if new_tp: 
                new_tp_int = int(new_tp)
                if new_tp_int != current_tp:
                    current_tp = new_tp_int
                    # RE-CALCULATE Context for the new TP
                    current_ctx = load_verified_context(model_id, current_tp)
            
        elif choice == "2":
            # Context Selection
            new_ctx = run_dialog([
                "--title", "Context Length",
                "--inputbox", "Enter Max Context Length:", "10", "40", str(current_ctx)
            ])
            if new_ctx: current_ctx = int(new_ctx)
            
        elif choice == "3":
            # Launch
            break
            
    # Build Command
    subprocess.run(["clear"])
    cmd = [
        "vllm", "serve", model_id,
        "--host", HOST,
        "--port", PORT,
        "--tensor-parallel-size", str(current_tp),
        "--max-model-len", str(current_ctx),
        "--dtype", "auto"
    ]
    
    if config.get("trust_remote"): cmd.append("--trust-remote-code")
    if config.get("enforce_eager"): cmd.append("--enforce-eager")
    
    # Env Vars
    env = os.environ.copy()
    env.update(config.get("env", {}))
    
    print("\n" + "="*60)
    print(f" Launching: {name}")
    print(f" Command:   {' '.join(cmd)}")
    print("="*60 + "\n")
    
    os.execvpe("vllm", cmd, env)

def main():
    check_dependencies()
    gpu_count = detect_gpus()
    
    while True:
        # Build Model Menu
        menu_items = []
        for i, m_id in enumerate(MODELS_TO_RUN):
            name = m_id.split("/")[-1]
            # Pre-calc verified ctx for 'default' TP to show in menu? 
            # Or just show names. Just names is cleaner.
            config = MODEL_TABLE[m_id]
            menu_items.extend([str(i), name])
            
        choice = run_dialog([
            "--clear", "--backtitle", f"AMD R9700 vLLM Launcher (GPUs: {gpu_count})",
            "--title", "Select Model",
            "--menu", "Choose a model to serve:", "20", "60", "10"
        ] + menu_items)
        
        if not choice:
            subprocess.run(["clear"])
            print("Selection cancelled.")
            sys.exit(0)
            
        configure_and_launch(int(choice), gpu_count)

if __name__ == "__main__":
    main()
