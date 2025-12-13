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

def get_verified_config(model_id, tp_size, max_seqs):
    """
    Reads max_context_results.json to find the best verified configuration.
    Returns dict: {'ctx': int, 'util': float}
    """
    default_config = {
        "ctx": int(MODEL_TABLE.get(model_id, {}).get("ctx", 8192)),
        "util": 0.90 # Safe default
    }
    
    if not RESULTS_FILE.exists():
        return default_config

    try:
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
            
        # Filter for Model + TP + Sequences
        matches = [r for r in data 
                  if r["model"] == model_id 
                  and r["tp"] == tp_size 
                  and r["max_seqs"] == max_seqs 
                  and r["status"] == "success"]
        
        if not matches:
            # Fallback 1: Try finding match with SAME TP but ANY Sequences (e.g. 1) to get base context?
            # Actually, safer to fallback to default or try finding nearest sequence?
            # Let's try finding exact match first. If fail, return default.
            return default_config
            
        # Sort by Util desc, then Context desc
        # We prefer higher utilization if available (performance), as long as it is verified success
        matches.sort(key=lambda x: (float(x["util"]), x["max_context_1_user"]), reverse=True)
        
        best = matches[0]
        return {
            "ctx": best["max_context_1_user"],
            "util": float(best["util"])
        }
        
    except Exception as e:
        return default_config

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

def nuke_vllm_cache():
    """Removes vLLM cache directory to fix potential graph/incompatibility issues."""
    cache = Path.home() / ".cache" / "vllm"
    if cache.exists():
        try:
            print(f"Clearing vLLM cache at {cache}...", end="", flush=True)
            subprocess.run(["rm", "-rf", str(cache)], check=True)
            cache.mkdir(parents=True, exist_ok=True)
            print(" Done.")
            time.sleep(1)
        except Exception as e:
            print(f" Failed: {e}")

def configure_and_launch(model_idx, gpu_count):
    model_id = MODELS_TO_RUN[model_idx]
    config = MODEL_TABLE[model_id]
    
    # Static Config
    valid_tps = config.get("valid_tp", [1])
    max_tp = max(valid_tps) if valid_tps else 1
    
    # Defaults
    current_tp = min(gpu_count, max_tp)
    current_seqs = 1 # Default to 1 concurrent user/request for stability
    
    # Initial Lookup
    verified = get_verified_config(model_id, current_tp, current_seqs)
    current_ctx = verified["ctx"]
    current_util = verified["util"]
    
    clear_cache = False
    use_eager = config.get("enforce_eager", False) # Default to model config, usually False
    
    name = model_id.split("/")[-1]
    
    while True:
        cache_status = "YES" if clear_cache else "NO"
        eager_status = "YES" if use_eager else "NO"
        
        menu_args = [
            "--clear", "--backtitle", f"AMD R9700 vLLM Launcher (GPUs: {gpu_count})",
            "--title", f"Configuration: {name}",
            "--menu", "Customize Launch Parameters:", "20", "65", "8",
            "1", f"Tensor Parallelism:   {current_tp}",
            "2", f"Concurrent Requests:  {current_seqs}",
            "3", f"Context Length:       {current_ctx} (Verified)",
            "4", f"GPU Utilization:      {current_util} (Verified)",
            "5", f"Erase vLLM Cache:     {cache_status}",
            "6", f"Force Eager Mode:     {eager_status}",
            "7", "LAUNCH SERVER"
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
                    # RE-CALCULATE Config
                    verified = get_verified_config(model_id, current_tp, current_seqs)
                    current_ctx = verified["ctx"]
                    current_util = verified["util"]
            
        elif choice == "2":
            # Max Seqs Selection
            new_seqs = run_dialog([
                "--title", "Concurrent Requests",
                "--menu", "Select Max Concurrent Requests:", "12", "40", "4",
                "1", "1 (Latency Focus)",
                "4", "4 (Balanced)",
                "8", "8 (Throughput)",
                "16", "16 (Max Load)"
            ])
            if new_seqs:
                current_seqs = int(new_seqs)
                # RE-CALCULATE Config based on new concurrency
                verified = get_verified_config(model_id, current_tp, current_seqs)
                current_ctx = verified["ctx"]
                current_util = verified["util"]

        elif choice == "3":
            # Configured Length Override
            new_ctx = run_dialog([
                "--title", "Context Length",
                "--inputbox", f"Override verified limit ({current_ctx}):", "10", "40", str(current_ctx)
            ])
            if new_ctx: current_ctx = int(new_ctx)

        elif choice == "4":
             # Util Override
             pass 

        elif choice == "5":
            # Toggle Cache
            if not clear_cache:
                # Enabling it -> Show Warning
                warn_msg = (
                    "WARNING: Erasing the vLLM cache will remove the compiled compute graphs.\n\n"
                    "This is useful if you are experiencing crashes, 'invalid graph' errors,\n"
                    "or have switched vLLM versions recently.\n\n"
                    "However, the next startup will take longer as graphs are re-compiled.\n\n"
                    "Are you sure you want to enable this?"
                )
                confirm = run_dialog([
                    "--title", "Erase Cache Warning", 
                    "--yesno", warn_msg, "12", "60"
                ])
                # dialog returns '0' string or empty string on some versions? 
                # Actually run_dialog returns content of stderr. 
                # Dialog --yesno returns exit code 0 for yes, 1 for no.
                # My run_dialog wrapper captures stderr but `subprocess.run(check=True)` raises error on non-zero exit code!
                # Wait, run_dialog implementation:
                # `subprocess.run(cmd, ..., check=True)`
                # If exit code is 1 (No/Cancel), it raises CalledProcessError.
                # So if it returns, it is YES.
                
                # Let's double check run_dialog implementation in the file view.
                # Lines 90-93:
                # try: subprocess.run(cmd, ..., check=True) ... return tf.read()
                # except CalledProcessError: return None
                
                # So if user selects YES (exit 0) -> returns empty string (stderr empty for yesno?) or something?
                # Actually dialog yesno does not write to stderr?
                # Whatever it returns, if it is not None, it is YES.
                # Let's verify that logic.
                
                # Actually, dialog --yesno output nothing to stderr. It communicates via exit code.
                # If logic is: check=True raises on exit code 1.
                # So YES = returns "" (empty string)
                # NO = raises -> returns None.
                
                # So:
                if confirm is not None:
                     clear_cache = True
            else:
                # Disabling it -> No warning needed
                clear_cache = False
             
        elif choice == "6":
            # Toggle Eager Mode
            use_eager = not use_eager
             
        elif choice == "7":
            # Launch
            break
            
    # Build Command
    subprocess.run(["clear"])
    
    if clear_cache:
        nuke_vllm_cache()
    
    cmd = [
        "vllm", "serve", model_id,
        "--host", HOST,
        "--port", PORT,
        "--tensor-parallel-size", str(current_tp),
        "--max-num-seqs", str(current_seqs),
        "--max-model-len", str(current_ctx),
        "--gpu-memory-utilization", str(current_util),
        "--dtype", "auto"
    ]
    
    if config.get("trust_remote"): cmd.append("--trust-remote-code")
    if use_eager: cmd.append("--enforce-eager")
    
    # Env Vars
    env = os.environ.copy()
    env.update(config.get("env", {}))
    
    print("\n" + "="*60)
    print(f" Launching: {name}")
    print(f" Config:    TP={current_tp} | Seqs={current_seqs} | Ctx={current_ctx} | Util={current_util}")
    if clear_cache:
        print(f" Action:    Clearing vLLM Cache (~/.cache/vllm)")
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
