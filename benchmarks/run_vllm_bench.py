#!/usr/bin/env python3
import subprocess, time, json, sys, os, requests, re, argparse
from pathlib import Path

# =========================
# ⚙️ GLOBAL SETTINGS
# =========================

# HARDWARE: 2x AMD Radeon AI PRO R9700 (32GB, RDNA 4)
GPU_UTIL = "0.98" 
PORT     = 8000
HOST     = "127.0.0.1"

# BENCHMARK TOGGLES
# AITER is disabled/removed.
 

# 1. THROUGHPUT CONFIG
OFF_NUM_PROMPTS      = 1000 
OFF_FORCED_OUTPUT    = "512"
# Default fallback if not specified in MODEL_TABLE
DEFAULT_BATCH_TOKENS = "8192"

# 2. LATENCY CONFIG
SRV_DURATION    = 180    
QPS_SWEEP       = [1.0, 4.0] 

# Fallbacks
FALLBACK_INPUT_LEN  = 1024
FALLBACK_OUTPUT_LEN = 512

RESULTS_DIR = Path("benchmark_results")
RESULTS_DIR.mkdir(exist_ok=True)

# =========================
# 🛠️ MODEL CONFIGURATION 🛠️
# =========================

MODEL_TABLE = {
    # 1. Llama 3.1 8B Instruct
    # MAD uses 131k tokens. We scale to 32k for 32GB VRAM safety.
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "ctx": "65536",  
        "trust_remote": False,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "32768" 
    },
    
    # 2. GPT-OSS 20B (MXFP4)
    # MAD Row 0 uses 8192. We match this exactly.
    "openai/gpt-oss-20b": {
        "ctx": "32768", 
        "trust_remote": True,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "8192"
    },

    # 3. Qwen 14B FP8
    # MAD uses 40k. We use 32k.
    "RedHatAI/Qwen3-14B-FP8-dynamic": {
        "ctx": "32768", 
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "64",
        "max_tokens": "32768"
    },

    # 4. Qwen 30B 4-bit
    "cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit": {
        "ctx": "24576", 
        "trust_remote": True,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "32768"
    },

    # 5. Qwen 80B AWQ (The Big One) [NEW]
    # Size: ~48GB. Fits on 2x32GB (64GB). Leftover for Cache: ~16GB.
    # Config: 20k ctx fits in that cache. Eager mode required for stability.
    "cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit": {
        "ctx": "20480", 
        "trust_remote": True,
        "valid_tp": [2], # Too big for single GPU
        "max_num_seqs": "32", # Lower concurrency for safety
        "max_tokens": "16384", # Lower batch size because Eager mode is CPU intensive
        "enforce_eager": False, 
        "env": {"VLLM_USE_TRITON_AWQ": "1"} # Fixes "Unsupported Hardware" error
    },

    # 76 Gemma 3 27B FP8
    "RedHatAI/gemma-3-27b-it-FP8-dynamic": {
        "ctx": "29000",
        "trust_remote": True,
        "valid_tp": [2],
        "max_num_seqs": "32",
        "max_tokens": "29000",
        "gpu_util": "0.94",
    },

    # 7. Gemma 3 12B FP8
    "RedHatAI/gemma-3-12b-it-FP8-dynamic": {
        "ctx": "9900",
        "trust_remote": True,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "9900",
    },
}

MODELS_TO_RUN = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "openai/gpt-oss-20b",
    "RedHatAI/Qwen3-14B-FP8-dynamic",
    "cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit",
    "cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit",
    "RedHatAI/gemma-3-27b-it-FP8-dynamic",
    "RedHatAI/gemma-3-12b-it-FP8-dynamic",
]

# =========================
# UTILS
# =========================

def log(msg): print(f"\n[BENCH] {msg}")

def get_gpu_count():
    try:
        # Using rocm-smi --showid to list GPUs. 
        # Output format: "GPU[0] : Device Name: ..."
        res = subprocess.run(["rocm-smi", "--showid"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            # Filter specifically for the target GPU as requested
            target_gpu = "AMD Radeon AI PRO R9700"
            count = 0
            for line in res.stdout.strip().split('\n'):
                if "Device Name" in line and target_gpu in line:
                    count += 1
            
            return count if count > 0 else 1
        else:
            log("rocm-smi failed, defaulting to 2 GPUs (Hardcoded Fallback)")
            return 2
    except Exception as e:
        log(f"Error detecting GPUs: {e}, defaulting to 2 GPUs")
        return 2

def kill_vllm():
    subprocess.run("pgrep -f 'vllm serve' | xargs -r kill -9", 
                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)

def nuke_vllm_cache():
    cache = Path.home() / ".cache" / "vllm"
    if cache.exists():
        try:
            subprocess.run(["rm", "-rf", str(cache)], check=True)
            cache.mkdir(parents=True, exist_ok=True)
            time.sleep(2)
        except: pass

def get_dataset():
    data_path = Path("ShareGPT_V3_unfiltered_cleaned_split.json")
    if data_path.exists(): return str(data_path)
    
    log("Downloading ShareGPT dataset...")
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    try:
        r = requests.get(url, stream=True, timeout=15)
        r.raise_for_status()
        with open(data_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return str(data_path)
    except Exception as e:
        log(f"WARNING: ShareGPT download failed ({e}). using RANDOM.")
        return None

def wait_for_server(url, process, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        if process.poll() is not None:
            log(f"CRITICAL: Server died! Ret: {process.returncode}")
            return False
        try:
            if requests.get(f"{url}/v1/models", timeout=2).status_code == 200:
                log("Server ready. Stabilizing...")
                time.sleep(5)
                return True
        except: pass
        time.sleep(2)
    return False

def get_model_args(model, tp_size):
    config = MODEL_TABLE.get(model, {"ctx": "8192", "max_num_seqs": "32"})
    
    # Allow per-model GPU utilization override
    util = config.get("gpu_util", GPU_UTIL)

    cmd = [
        "--model", model,
        "--gpu-memory-utilization", util,
        "--max-model-len", config["ctx"], 
        "--dtype", "auto",
        "--tensor-parallel-size", str(tp_size),
        "--max-num-seqs", config["max_num_seqs"]
    ]
    
    if config.get("trust_remote"): cmd.append("--trust-remote-code")
    if config.get("enforce_eager"): cmd.append("--enforce-eager")
    
    return cmd

def run_throughput(model, tp_size):
    if tp_size not in MODEL_TABLE[model]["valid_tp"]: return
    
    model_safe = model.replace("/", "_")
    output_file = RESULTS_DIR / f"{model_safe}_tp{tp_size}_throughput.json"
    
    if output_file.exists():
        log(f"SKIP Throughput {model} (TP={tp_size})")
        return

    dataset_path = get_dataset()
    dataset_args = ["--dataset-name", "sharegpt", "--dataset-path", dataset_path] if dataset_path else ["--input-len", "1024"]
    
    # Retrieve Model-Specific Batch Tokens
    batch_tokens = MODEL_TABLE[model].get("max_tokens", DEFAULT_BATCH_TOKENS)

    log(f"START Throughput {model} (TP={tp_size}) [Batch: {batch_tokens}]...")
    kill_vllm()
    nuke_vllm_cache()

    cmd = ["vllm", "bench", "throughput"] + get_model_args(model, tp_size)
    cmd.extend([
        "--num-prompts", str(OFF_NUM_PROMPTS),
        "--max-num-batched-tokens", batch_tokens,
        "--output-len", OFF_FORCED_OUTPUT,
        "--output-json", str(output_file),
        "--disable-log-stats"
    ])
    cmd.extend(dataset_args)

    # ENV Setup: Global + Model Specific
    env = os.environ.copy()
    
    # Inject model specific env vars (e.g. for AWQ)
    model_env = MODEL_TABLE[model].get("env", {})
    env.update(model_env)

    try: 
        subprocess.run(cmd, check=True, env=env)
    except: 
        log(f"ERROR: Throughput failed {model}")

def run_latency(model, tp_size):
    if tp_size not in MODEL_TABLE[model]["valid_tp"]: return
    model_safe = model.replace("/", "_")

    if all((RESULTS_DIR / f"{model_safe}_tp{tp_size}_qps{q}_latency.json").exists() for q in QPS_SWEEP):
        return

    dataset_path = get_dataset()
    log(f"START Server {model} (TP={tp_size})...")
    kill_vllm()
    nuke_vllm_cache()

    srv_log = open(RESULTS_DIR / f"{model_safe}_tp{tp_size}_server.log", "w")
    srv_args = [x for x in get_model_args(model, tp_size) if x != "--model" and x != model]
    
    # ENV Setup: Global + Model Specific
    env = os.environ.copy()

    model_env = MODEL_TABLE[model].get("env", {})
    env.update(model_env)

    proc = subprocess.Popen(["vllm", "serve", model] + srv_args + ["--host", HOST, "--port", str(PORT)], 
                            stdout=srv_log, stderr=srv_log, env=env)

    try:
        if not wait_for_server(f"http://{HOST}:{PORT}", proc): return

        for qps in QPS_SWEEP:
            out_file = RESULTS_DIR / f"{model_safe}_tp{tp_size}_qps{qps}_latency.json"
            if out_file.exists(): continue
            
            log(f"BENCH QPS={qps}...")
            bench_cmd = [
                "vllm", "bench", "serve",
                "--model", model,
                "--base-url", f"http://{HOST}:{PORT}",
                "--request-rate", str(qps),
                "--num-prompts", str(int(max(10, SRV_DURATION * qps))),
                "--trust-remote-code"
            ]
            
            if dataset_path: bench_cmd.extend(["--dataset-name", "sharegpt", "--dataset-path", dataset_path])
            else: bench_cmd.extend(["--dataset-name", "random", "--random-input-len", "1024", "--random-output-len", "512"])

            res = subprocess.run(bench_cmd, capture_output=True, text=True, env=env)
            with open(out_file, "w") as f:
                f.write(json.dumps({"success": res.returncode==0, "raw_output": res.stdout}, indent=2))

    except Exception as e: log(f"CRASH: {e}")
    finally:
        proc.terminate()
        kill_vllm()

def print_summary(tps):
    print(f"\n{'MODEL':<40} | {'TP':<2} | {'TOK/S':<8} | {'QPS':<4} | {'TTFT':<6} | {'TPOT':<6}")
    print("-" * 105)
    
    for m in MODELS_TO_RUN:
        msafe = m.replace("/", "_")
        for tp in tps:
            if tp not in MODEL_TABLE[m]["valid_tp"]: continue
            
            try: 
                tdata = json.loads((RESULTS_DIR / f"{msafe}_tp{tp}_throughput.json").read_text())
                tok_s = f"{tdata.get('tokens_per_second', 0):.1f}"
            except: tok_s = "N/A"

            first_row = True
            for q in QPS_SWEEP:
                try:
                    ldata = json.loads((RESULTS_DIR / f"{msafe}_tp{tp}_qps{q}_latency.json").read_text())
                    raw = ldata["raw_output"]
                    ttft = re.search(r"(?:Mean TTFT|TTFT).*?([\d\.]+)", raw).group(1)
                    tpot = re.search(r"(?:Mean TPOT|TPOT).*?([\d\.]+)", raw).group(1)
                except: ttft, tpot = "-", "-"
                
                name_cell = m.split('/')[-1] if (first_row and q == QPS_SWEEP[0]) else ""
                
                print(f"{name_cell:<40} | {tp:<2} | {tok_s:<8} | {q:<4} | {ttft:<6} | {tpot:<6}")
                first_row = False
            print("-" * 105)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    
    gpu_count = get_gpu_count()
    log(f"Detected {gpu_count} AMD GPU(s)")
    
    valid_tp_args = [t for t in args.tp if t <= gpu_count]
    if not valid_tp_args:
        log(f"Requested TP={args.tp} but only {gpu_count} GPU(s) detected. Nothing to run.")
        sys.exit(0)

    kill_vllm()
    for tp in valid_tp_args:
        for m in MODELS_TO_RUN:
            run_throughput(m, tp)
            run_latency(m, tp)
    print_summary(valid_tp_args)