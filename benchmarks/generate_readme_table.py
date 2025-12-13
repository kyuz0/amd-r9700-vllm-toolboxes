
import json
from pathlib import Path

RESULTS_FILE = Path("max_context_results.json")

def format_context(val):
    if val is None or val == 0:
        return "Fail"
    if val >= 1000:
        return f"{val/1000:.0f}k"
    return str(val)

def main():
    if not RESULTS_FILE.exists():
        print(f"Error: {RESULTS_FILE} not found.")
        return

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    # Organize data: model -> tp -> seq -> list of (util, context)
    tree = {}
    for row in data:
        if row.get("status") != "success":
            continue
            
        model = row["model"]
        tp = row["tp"]
        util = float(row["util"])
        seqs = row["max_seqs"]
        context = row.get("max_context_1_user", 0)
        
        if model not in tree: tree[model] = {}
        if tp not in tree[model]: tree[model][tp] = {}
        if seqs not in tree[model][tp]: tree[model][tp][seqs] = []
        
        tree[model][tp][seqs].append((context, util))

    # Define headers
    # Moving Memory Utilization into the cells to allow per-concurrency variation
    print("\n**Table Key:** Cell values represent `Max Context Length (GPU Memory Utilization)`.\n")
    print("| Model | TP | 1 Req | 4 Reqs | 8 Reqs | 16 Reqs |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")

    model_order = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "openai/gpt-oss-20b",
        "RedHatAI/Qwen3-14B-FP8-dynamic",
        "cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit",
        "cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit",
        "RedHatAI/gemma-3-12b-it-FP8-dynamic",
        "RedHatAI/gemma-3-27b-it-FP8-dynamic",
    ]

    for model_name in model_order:
        if model_name not in tree:
            continue
            
        tps = sorted(tree[model_name].keys())
        
        for i, tp in enumerate(tps):
            row_cells = []
            
            # Model Name (only first row per model)
            if i == 0:
                row_cells.append(f"**`{model_name}`**")
            else:
                row_cells.append("")
                
            # TP
            row_cells.append(str(tp))
            
            # Concurrency Columns
            for seq in [1, 4, 8, 16]:
                candidates = tree[model_name][tp].get(seq, [])
                if not candidates:
                    row_cells.append("Fail")
                    continue
                
                # Sort criteria: 
                # 1. Maximize Context (desc)
                # 2. Minimize Utilization (asc) - tie breaker for same context
                best = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
                
                ctx_val = best[0]
                util_val = best[1]
                
                # Display format: "127k (0.98)"
                row_cells.append(f"{format_context(ctx_val)} ({util_val:.2f})")
            
            print("| " + " | ".join(row_cells) + " |")

if __name__ == "__main__":
    main()
