
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
            # 1. Gather best raw results for each concurrency level
            best_by_seq = {}
            seq_levels = [1, 4, 8, 16]
            
            for seq in seq_levels:
                candidates = tree[model_name][tp].get(seq, [])
                if not candidates:
                    continue
                
                # Sort criteria: Maximize Context, then Minimize Util
                best = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
                best_by_seq[seq] = best

            # 2. Smooth/Backfill: Ensure Ctx(reqs=low) >= Ctx(reqs=high)
            # If 4 users can do 156k, 1 user certainly can too.
            # We take the standard of "Better" = (Higher Context, Lower Util)
            final_values = {}
            for i_seq, seq in enumerate(seq_levels):
                # Gather all valid results from this level and higher
                valid_futures = []
                for future_seq in seq_levels[i_seq:]:
                    if future_seq in best_by_seq:
                        valid_futures.append(best_by_seq[future_seq])
                
                if not valid_futures:
                    final_values[seq] = None
                else:
                    # Pick the best among them
                    # best tuple -> max context, then min util
                    # key function for max(): (context, -util)
                    final_values[seq] = max(valid_futures, key=lambda x: (x[0], -x[1]))

            # 3. Format Output
            row_cells = []
            
            # Model Name
            if i == 0:
                row_cells.append(f"**`{model_name}`**")
            else:
                row_cells.append("")
                
            # TP
            row_cells.append(str(tp))
            
            # Concurrency Columns
            for seq in seq_levels:
                val = final_values.get(seq)
                if val is None:
                    row_cells.append("Fail")
                else:
                    ctx_val, util_val = val
                    row_cells.append(f"{format_context(ctx_val)} ({util_val:.2f})")
            
            print("| " + " | ".join(row_cells) + " |")

if __name__ == "__main__":
    main()
