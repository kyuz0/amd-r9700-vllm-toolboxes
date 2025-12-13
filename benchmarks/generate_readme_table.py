
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

    # Organize data: model -> tp -> util -> seqs -> result
    tree = {}
    for row in data:
        model = row["model"]
        tp = row["tp"]
        util = row["util"]
        seqs = row["max_seqs"]
        
        if model not in tree: tree[model] = {}
        if tp not in tree[model]: tree[model][tp] = {}
        if util not in tree[model][tp]: tree[model][tp][util] = {}
        
        tree[model][tp][util][seqs] = row

    print("| Model | GPUs (TP) | Mem Util | Context Capacity (1 / 4 / 8 / 16 Concurrency) |")
    print("| :--- | :--- | :--- | :--- |")

    # Order models if needed, or just iterate
    # Let's try to keep the order somewhat consistent with the file or predefined
    # For now, simply sorted by name is okay, or we can use a list
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
            utils = tree[model_name][tp]
            
            # Select best util
            # Criteria: Maximize number of successful runs.
            # Tie-breaker: Maximize util value.
            best_util = None
            best_score = -1
            
            for u_val in utils:
                u_data = utils[u_val]
                # Score = count of successful runs
                score = sum(1 for s in [1, 4, 8, 16] 
                           if s in u_data and u_data[s]["status"] == "success")
                
                if score > best_score:
                    best_score = score
                    best_util = u_val
                elif score == best_score:
                    if best_util is None or float(u_val) > float(best_util):
                        best_util = u_val
            
            if best_util is None:
                continue

            # Prepare row data
            row_data = utils[best_util]
            
            # Extract context for each seq step
            ctx_strs = []
            for seq in [1, 4, 8, 16]:
                if seq in row_data and row_data[seq]["status"] == "success":
                    ctx = row_data[seq].get("max_context_1_user", 0)
                    ctx_strs.append(format_context(ctx))
                else:
                    ctx_strs.append("Fail")
            
            ctx_col = " / ".join(ctx_strs)
            
            # Formatting
            model_cell = f"**`{model_name}`**" if i == 0 else ""
            
            print(f"| {model_cell} | {tp} | {best_util} | {ctx_col} |")


if __name__ == "__main__":
    main()
