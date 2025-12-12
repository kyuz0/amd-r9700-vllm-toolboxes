#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

RESULTS_FILE = Path("max_context_results.json")
REPORT_FILE = Path("max_context_report.md")

def generate_report(results_file, report_file):
    if not results_file.exists():
        print(f"Error: Results file {results_file} not found.")
        return

    try:
        with open(results_file, "r") as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error reading results file: {e}")
        return

    with open(report_file, "w") as f:
        f.write("# AMD R9700 vLLM Context Limits\n\n")
        f.write("This table shows the **Maximum Working Context** found for each configuration.\n")
        f.write("- **Target Limit**: The max context length we requested vLLM to initialize with.\n")
        f.write("- **True Capacity**: The actual space (tokens) available in GPU memory (KV cache).\n")
        f.write("- **Verified Limit**: The usable context size (limited by either Target or Capacity), verified by a test PROMPT.\n\n")
        
        f.write("| Model | TP | Util | Seqs | Target Limit | True Capacity | **Verified Limit** | Error |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        
        # Sort results: Model -> TP -> Util -> Seqs
        results.sort(key=lambda x: (x["model"], x["tp"], str(x["util"]), x["max_seqs"]))

        for r in results:
            if r["status"] == "success":
                cap = r["real_capacity"]
                c1 = r["max_context_1_user"]
                target = r["configured_len"]
                f.write(f"| {r['model']} | {r['tp']} | {r['util']} | {r['max_seqs']} | {target} | {cap} | **{c1}** | - |\n")
            else:
                target = r.get("configured_len", "-")
                err = r.get("error", "-").replace("\n", " ").replace("|", "/") # Escape MD table chars
                f.write(f"| {r['model']} | {r['tp']} | {r['util']} | {r['max_seqs']} | {target} | ERROR | - | {err} |\n")

    print(f"Report generated at {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=str(RESULTS_FILE), help="Path to input JSON results")
    parser.add_argument("--out", type=str, default=str(REPORT_FILE), help="Path to output Markdown file")
    args = parser.parse_args()
    
    generate_report(Path(args.json), Path(args.out))
