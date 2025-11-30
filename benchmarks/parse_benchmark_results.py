import os
import json
import re
import argparse
from pathlib import Path

def parse_log_file(file_path):
    """Parses a single vLLM benchmark log file."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract metrics using regex
    # Example output line: 
    # "Throughput: 12.34 tokens/s, 5.67 requests/s"
    throughput_match = re.search(r"Throughput:\s+([\d\.]+)\s+tokens/s", content)
    req_rate_match = re.search(r"Throughput:.*,\s+([\d\.]+)\s+requests/s", content)
    
    # Extract config from filename or content if possible, but filename is reliable based on our script
    # Filename format: {model_name}_{scenario_name}.log
    filename = os.path.basename(file_path)
    name_parts = filename.replace('.log', '').split('_')
    
    # Heuristic to separate model name from scenario
    # We know scenarios are: standard, long_context, throughput
    # Everything before the scenario name is the model name
    scenario_map = {
        'standard': 'Standard (512/128)',
        'long': 'Long Context (16384/128)', # 'long_context' splits to 'long', 'context'
        'throughput': 'Throughput (128/128)'
    }
    
    scenario_key = None
    model_name_parts = []
    
    # Work backwards to find the scenario
    if 'standard' in filename:
        scenario_key = 'standard'
        model_name = filename.split('_standard')[0]
    elif 'long_context' in filename:
        scenario_key = 'long'
        model_name = filename.split('_long_context')[0]
    elif 'throughput' in filename:
        scenario_key = 'throughput'
        model_name = filename.split('_throughput')[0]
    else:
        # Fallback
        model_name = "Unknown"
        scenario_key = "Unknown"

    scenario_display = scenario_map.get(scenario_key, scenario_key)

    if throughput_match:
        return {
            "model": model_name,
            "scenario": scenario_display,
            "throughput_tokens_per_sec": float(throughput_match.group(1)),
            "requests_per_sec": float(req_rate_match.group(1)) if req_rate_match else 0.0,
            "filename": filename
        }
    else:
        print(f"Warning: Could not parse metrics from {filename}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Parse vLLM benchmark logs.")
    parser.add_argument("--log-dir", default="results/logs", help="Directory containing log files")
    parser.add_argument("--output", default="docs/data.json", help="Output JSON file path")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    results = []

    if not log_dir.exists():
        print(f"Error: Log directory '{log_dir}' does not exist.")
        return

    print(f"Scanning {log_dir} for log files...")
    for log_file in log_dir.glob("*.log"):
        data = parse_log_file(log_file)
        if data:
            results.append(data)

    # Sort results by model then scenario
    results.sort(key=lambda x: (x['model'], x['scenario']))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Successfully parsed {len(results)} logs. Data saved to {output_path}")

if __name__ == "__main__":
    main()
