# AMD R9700 vLLM Context Limits

This table shows the **Maximum Working Context** found for each configuration.
- **Target Limit**: The max context length we requested vLLM to initialize with.
- **True Capacity**: The actual space (tokens) available in GPU memory (KV cache).
- **Verified Limit**: The usable context size (limited by either Target or Capacity), verified by a test PROMPT.

| Model | TP | Util | Seqs | Target Limit | True Capacity | **Verified Limit** | Error |
|---|---|---|---|---|---|---|---|
| RedHatAI/Qwen3-14B-FP8-dynamic | 1 | 0.98 | 1 | 40960 | 99024 | **40960** | - |
| RedHatAI/Qwen3-14B-FP8-dynamic | 1 | 0.98 | 4 | 40960 | 98960 | **40960** | - |
| RedHatAI/Qwen3-14B-FP8-dynamic | 1 | 0.98 | 8 | 40960 | 98928 | **40960** | - |
| RedHatAI/Qwen3-14B-FP8-dynamic | 1 | 0.98 | 16 | 40960 | 98928 | **40960** | - |
| RedHatAI/Qwen3-14B-FP8-dynamic | 2 | 0.95 | 1 | 40960 | 287168 | **40960** | - |
| RedHatAI/Qwen3-14B-FP8-dynamic | 2 | 0.95 | 4 | 40960 | 287040 | **40960** | - |
| RedHatAI/Qwen3-14B-FP8-dynamic | 2 | 0.95 | 8 | 40960 | 287040 | **40960** | - |
| RedHatAI/Qwen3-14B-FP8-dynamic | 2 | 0.95 | 16 | 40960 | 286688 | **40960** | - |
| RedHatAI/Qwen3-14B-FP8-dynamic | 2 | 0.98 | 1 | 0 | ERROR | - | Verification Failed |
| RedHatAI/gemma-3-12b-it-FP8-dynamic | 1 | 0.98 | 1 | 8192 | 44288 | **8192** | - |
| RedHatAI/gemma-3-12b-it-FP8-dynamic | 1 | 0.98 | 4 | 8192 | 44768 | **8192** | - |
| RedHatAI/gemma-3-12b-it-FP8-dynamic | 1 | 0.98 | 8 | 8192 | 44768 | **8192** | - |
| RedHatAI/gemma-3-12b-it-FP8-dynamic | 1 | 0.98 | 16 | 8192 | 44768 | **8192** | - |
| RedHatAI/gemma-3-12b-it-FP8-dynamic | 2 | 0.98 | 1 | 8192 | 124784 | **8192** | - |
| RedHatAI/gemma-3-12b-it-FP8-dynamic | 2 | 0.98 | 4 | 8192 | 126256 | **8192** | - |
| RedHatAI/gemma-3-12b-it-FP8-dynamic | 2 | 0.98 | 8 | 8192 | 126256 | **8192** | - |
| RedHatAI/gemma-3-12b-it-FP8-dynamic | 2 | 0.98 | 16 | 8192 | 126256 | **8192** | - |
| RedHatAI/gemma-3-27b-it-FP8-dynamic | 2 | 0.95 | 1 | 8192 | 56112 | **8192** | - |
| RedHatAI/gemma-3-27b-it-FP8-dynamic | 2 | 0.95 | 4 | 0 | ERROR | - | Verification Failed |
| RedHatAI/gemma-3-27b-it-FP8-dynamic | 2 | 0.98 | 1 | 0 | ERROR | - | Verification Failed |
| cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit | 1 | 0.98 | 1 | 150560 | 157312 | **150560** | - |
| cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit | 1 | 0.98 | 4 | 150560 | 157184 | **150560** | - |
| cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit | 1 | 0.98 | 8 | 150560 | 157136 | **150560** | - |
| cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit | 1 | 0.98 | 16 | 150560 | 157136 | **150560** | - |
| cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit | 2 | 0.98 | 1 | 262144 | 485264 | **262144** | - |
| cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit | 2 | 0.98 | 4 | 262144 | 492752 | **262144** | - |
| cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit | 2 | 0.98 | 8 | 262144 | 492752 | **262144** | - |
| cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit | 2 | 0.98 | 16 | 262144 | 492752 | **262144** | - |
| cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit | 2 | 0.98 | 1 | 262144 | 145248 | **145248** | - |
| cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit | 2 | 0.98 | 4 | 262144 | 156128 | **156128** | - |
| cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit | 2 | 0.98 | 8 | 262144 | 156128 | **156128** | - |
| cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit | 2 | 0.98 | 16 | 262144 | 156128 | **156128** | - |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 1 | 0.98 | 1 | 127120 | 127120 | **127120** | - |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 1 | 0.98 | 4 | 127120 | 127168 | **127120** | - |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 1 | 0.98 | 8 | 127120 | 127184 | **127120** | - |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 1 | 0.98 | 16 | 127120 | 127152 | **127120** | - |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 2 | 0.98 | 1 | 104857 | 379984 | **104857** | - |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 2 | 0.98 | 4 | 104857 | 380080 | **104857** | - |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 2 | 0.98 | 8 | 104857 | 380112 | **104857** | - |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 2 | 0.98 | 16 | 104857 | 380112 | **104857** | - |
| openai/gpt-oss-20b | 1 | 0.98 | 1 | 131072 | 355520 | **131072** | - |
| openai/gpt-oss-20b | 1 | 0.98 | 4 | 131072 | 355312 | **131072** | - |
| openai/gpt-oss-20b | 1 | 0.98 | 8 | 131072 | 355312 | **131072** | - |
| openai/gpt-oss-20b | 1 | 0.98 | 16 | 131072 | 354432 | **131072** | - |
| openai/gpt-oss-20b | 2 | 0.95 | 1 | 131072 | 973840 | **131072** | - |
| openai/gpt-oss-20b | 2 | 0.95 | 4 | 131072 | 973408 | **131072** | - |
| openai/gpt-oss-20b | 2 | 0.95 | 8 | 131072 | 973408 | **131072** | - |
| openai/gpt-oss-20b | 2 | 0.95 | 16 | 131072 | 971776 | **131072** | - |
| openai/gpt-oss-20b | 2 | 0.98 | 1 | 0 | ERROR | - | Verification Failed |
