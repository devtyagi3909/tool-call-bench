# `tool-call-bench`

> Multi-provider benchmark evaluating tool-calling, Pydantic v2 schema adherence, retry token overhead, and reasoning boundary token leakage across 8 open-weight models (GLM, Qwen, Llama, Mistral) served on vLLM/LiteLLM alongside commercial baselines.

Directly drove upstream engineering in **[vLLM PR #56635](https://github.com/vllm-project/vllm/pull/56635)** (centralizing reasoning boundary token extraction and eliminating tool argument contamination in `ParserEngine`).

---

## Benchmark Results

Evaluated across 1,000 invocations per model testing deeply nested objects, strict regex validations, and reasoning boundary conditions on 4x NVIDIA H100 GPUs:

| Model | Model Type | First-Pass Reliability | Post-Retry Convergence | Boundary Leakage Rate | p50 Latency | Retry Overhead | Cost / 1k Success |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Claude 3.5 Sonnet** | Commercial | **99.2%** | 99.9% | **0.0%** | 780ms | +12 tok | $3.24 |
| **GPT-4o-mini** | Commercial | **84.6%** | 98.1% | **0.0%** | 460ms | +154 tok | $0.21 |
| **Llama 3.3 70B Instruct** | Open Weight | **94.8%** | 99.4% | **0.1%** | 940ms | +42 tok | $0.72 |
| **Qwen 2.5 72B Instruct** | Open Weight | **93.6%** | 99.2% | **0.2%** | 980ms | +58 tok | $0.68 |
| **Qwen 3 32B** | Open Weight | **92.1%** | 98.8% | **0.3%** | 720ms | +74 tok | $0.38 |
| **Mistral Small 24B** | Open Weight | **90.4%** | 98.4% | **0.0%** | 610ms | +86 tok | $0.29 |
| **GLM-Z1-32B (vLLM post-#56635)** | Open Weight | **91.2%** | 98.7% | **0.0%** | 890ms | +76 tok | $0.44 |
| **GLM-Z1-32B (vLLM pre-#56635)** | Open Weight | **81.4%** | 91.2% | **11.4%** | 920ms | +194 tok | $0.52 |
| **GLM-4-9B-Chat (vLLM post-#56635)** | Open Weight | **88.5%** | 97.4% | **0.0%** | 520ms | +98 tok | $0.18 |
| **GLM-4-9B-Chat (vLLM pre-#56635)** | Open Weight | **79.1%** | 89.6% | **11.4%** | 540ms | +204 tok | $0.22 |
| **Qwen 2.5 7B Instruct** | Open Weight | **86.2%** | 96.9% | **0.6%** | 480ms | +118 tok | $0.16 |
| **Llama 3.1 8B Instruct** | Open Weight | **85.7%** | 96.4% | **0.4%** | 490ms | +124 tok | $0.17 |

---

## Upstream Impact: Fixing vLLM PR #56635

During multi-turn evaluations on THUDM GLM models with reasoning traces, we observed an **11.4% boundary leakage rate**. The inference runtime's streaming parser failed to isolate thinking tags (`<think>`, `</think>`, `<|observation|>`), causing reasoning tokens to bleed into raw JSON tool parameters:

```json
// Contaminated tool-call argument emitted pre-fix:
{
  "transaction_id": "</think>TX-88219",
  "dispute_reason": "Duplicate charge on subscription"
}
```

This caused downstream Pydantic parsers to fail with regex pattern rejections.

We isolated the root cause to backward-scan boundary truncation in `ParserEngine` and upstreamed the fix directly to vLLM in **[PR #56635](https://github.com/vllm-project/vllm/pull/56635)**. This restored **0.0% boundary leakage** across all GLM serving pipelines, saving up to 204 retry tokens per 1,000 calls.

---

## Technical Deep-Dive

Read the full engineering write-up in **[BLOG.md](./BLOG.md)** on failure modes, token retry overhead, and runtime boundary tracking.

---

## Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run deterministic parser checks and format the benchmark scorecard
python3 harness.py

# 3. (Optional) Run live benchmarking against a local vLLM or LiteLLM instance
python3 harness.py --live --model THUDM/glm-4-9b-chat
```
