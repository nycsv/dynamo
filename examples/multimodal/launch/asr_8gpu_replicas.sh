#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Scale-out deployment for Qwen3-ASR on 8 GPUs.
#
# Launches 4 independent E/PD replica pairs.  Dynamo's round_robin routing
# distributes requests across all replicas automatically, giving ~4× the
# throughput of a single E/PD pair.
#
# GPU assignment (default, --tp 1):
#   GPU 0  Encoder replica 1     GPU 1  LLM (PD) replica 1
#   GPU 2  Encoder replica 2     GPU 3  LLM (PD) replica 2
#   GPU 4  Encoder replica 3     GPU 5  LLM (PD) replica 3
#   GPU 6  Encoder replica 4     GPU 7  LLM (PD) replica 4
#
# GPU assignment (--tp 2, uses tensor-parallel LLM workers):
#   GPU 0  Encoder replica 1     GPU 1-2  LLM TP=2 replica 1
#   GPU 3  Encoder replica 2     GPU 4-5  LLM TP=2 replica 2
#   GPU 6  Encoder replica 3     GPU 7    LLM TP=1 replica 3 (odd one out)
#   (only 3 pairs fit; consider --tp 1 for 4 full pairs)
#
# Prerequisites:
#   etcd running on :2379
#   NATS running on :4222
#   vllm-omni installed (required for Qwen3-ASR encoder)
#   pip install 'vllm[audio]' accelerate cupy-cuda12x
#
# Usage:
#   bash launch/asr_8gpu_replicas.sh
#   bash launch/asr_8gpu_replicas.sh --model Qwen/Qwen3-ASR-0.6B
#   bash launch/asr_8gpu_replicas.sh --model Qwen/Qwen3-ASR-1.7B --tp 1

set -e
trap 'echo "Cleaning up..."; kill 0' EXIT

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_NAME="Qwen/Qwen3-ASR-1.7B"
PROMPT_TEMPLATE=""
PROVIDED_PROMPT_TEMPLATE=""
TP=1          # tensor-parallel size per LLM worker

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2; shift 2 ;;
        --prompt-template)
            PROVIDED_PROMPT_TEMPLATE=$2; shift 2 ;;
        --tp)
            TP=$2; shift 2 ;;
        -h|--help)
            cat <<USAGE
Usage: $0 [OPTIONS]

Options:
  --model <name>           ASR model (default: $MODEL_NAME)
  --prompt-template <tpl>  Audio token template (auto-set for Qwen3-ASR)
  --tp <n>                 Tensor-parallel size per LLM worker (default: $TP)
                           Use 1 for 4 replica pairs on 8 GPUs.
                           Use 2 for 2 replica pairs (each pair uses 3 GPUs).
  -h, --help               Show this help
USAGE
            exit 0 ;;
        *)
            echo "Unknown option: $1"; echo "Use --help for usage."; exit 1 ;;
    esac
done

# ── Prompt template ───────────────────────────────────────────────────────────
if [[ -n "$PROVIDED_PROMPT_TEMPLATE" ]]; then
    PROMPT_TEMPLATE="$PROVIDED_PROMPT_TEMPLATE"
elif [[ "$MODEL_NAME" == "Qwen/Qwen3-ASR-1.7B" || "$MODEL_NAME" == "Qwen/Qwen3-ASR-0.6B" ]]; then
    PROMPT_TEMPLATE="<|startofaudio|><|AUDIO|><|endofaudio|>"
else
    echo "No prompt template defined for: $MODEL_NAME"
    echo "Pass one with --prompt-template."
    exit 1
fi

# ── Dependency check ─────────────────────────────────────────────────────────
echo "Checking dependencies..."
DEPS_MISSING=false

for pkg in accelerate librosa; do
    if ! python -c "import $pkg" &>/dev/null; then
        echo "  $pkg not found"
        DEPS_MISSING=true
    else
        echo "  ✓ $pkg"
    fi
done

if ! python -c "from vllm.transformers_utils.configs.qwen3_asr import Qwen3ASRConfig" &>/dev/null; then
    echo "  vllm-omni not found (required for Qwen3-ASR encoder)"
    echo "  Install with: pip install vllm-omni==0.16.0rc1"
    DEPS_MISSING=true
else
    echo "  ✓ vllm-omni (Qwen3-ASR support)"
fi

if [ "$DEPS_MISSING" = true ]; then
    echo "Installing missing base audio deps..."
    pip install 'vllm[audio]' accelerate -q
fi

# cupy is optional (falls back to numpy; NIXL still works, slightly slower)
if ! python -c "import cupy; assert cupy.cuda.is_available()" &>/dev/null; then
    echo "  WARNING: cupy not available — NIXL will use numpy fallback (slower)"
    echo "  Install with: pip install cupy-cuda12x"
else
    echo "  ✓ cupy (GPU-mode NIXL)"
fi

echo ""
echo "Starting Qwen3-ASR 8-GPU pipeline"
echo "  model   : $MODEL_NAME"
echo "  template: $PROMPT_TEMPLATE"
echo "  TP      : $TP"
echo ""

# ── Frontend (CPU) ────────────────────────────────────────────────────────────
python -m dynamo.frontend --http-port 8000 &

# ── Processor (CPU) ───────────────────────────────────────────────────────────
python3 components/processor.py \
    --model "$MODEL_NAME" \
    --prompt-template "$PROMPT_TEMPLATE" &

# ── Replica pairs (TP=1 → 4 pairs on 8 GPUs) ─────────────────────────────────
#
# Each pair gets:
#   - Unique DYN_VLLM_KV_EVENT_PORT  (20081, 20082, 20083, 20084)
#   - Unique VLLM_NIXL_SIDE_CHANNEL_PORT (20091, 20092, 20093, 20094)
#
# For --tp 2, change GPU assignments to consecutive pairs for the LLM worker
# and reduce to 2 encoder + 2 LLM-TP2 replicas.

if [[ "$TP" -eq 1 ]]; then
    # ── 4 replica pairs, 1 GPU each ───────────────────────────────────────────
    echo "Launching 4× E/PD replica pairs (TP=1, GPUs 0-7)..."

    # Pair 1 — GPU 0 (encoder), GPU 1 (LLM)
    CUDA_VISIBLE_DEVICES=0 python3 components/asr_encode_worker.py \
        --model "$MODEL_NAME" &
    DYN_VLLM_KV_EVENT_PORT=20081 VLLM_NIXL_SIDE_CHANNEL_PORT=20091 \
        CUDA_VISIBLE_DEVICES=1 python3 components/worker.py \
        --model "$MODEL_NAME" --worker-type prefill &

    # Pair 2 — GPU 2 (encoder), GPU 3 (LLM)
    CUDA_VISIBLE_DEVICES=2 python3 components/asr_encode_worker.py \
        --model "$MODEL_NAME" &
    DYN_VLLM_KV_EVENT_PORT=20082 VLLM_NIXL_SIDE_CHANNEL_PORT=20092 \
        CUDA_VISIBLE_DEVICES=3 python3 components/worker.py \
        --model "$MODEL_NAME" --worker-type prefill &

    # Pair 3 — GPU 4 (encoder), GPU 5 (LLM)
    CUDA_VISIBLE_DEVICES=4 python3 components/asr_encode_worker.py \
        --model "$MODEL_NAME" &
    DYN_VLLM_KV_EVENT_PORT=20083 VLLM_NIXL_SIDE_CHANNEL_PORT=20093 \
        CUDA_VISIBLE_DEVICES=5 python3 components/worker.py \
        --model "$MODEL_NAME" --worker-type prefill &

    # Pair 4 — GPU 6 (encoder), GPU 7 (LLM)
    CUDA_VISIBLE_DEVICES=6 python3 components/asr_encode_worker.py \
        --model "$MODEL_NAME" &
    DYN_VLLM_KV_EVENT_PORT=20084 VLLM_NIXL_SIDE_CHANNEL_PORT=20094 \
        CUDA_VISIBLE_DEVICES=7 python3 components/worker.py \
        --model "$MODEL_NAME" --worker-type prefill &

elif [[ "$TP" -eq 2 ]]; then
    # ── 2 replica pairs, TP=2 LLM workers ────────────────────────────────────
    # GPU 0: encoder 1   GPU 1-2: LLM TP=2 replica 1
    # GPU 3: encoder 2   GPU 4-5: LLM TP=2 replica 2
    # GPU 6-7: unused (or add a third pair with TP=2 if available)
    echo "Launching 2× E/PD replica pairs (TP=2, GPUs 0-5)..."

    CUDA_VISIBLE_DEVICES=0 python3 components/asr_encode_worker.py \
        --model "$MODEL_NAME" &
    DYN_VLLM_KV_EVENT_PORT=20081 VLLM_NIXL_SIDE_CHANNEL_PORT=20091 \
        CUDA_VISIBLE_DEVICES=1,2 python3 components/worker.py \
        --model "$MODEL_NAME" --worker-type prefill \
        --tensor-parallel-size 2 &

    CUDA_VISIBLE_DEVICES=3 python3 components/asr_encode_worker.py \
        --model "$MODEL_NAME" &
    DYN_VLLM_KV_EVENT_PORT=20082 VLLM_NIXL_SIDE_CHANNEL_PORT=20092 \
        CUDA_VISIBLE_DEVICES=4,5 python3 components/worker.py \
        --model "$MODEL_NAME" --worker-type prefill \
        --tensor-parallel-size 2 &

    echo "Note: GPUs 6-7 unused with TP=2. Add a third pair or use --tp 1 for full utilization."

else
    echo "ERROR: --tp $TP not supported by this script. Use 1 or 2."
    exit 1
fi

# ── Wait ──────────────────────────────────────────────────────────────────────
echo ""
echo "All processes launched. Waiting for readiness..."
echo "  Poll: until curl -sf http://localhost:8000/v1/models; do sleep 5; done"
echo "  Test: bash launch/validate_asr_speechllm_agg.sh"
echo ""

wait
