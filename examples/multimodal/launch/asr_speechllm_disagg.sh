#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Optional E/P/D fully disaggregated deployment for Qwen3-ASR SpeechLLM pipeline.
# Encoder on GPU 0, prefill on GPU 1, decode on GPU 2.
#
# Use this for batch high-throughput scenarios where prefill and decode
# contend for GPU compute. For low-latency and streaming use cases,
# prefer asr_speechllm_agg.sh (E/PD) instead.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL_NAME="Qwen/Qwen3-ASR-1.7B"
PROMPT_TEMPLATE=""
PROVIDED_PROMPT_TEMPLATE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --prompt-template)
            PROVIDED_PROMPT_TEMPLATE=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <model_name> Specify the ASR model to use (default: $MODEL_NAME)"
            echo "  --prompt-template <template> Specify the prompt template for ASR audio tokens."
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set PROMPT_TEMPLATE based on the MODEL_NAME
if [[ -n "$PROVIDED_PROMPT_TEMPLATE" ]]; then
    PROMPT_TEMPLATE="$PROVIDED_PROMPT_TEMPLATE"
elif [[ "$MODEL_NAME" == "Qwen/Qwen3-ASR-1.7B" || "$MODEL_NAME" == "Qwen/Qwen3-ASR-0.6B" ]]; then
    PROMPT_TEMPLATE="<|startofaudio|><|AUDIO|><|endofaudio|>"
else
    echo "No prompt template is defined for the model: $MODEL_NAME"
    echo "Please provide a prompt template using --prompt-template option."
    exit 1
fi

# Check and install required dependencies for ASR models
echo "Checking ASR dependencies..."
DEPS_MISSING=false

# Check for accelerate
if ! python -c "import accelerate" &> /dev/null; then
    echo "  accelerate not found"
    DEPS_MISSING=true
else
    echo "  ✓ accelerate is installed"
fi

# Check for vllm with audio support
if ! python -c "import vllm" &> /dev/null; then
    echo "  vllm not found"
    DEPS_MISSING=true
else
    if ! python -c "import librosa" &> /dev/null; then
        echo "  vllm audio dependencies not found"
        DEPS_MISSING=true
    else
        echo "  ✓ vllm with audio support is installed"
    fi
fi

# Install missing dependencies
if [ "$DEPS_MISSING" = true ]; then
    echo "Installing missing dependencies..."
    pip install 'vllm[audio]' accelerate
    echo "Dependencies installed successfully"
else
    echo "All required dependencies are already installed"
fi

# run ingress
python -m dynamo.frontend --http-port 8000 &

# run processor
python3 components/processor.py --model $MODEL_NAME --prompt-template "$PROMPT_TEMPLATE" &

# run E/P/D workers (encoder, prefill, decode on separate GPUs)
CUDA_VISIBLE_DEVICES=0 python3 components/asr_encode_worker.py --model $MODEL_NAME &
DYN_VLLM_KV_EVENT_PORT=20081 VLLM_NIXL_SIDE_CHANNEL_PORT=20098 CUDA_VISIBLE_DEVICES=1 python3 components/worker.py --model $MODEL_NAME --worker-type prefill --enable-disagg &
DYN_VLLM_KV_EVENT_PORT=20082 VLLM_NIXL_SIDE_CHANNEL_PORT=20099 CUDA_VISIBLE_DEVICES=2 python3 components/worker.py --model $MODEL_NAME --worker-type decode --enable-disagg &

# Wait for all background processes to complete
wait
