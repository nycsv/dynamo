#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Recommended E/PD deployment for Qwen3-ASR SpeechLLM pipeline.
# Encoder on GPU 0, combined prefill+decode on GPU 1.
#
# This is the primary topology for Qwen3-ASR because:
#   - ASR input is processed once (no iterative prefill benefit from separation)
#   - Streaming-ready: future chunked audio can be incrementally appended
#     and decoded without P/D round-trip latency per chunk
#
# For batch high-throughput scenarios, use asr_speechllm_disagg.sh (E/P/D) instead.

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
# NOTE: The exact audio special tokens for Qwen3-ASR must be verified against the
# model's tokenizer. The tokens below are based on the Qwen3-ASR config:
#   audio_start_token_id=151669, audio_end_token_id=151670, audio_token_id=151676
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
    # Check if audio dependencies are available (librosa is a key audio dependency)
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

# run E/PD workers (recommended: encoder separate, prefill+decode combined)
CUDA_VISIBLE_DEVICES=0 python3 components/asr_encode_worker.py --model $MODEL_NAME &
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 CUDA_VISIBLE_DEVICES=1 python3 components/worker.py --model $MODEL_NAME --worker-type prefill &

# Wait for all background processes to complete
wait
