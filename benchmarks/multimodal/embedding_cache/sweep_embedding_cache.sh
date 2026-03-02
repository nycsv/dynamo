#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark sweep: multimodal embedding cache OFF (control) vs ON (test)
# for disagg_multimodal_e_pd.sh
#
# Usage:
#   ./benchmarks/multimodal/sweep_embedding_cache.sh [OPTIONS]
#
# Options:
#   --workflow <script>     Launch script path relative to repo root
#                           (default: examples/backends/vllm/launch/disagg_multimodal_e_pd.sh)
#   --model <model>         Model name (default: Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)
#   --concurrencies <list>  Comma-separated concurrency levels (default: 1,2,5,10)
#   --osl <tokens>          Output sequence length (default: 150)
#   --request-count <n>     Requests per concurrency level (default: 100)
#   --warmup <n>            Warmup requests (default: 5)
#   --cache-gb <gb>         Embedding cache capacity in GB for test (default: 2)
#   --input-file <path>     Custom aiperf JSONL input file (default: auto-generated)
#   --output-dir <dir>      Results output directory (default: benchmarks/results/embedding_cache_sweep)
#   --port <port>           Frontend port to poll for readiness (default: 8000)
#   --timeout <sec>         Max seconds to wait for server readiness (default: 600)
#   --skip-control          Skip the control run (cache OFF)
#   --skip-test             Skip the test run (cache ON)
#   --skip-plots            Skip plot generation
#   -h, --help              Show this help message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# Defaults
WORKFLOW="examples/backends/vllm/launch/disagg_multimodal_e_pd.sh"
MODEL_NAME="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
CONCURRENCIES="1,2,5,10"
OSL=150
REQUEST_COUNT=100
WARMUP_COUNT=5
CACHE_GB=2
INPUT_FILE=""
OUTPUT_DIR="benchmarks/results/embedding_cache_sweep"
PORT=8000
TIMEOUT=600
SKIP_CONTROL=false
SKIP_TEST=false
SKIP_PLOTS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workflow)
            WORKFLOW=$2
            shift 2
            ;;
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --concurrencies)
            CONCURRENCIES=$2
            shift 2
            ;;
        --osl)
            OSL=$2
            shift 2
            ;;
        --request-count)
            REQUEST_COUNT=$2
            shift 2
            ;;
        --warmup)
            WARMUP_COUNT=$2
            shift 2
            ;;
        --cache-gb)
            CACHE_GB=$2
            shift 2
            ;;
        --input-file)
            INPUT_FILE=$2
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --port)
            PORT=$2
            shift 2
            ;;
        --timeout)
            TIMEOUT=$2
            shift 2
            ;;
        --skip-control)
            SKIP_CONTROL=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --skip-plots)
            SKIP_PLOTS=true
            shift
            ;;
        -h|--help)
            # Print usage from the header comment
            sed -n '6,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

LAUNCH_SCRIPT="$REPO_ROOT/$WORKFLOW"

if [[ ! -x "$LAUNCH_SCRIPT" ]]; then
    echo "ERROR: Workflow script not found or not executable: $LAUNCH_SCRIPT"
    exit 1
fi

IFS=',' read -ra CONCURRENCY_LIST <<< "$CONCURRENCIES"

echo "=================================================="
echo "Embedding Cache Benchmark Sweep"
echo "=================================================="
echo "Workflow:      $WORKFLOW"
echo "Model:         $MODEL_NAME"
echo "Concurrencies: ${CONCURRENCY_LIST[*]}"
echo "OSL:           $OSL"
echo "Requests:      $REQUEST_COUNT per concurrency"
echo "Cache (test):  ${CACHE_GB} GB"
echo "Output:        $OUTPUT_DIR"
echo "=================================================="

# ---------------------------------------------------------------------------
# Generate default input JSONL if none provided
# ---------------------------------------------------------------------------
if [[ -z "$INPUT_FILE" ]]; then
    JSONL_DIR="$SCRIPT_DIR/jsonl"
    # Use the pre-existing JSONL with image pool reuse (good for cache benchmarking)
    DEFAULT_JSONL="$JSONL_DIR/700req_30img_1000pool_300word_http.jsonl"
    if [[ -f "$DEFAULT_JSONL" ]]; then
        INPUT_FILE="$DEFAULT_JSONL"
        echo "Using existing JSONL: $INPUT_FILE"
    else
        echo "Generating JSONL dataset..."
        python "$JSONL_DIR/main.py" \
            -n 700 \
            --images-per-request 30 \
            --images-pool 1000 \
            --image-mode http \
            -o "$DEFAULT_JSONL"
        INPUT_FILE="$DEFAULT_JSONL"
        echo "Generated: $INPUT_FILE"
    fi
fi

# ---------------------------------------------------------------------------
# Helper: wait for the server to become ready
# ---------------------------------------------------------------------------
wait_for_server() {
    local url="http://localhost:${PORT}/v1/models"
    local deadline=$((SECONDS + TIMEOUT))
    echo "Waiting for server at $url (timeout: ${TIMEOUT}s)..."
    while (( SECONDS < deadline )); do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo "Server is ready."
            return 0
        fi
        sleep 5
    done
    echo "ERROR: Server did not become ready within ${TIMEOUT}s"
    return 1
}

# ---------------------------------------------------------------------------
# Helper: launch the serving stack and return its PID
# ---------------------------------------------------------------------------
SERVER_PID=""

start_server() {
    local extra_args=("$@")
    echo "Launching disagg_multimodal_e_pd.sh ${extra_args[*]:-}..."
    bash "$LAUNCH_SCRIPT" \
        --model "$MODEL_NAME" \
        "${extra_args[@]}" &
    SERVER_PID=$!
    wait_for_server
}

stop_server() {
    if [[ -n "$SERVER_PID" ]]; then
        echo "Stopping server (PID $SERVER_PID)..."
        # Kill the process group so all children (frontend, encode, PD) are stopped
        kill -- -"$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        # Give ports time to release
        sleep 5
    fi
}

cleanup() {
    echo "Cleaning up..."
    stop_server
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Helper: run aiperf concurrency sweep
# ---------------------------------------------------------------------------
run_sweep() {
    local label=$1
    local result_dir="$OUTPUT_DIR/$label"
    mkdir -p "$result_dir"

    for c in "${CONCURRENCY_LIST[@]}"; do
        local c_dir="$result_dir/c${c}"
        mkdir -p "$c_dir"
        echo ""
        echo "--- [$label] concurrency=$c ---"
        aiperf profile \
            -m "$MODEL_NAME" \
            --endpoint-type chat \
            --streaming \
            -u "http://localhost:${PORT}" \
            --concurrency "$c" \
            --osl "$OSL" \
            --request-count "$REQUEST_COUNT" \
            --warmup-request-count "$WARMUP_COUNT" \
            --input-file "$INPUT_FILE" \
            --custom-dataset-type single_turn \
            --extra-inputs "max_tokens:${OSL}" \
            --extra-inputs "min_tokens:${OSL}" \
            --extra-inputs "ignore_eos:true" \
            --artifact-dir "$c_dir" \
            --ui none \
            --no-server-metrics
    done
    echo ""
    echo "[$label] sweep complete. Results in $result_dir"
}

# ---------------------------------------------------------------------------
# Control: embedding cache OFF
# ---------------------------------------------------------------------------
if [[ "$SKIP_CONTROL" != "true" ]]; then
    echo ""
    echo "========== CONTROL: embedding cache OFF =========="
    start_server --multimodal-embedding-cache-capacity-gb 0
    run_sweep "cache-off"
    stop_server
fi

# ---------------------------------------------------------------------------
# Test: embedding cache ON
# ---------------------------------------------------------------------------
if [[ "$SKIP_TEST" != "true" ]]; then
    echo ""
    echo "========== TEST: embedding cache ON (${CACHE_GB} GB) =========="
    start_server --multimodal-embedding-cache-capacity-gb "$CACHE_GB"
    run_sweep "cache-on"
    stop_server
fi

# ---------------------------------------------------------------------------
# Generate comparison plots
# ---------------------------------------------------------------------------
if [[ "$SKIP_PLOTS" != "true" ]]; then
    echo ""
    echo "========== Generating comparison plots =========="
    python -m benchmarks.utils.plot \
        --data-dir "$OUTPUT_DIR" \
        --benchmark-name cache-off \
        --benchmark-name cache-on
    echo "Plots saved to: $OUTPUT_DIR/plots/"
fi

echo ""
echo "=================================================="
echo "Sweep complete!"
echo "  Results: $OUTPUT_DIR"
echo "  Control: $OUTPUT_DIR/cache-off/"
echo "  Test:    $OUTPUT_DIR/cache-on/"
if [[ "$SKIP_PLOTS" != "true" ]]; then
    echo "  Plots:   $OUTPUT_DIR/plots/"
fi
echo "=================================================="
