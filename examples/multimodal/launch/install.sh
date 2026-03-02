#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-time setup for the Dynamo multimodal ASR pipeline on a notebook A100 GPU.
# Downloads etcd + nats-server binaries locally (no Docker, no sudo required)
# and installs Python deps into a venv.
#
# Usage:
#   bash launch/install.sh [--venv /path/to/venv] [--model Qwen/Qwen3-ASR-1.7B]
#
# After running, activate the venv before launching the pipeline:
#   source ./venv/bin/activate
#   bash launch/asr_speechllm_agg.sh --model Qwen/Qwen3-ASR-1.7B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$REPO_DIR/venv"
BIN_DIR="$REPO_DIR/.local/bin"

ETCD_VERSION="v3.5.21"
NATS_VERSION="v2.12.4"
MODEL_NAME="Qwen/Qwen3-ASR-1.7B"

# ── Parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --venv)  VENV_DIR="$2";   shift 2 ;;
        --model) MODEL_NAME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$BIN_DIR"

echo "=== Dynamo ASR Pipeline: One-Time Setup ==="
echo "  venv  : $VENV_DIR"
echo "  bins  : $BIN_DIR"
echo "  model : $MODEL_NAME"
echo ""

# ── Python venv ────────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/5] Creating Python venv..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/5] Using existing venv: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[2/5] Installing Python dependencies..."
pip install --upgrade pip -q
pip install \
    'vllm[audio]' \
    accelerate \
    ai-dynamo \
    uvloop \
    huggingface_hub \
    safetensors \
    cupy-cuda12x \
    -q
echo "  ✓ Python dependencies installed"

# ── Model download ─────────────────────────────────────────────────────────────
echo "[3/5] Downloading model: $MODEL_NAME ..."
python3 - <<EOF
from huggingface_hub import snapshot_download
path = snapshot_download(
    "$MODEL_NAME",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
)
print(f"  Model cached at: {path}")
EOF
echo "  ✓ Model ready"

# ── etcd binary ────────────────────────────────────────────────────────────────
if [ ! -f "$BIN_DIR/etcd" ]; then
    echo "[4/5] Downloading etcd $ETCD_VERSION..."
    TMP=$(mktemp -d)
    curl -fsSL "https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-amd64.tar.gz" \
        | tar -xz -C "$TMP"
    mv "$TMP/etcd-${ETCD_VERSION}-linux-amd64/etcd" "$BIN_DIR/etcd"
    rm -rf "$TMP"
    echo "  ✓ etcd installed to $BIN_DIR/etcd"
else
    echo "[4/5] etcd already present: $BIN_DIR/etcd"
fi

# ── nats-server binary ─────────────────────────────────────────────────────────
if [ ! -f "$BIN_DIR/nats-server" ]; then
    echo "[5/5] Downloading nats-server $NATS_VERSION..."
    TMP=$(mktemp -d)
    curl -fsSL "https://github.com/nats-io/nats-server/releases/download/${NATS_VERSION}/nats-server-${NATS_VERSION}-linux-amd64.tar.gz" \
        | tar -xz -C "$TMP"
    mv "$TMP/nats-server-${NATS_VERSION}-linux-amd64/nats-server" "$BIN_DIR/nats-server"
    rm -rf "$TMP"
    echo "  ✓ nats-server installed to $BIN_DIR/nats-server"
else
    echo "[5/5] nats-server already present: $BIN_DIR/nats-server"
fi

# ── Start services ─────────────────────────────────────────────────────────────
echo ""
echo "Starting background services..."

# etcd
if ! nc -z localhost 2379 2>/dev/null; then
    "$BIN_DIR/etcd" \
        --listen-client-urls http://0.0.0.0:2379 \
        --advertise-client-urls http://0.0.0.0:2379 \
        --data-dir /tmp/dynamo-etcd \
        --log-level warn \
        > /tmp/dynamo-etcd.log 2>&1 &
    echo "  etcd started (pid $!), log: /tmp/dynamo-etcd.log"
else
    echo "  etcd already running on :2379"
fi

# nats-server
if ! nc -z localhost 4222 2>/dev/null; then
    "$BIN_DIR/nats-server" -js \
        > /tmp/dynamo-nats.log 2>&1 &
    echo "  nats-server started (pid $!), log: /tmp/dynamo-nats.log"
else
    echo "  nats-server already running on :4222"
fi

# Wait for readiness
echo ""
echo "Waiting for services..."
for i in $(seq 1 30); do
    etcd_ok=false; nats_ok=false
    nc -z localhost 2379 2>/dev/null && etcd_ok=true
    nc -z localhost 4222 2>/dev/null && nats_ok=true
    $etcd_ok && $nats_ok && break
    [ "$i" -eq 30 ] && { echo "ERROR: services did not start in 30s"; exit 1; }
    sleep 1
done
echo "  ✓ etcd  :2379"
echo "  ✓ NATS  :4222"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To launch the pipeline:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd $REPO_DIR"
echo "  bash launch/asr_speechllm_agg.sh --model Qwen/Qwen3-ASR-1.7B"
