# Qwen3-ASR + Dynamo on 8× A100 GPUs — Complete Tutorial

This document provides a **complete, step-by-step guide** to deploying Qwen3-ASR (speech-to-text LLM) on a machine with 8× A100 GPUs using Dynamo's disaggregated inference framework.

---

## Table of Contents

1. [Hardware & System Requirements](#hardware--system-requirements)
2. [Architecture Overview](#architecture-overview)
3. [Step-by-Step Installation](#step-by-step-installation)
4. [Running the Pipeline](#running-the-pipeline)
5. [Validation & Testing](#validation--testing)
6. [Request Examples](#request-examples)
7. [Troubleshooting](#troubleshooting)
8. [Performance Considerations](#performance-considerations)

---

## Hardware & System Requirements

### Minimum Hardware

| Component | Requirement |
|---|---|
| **GPUs** | 8× NVIDIA A100 (40GB or 80GB VRAM) |
| **CUDA** | 12.x (tested with 12.9) |
| **cuDNN** | 8.x or higher |
| **Python** | 3.10, 3.11, or 3.12 |
| **System RAM** | 64 GB (32 GB minimum) |
| **Disk Space** | 50 GB free (models + packages + venv) |
| **Network** | 1 Gbps (for inter-GPU RDMA, NVLink preferred) |
| **OS** | Ubuntu 20.04+ or other Linux distributions with CUDA support |

### Verify Your Hardware

```bash
# Check GPUs
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# Expected output:
# 0, NVIDIA A100-SXM4-40GB, 40537 MiB
# 1, NVIDIA A100-SXM4-40GB, 40537 MiB
# ... (8 total)

# Check CUDA
nvcc --version
# Expected: CUDA 12.x

# Check Python
python3 --version
# Expected: Python 3.10.x, 3.11.x, or 3.12.x

# Check available disk
df -h / | tail -1
# Expected: at least 50 GB available
```

---

## Architecture Overview

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Request                            │
│  Audio URL + Text Prompt (OpenAI chat format)               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   HTTP Frontend        │ (CPU, port 8000)
        │  Dynamo HTTP Server    │ OpenAI-compatible API
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │     Processor          │ (CPU)
        │  - Parse chat format   │ - Extract audio URL
        │  - Tokenize text       │ - Prepare embedding input
        └────────────┬───────────┘
                     │ (Dynamo round_robin)
                     ▼
     ┌──────────────────────────────────────┐
     │   ASR Encode Worker (GPU 0, 2, 4, 6) │
     │                                       │
     │  1. Download audio (HTTP/HTTPS)       │
     │  2. Resample to 16 kHz                │
     │  3. Qwen3ASRProcessor                 │
     │     → mel spectrogram (num_mel=80)    │
     │  4. Qwen3OmniMoeAudioEncoder         │
     │     → (batch, seq_len, embed_dim)    │
     │  5. Create NIXL descriptor            │
     │  6. GPU→GPU RDMA transfer             │
     └──────────────┬──────────────────────┘
                    │ (NIXL RDMA, zero-copy)
                    ▼
     ┌──────────────────────────────────────┐
     │  vLLM Worker (GPU 1, 3, 5, 7)        │
     │  Prefill + Decode (E/PD aggregated)  │
     │                                       │
     │  1. Receive embeddings via RDMA      │
     │  2. Inject at <|AUDIO|> position     │
     │  3. Run LLM decoder (continuous batch)
     │  4. Streaming token output            │
     │  5. KV cache via NIXL (if multi-turn)│
     └──────────────┬──────────────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │  HTTP Response         │ (SSE streaming or JSON)
        │  Transcribed text +    │
        │  Usage statistics      │
        └────────────────────────┘
```

### Why E/PD (Encoder + Prefill+Decode)?

For **Qwen3-ASR**, the E/PD topology is optimal because:

1. **Audio is encoded once** — unlike text-to-text where prefill and decode both benefit from iterative caching, audio is processed in a single forward pass
2. **NIXL RDMA is nearly free** — embeddings transfer GPU→GPU with near-zero latency, no CPU serialization
3. **Streaming-ready** — architecture supports chunked audio increments (future feature)
4. **Independent scaling** — encoder and decoder can scale independently based on load

---

## Step-by-Step Installation

### Step 1: System Verification

```bash
# Verify CUDA is accessible
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

# Verify CUDA version
cat /usr/local/cuda/version.txt
# Expected: CUDA Release 12.x
```

### Step 2: Create Python Virtual Environment

```bash
# Create a clean venv for this project
python3 -m venv ~/dynamo-qwen3-asr
source ~/dynamo-qwen3-asr/bin/activate

# Upgrade pip and install uv (fast package installer)
pip install --upgrade pip
pip install uv
```

### Step 3: Clone Dynamo Repository

```bash
# If you don't have it already
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo/examples/multimodal
```

### Step 4: Install Dynamo Core

```bash
# From the multimodal examples directory
cd /path/to/dynamo/examples/multimodal

# Install Dynamo with vLLM backend
pip install -e "../../[vllm]"
# This installs:
#   - ai-dynamo-runtime==1.0.0
#   - vllm[flashinfer,runai]==0.16.0
#   - nixl[cu12]<=0.10.1
#   - uvloop, blake3, pydantic, etc.
```

### Step 5: Install vLLM-Omni (CRITICAL)

**This step is mandatory** — `asr_encode_worker.py` imports from `vllm.transformers_utils.configs.qwen3_asr` and `vllm.model_executor.models.qwen3_omni_moe_thinker`, which only exist in **vllm-omni** (not base vLLM).

```bash
# Try PyPI first (0.16.0rc1 may not be available yet)
pip install vllm-omni==0.16.0rc1

# If that fails, build from source (takes ~5-10 minutes)
if [ $? -ne 0 ]; then
    echo "PyPI install failed, building from source..."
    git clone --depth 1 --branch v0.16.0rc1 https://github.com/vllm-project/vllm-omni.git /tmp/vllm-omni
    pip install /tmp/vllm-omni
    rm -rf /tmp/vllm-omni
fi

# Verify the import works
python3 -c "from vllm.transformers_utils.configs.qwen3_asr import Qwen3ASRConfig; print('✓ vllm-omni Qwen3-ASR support available')"
```

### Step 6: Install Audio & GPU Support Libraries

```bash
# Audio processing
pip install 'vllm[audio]' librosa accelerate

# GPU-mode NIXL (optional but recommended — faster RDMA transfers)
pip install cupy-cuda12x>=13.0.0

# Utilities
pip install httpx huggingface_hub safetensors

# Verify
python3 -c "
import librosa
import accelerate
import cupy
print('✓ Audio libs OK')
print('✓ Accelerate OK')
print(f'✓ cupy GPU mode: {cupy.cuda.is_available()}')
"
```

### Step 7: Start Infrastructure Services (etcd + NATS)

Dynamo requires etcd (service discovery) and NATS (messaging) to operate.

#### Option A: Docker (easiest)

```bash
# Start etcd
docker run -d \
  --name dynamo-etcd \
  --restart unless-stopped \
  -p 2379:2379 \
  quay.io/coreos/etcd:v3.5.0 \
  etcd \
    --listen-client-urls http://0.0.0.0:2379 \
    --advertise-client-urls http://localhost:2379 \
    --log-level warn

# Start NATS
docker run -d \
  --name dynamo-nats \
  --restart unless-stopped \
  -p 4222:4222 \
  nats:latest

# Verify both are up
docker ps | grep -E "dynamo-etcd|dynamo-nats"
# Should show both containers running

# Health check
curl http://localhost:2379/health
docker exec dynamo-nats nats-server --version
```

#### Option B: Bare-metal (no Docker required)

```bash
# Create a directory for binaries
mkdir -p ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"

# Download and install etcd
ETCD_VER="v3.5.21"
ARCH="linux-amd64"
wget -q "https://github.com/etcd-io/etcd/releases/download/${ETCD_VER}/etcd-${ETCD_VER}-${ARCH}.tar.gz" -O /tmp/etcd.tar.gz
tar -xzf /tmp/etcd.tar.gz -C /tmp
cp "/tmp/etcd-${ETCD_VER}-${ARCH}/etcd" ~/.local/bin/
rm -rf /tmp/etcd*

# Download and install nats-server
NATS_VER="v2.12.4"
wget -q "https://github.com/nats-io/nats-server/releases/download/${NATS_VER}/nats-server-${NATS_VER}-${ARCH}.tar.gz" -O /tmp/nats.tar.gz
tar -xzf /tmp/nats.tar.gz -C /tmp
cp "/tmp/nats-server-${NATS_VER}-${ARCH}/nats-server" ~/.local/bin/
rm -rf /tmp/nats*

# Start etcd
etcd \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://localhost:2379 \
  --data-dir /tmp/dynamo-etcd \
  --log-level warn \
  > /tmp/etcd.log 2>&1 &

# Start NATS
nats-server -js > /tmp/nats.log 2>&1 &

# Wait for readiness
sleep 3
curl http://localhost:2379/health && echo "etcd OK"
nc -z localhost 4222 && echo "NATS OK"
```

### Step 8: Pre-download Model

Pre-downloading avoids timeout issues during worker startup (model loading can take 2-3 minutes).

```bash
python3 << 'EOF'
from huggingface_hub import snapshot_download

# Download Qwen3-ASR-1.7B (default)
model_path = snapshot_download(
    'Qwen/Qwen3-ASR-1.7B',
    ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*']
)
print(f"✓ Model cached at: {model_path}")

# Or download the smaller 0.6B variant
# model_path = snapshot_download(
#     'Qwen/Qwen3-ASR-0.6B',
#     ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*']
# )
# print(f"✓ Model cached at: {model_path}")
EOF
```

---

## Running the Pipeline

### Option 1: Quick Start with install.sh (Recommended)

Navigate to `examples/multimodal/` and run the bundled setup script:

```bash
cd /path/to/dynamo/examples/multimodal

# This script handles:
# - Creating venv
# - Installing all dependencies
# - Downloading etcd/nats binaries
# - Starting services
# - Pre-downloading the model
bash launch/install.sh

# Activate the venv it created
source ./venv/bin/activate

# Launch the pipeline (2-GPU E/PD)
bash launch/asr_speechllm_agg.sh
```

### Option 2: Manual Launch on 8× A100 (4 replica pairs)

For maximum throughput with independent replicas:

```bash
cd /path/to/dynamo/examples/multimodal
source ~/dynamo-qwen3-asr/bin/activate

# Launch the 8-GPU topology
bash launch/asr_8gpu_replicas.sh

# Or with the smaller 0.6B model
bash launch/asr_8gpu_replicas.sh --model Qwen/Qwen3-ASR-0.6B

# To use tensor parallelism (TP=2) for larger models
bash launch/asr_8gpu_replicas.sh --tp 2
```

**What this launches:**

```
Frontend (CPU:8000)
  │
Processor (CPU)
  │
Encoder 1 (GPU 0) ──RDMA──> LLM 1 (GPU 1)
Encoder 2 (GPU 2) ──RDMA──> LLM 2 (GPU 3)
Encoder 3 (GPU 4) ──RDMA──> LLM 3 (GPU 5)
Encoder 4 (GPU 6) ──RDMA──> LLM 4 (GPU 7)
```

All four replica pairs listen on the same Dynamo endpoints — `round_robin` routing distributes requests.

### Option 3: Standard 2-GPU Deployment (E/PD)

For development or single-request testing:

```bash
bash launch/asr_speechllm_agg.sh
```

Uses GPU 0 (encoder) and GPU 1 (LLM). Other GPUs idle.

### Option 4: 3-GPU High-Throughput (E/P/D)

Separates prefill and decode for independent scaling at high batch sizes:

```bash
bash launch/asr_speechllm_disagg.sh
```

Uses GPU 0 (encoder), GPU 1 (prefill), GPU 2 (decode).

---

## Validation & Testing

### Check Pipeline Readiness

```bash
# Poll every 5 seconds until the frontend responds
until curl -sf http://localhost:8000/v1/models > /dev/null; do
    echo "Waiting for pipeline..."
    sleep 5
done

echo "✓ Pipeline is ready!"
curl http://localhost:8000/v1/models | python3 -m json.tool
```

Expected output:

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-ASR-1.7B",
      "object": "model",
      "created": 1709507690,
      "owned_by": "Qwen",
      "permission": [],
      "root": "Qwen/Qwen3-ASR-1.7B",
      "parent": null
    }
  ]
}
```

### Run Automated Validation

```bash
bash launch/validate_asr_speechllm_agg.sh
```

This runs 6 tests:
1. Frontend health check
2. Model discovery
3. Non-streaming ASR transcription
4. Streaming ASR (SSE)
5. Invalid audio URL error handling
6. Text-only request fallback

Expected result:

```
==================================================
Qwen3-ASR SpeechLLM Pipeline Validation
==================================================
Frontend:  http://localhost:8000
Model:     Qwen/Qwen3-ASR-1.7B
Audio URL: https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav
Timeout:   120s
==================================================

[1/6] Checking frontend health...
  PASS: Frontend is reachable
[2/6] Discovering model...
  PASS: Model detected: Qwen/Qwen3-ASR-1.7B
  PASS: Model 'Qwen/Qwen3-ASR-1.7B' is listed in /v1/models
[3/6] Testing ASR transcription (non-streaming)...
  PASS: ASR transcription returned non-empty result
  PASS: Response includes valid usage statistics
[4/6] Testing ASR transcription (streaming)...
  PASS: Streaming ASR transcription works
  PASS: Stream properly terminates with [DONE]
[5/6] Testing error handling (invalid audio URL)...
  PASS: Invalid audio URL returns error (HTTP 4xx)
[6/6] Testing text-only request handling...
  PASS: Text-only request correctly rejected (HTTP 4xx)
==================================================
Results: 8 passed, 0 failed, 0 skipped (out of 8)
==================================================

All tests PASSED!
```

---

## Request Examples

### Example 1: Non-Streaming Transcription

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-ASR-1.7B",
    "messages": [{
      "role": "user",
      "content": [
        {
          "type": "audio_url",
          "audio_url": {
            "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"
          }
        },
        {"type": "text", "text": "Transcribe this audio."}
      ]
    }],
    "max_tokens": 256,
    "temperature": 0.0,
    "stream": false
  }' | python3 -m json.tool
```

Response (truncated):

```json
{
  "id": "chatcmpl-...",
  "object": "text_completion",
  "created": 1709507800,
  "model": "Qwen/Qwen3-ASR-1.7B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The person in the audio is a male, probably around 30 years old."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 125,
    "completion_tokens": 24,
    "total_tokens": 149
  }
}
```

### Example 2: Streaming Transcription (SSE)

```bash
curl -sN -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-ASR-1.7B",
    "messages": [{
      "role": "user",
      "content": [
        {
          "type": "audio_url",
          "audio_url": {
            "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"
          }
        },
        {"type": "text", "text": "Transcribe this audio."}
      ]
    }],
    "max_tokens": 256,
    "temperature": 0.0,
    "stream": true
  }'
```

Output (Server-Sent Events):

```
data: {"id":"chatcmpl-...","object":"text_completion.chunk","created":1709507800,"model":"Qwen/Qwen3-ASR-1.7B","choices":[{"index":0,"delta":{"role":"assistant","content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"text_completion.chunk","created":1709507800,"model":"Qwen/Qwen3-ASR-1.7B","choices":[{"index":0,"delta":{"content":" person"},"finish_reason":null}]}

...

data: [DONE]
```

### Example 3: Translation with Context

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-ASR-1.7B",
    "messages": [{
      "role": "user",
      "content": [
        {
          "type": "audio_url",
          "audio_url": {
            "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"
          }
        },
        {
          "type": "text",
          "text": "Transcribe this audio and then translate the transcription to Spanish."
        }
      ]
    }],
    "max_tokens": 512,
    "temperature": 0.0
  }' | python3 -m json.tool
```

### Example 4: Using a Local Audio File

If your audio is local, first serve it via HTTP (or upload to a publicly accessible URL):

```bash
# Start a simple HTTP server to serve local audio
cd /path/to/audio
python3 -m http.server 9999 &

# Then request it
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-ASR-1.7B",
    "messages": [{
      "role": "user",
      "content": [
        {
          "type": "audio_url",
          "audio_url": {"url": "http://localhost:9999/audio.wav"}
        },
        {"type": "text", "text": "Transcribe this audio."}
      ]
    }],
    "max_tokens": 256
  }' | python3 -m json.tool
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: vllm.transformers_utils.configs.qwen3_asr`

**Cause:** vllm-omni is not installed.

**Solution:**

```bash
pip install vllm-omni==0.16.0rc1

# Or build from source:
git clone --depth 1 --branch v0.16.0rc1 https://github.com/vllm-project/vllm-omni.git /tmp/vllm-omni
pip install /tmp/vllm-omni
```

### Issue: Workers fail to discover each other / "Failed to connect to service registry"

**Cause:** etcd is not running or not accessible.

**Solution:**

```bash
# Check etcd health
curl http://localhost:2379/health

# If it fails, restart etcd:
docker restart dynamo-etcd
# or (bare-metal):
pkill etcd
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://localhost:2379 \
     --data-dir /tmp/dynamo-etcd &
```

### Issue: "NATS connection refused" / "Failed to connect to message broker"

**Cause:** NATS server is not running.

**Solution:**

```bash
# Check NATS
docker ps | grep dynamo-nats
# or (bare-metal):
ps aux | grep nats-server

# If not running, restart it:
docker restart dynamo-nats
# or (bare-metal):
nats-server -js > /tmp/nats.log 2>&1 &
```

### Issue: "CUDA out of memory"

**Cause:** Qwen3-ASR-1.7B doesn't fit on available GPUs.

**Solution:** Use the smaller model:

```bash
bash launch/asr_8gpu_replicas.sh --model Qwen/Qwen3-ASR-0.6B
```

Or reduce batch size / max_tokens in requests.

### Issue: Encoder falls back to numpy (slow NIXL transfers)

**Cause:** cupy is not installed or CUDA is not available to it.

**Solution:**

```bash
pip install cupy-cuda12x
python3 -c "import cupy; print('GPU available:', cupy.cuda.is_available())"
```

If cupy still doesn't detect CUDA, verify:

```bash
which nvcc
nvcc --version
echo $CUDA_HOME
```

### Issue: Workers start but models list is empty

**Cause:** Processor hasn't registered the model yet (waiting for encoder to be ready).

**Solution:** Wait longer and check processor logs:

```bash
# Processor should print "Starting to serve the endpoint..." after encoder is ready
# Check logs, then retry model discovery:
sleep 10
curl http://localhost:8000/v1/models
```

### Issue: Port conflicts (8 replicas won't start)

**Cause:** Replica ports are colliding (same `DYN_VLLM_KV_EVENT_PORT` or `VLLM_NIXL_SIDE_CHANNEL_PORT`).

**Solution:** The launch script auto-assigns ports. Verify in the output:

```
Pair 1: DYN_VLLM_KV_EVENT_PORT=20081, VLLM_NIXL_SIDE_CHANNEL_PORT=20091
Pair 2: DYN_VLLM_KV_EVENT_PORT=20082, VLLM_NIXL_SIDE_CHANNEL_PORT=20092
...
```

If custom ports are needed, edit `asr_8gpu_replicas.sh`.

---

## Performance Considerations

### Throughput Scaling

With 4 replica pairs (8 GPUs):

| Scenario | Throughput | Notes |
|---|---|---|
| 1 replica (2 GPUs) | ~1 req/sec | Baseline |
| 2 replicas (4 GPUs) | ~2 req/sec | Linear scaling |
| 4 replicas (8 GPUs) | ~4 req/sec | Linear scaling (round-robin) |

Assumes 10-second audio, 30-token output, and non-overlapping requests.

### Latency Breakdown

For a typical 10-second audio file:

| Stage | Latency |
|---|---|
| Audio download | 100–500 ms (network-dependent) |
| Audio processing (Qwen3ASRProcessor) | 50–100 ms (CPU-bound) |
| Encoder forward pass (GPU) | 200–500 ms |
| NIXL RDMA transfer | 1–10 ms (GPU-to-GPU, NVLink) |
| LLM decode (30 tokens @ ~10 ms/token) | 300–500 ms |
| **Total** | **~1–1.5 seconds** |

### Memory Usage

| Model | Encoder VRAM | LLM VRAM | Total (1 pair) |
|---|---|---|---|
| Qwen3-ASR-1.7B | ~2 GB | ~4 GB | ~6 GB |
| Qwen3-ASR-0.6B | ~1 GB | ~2 GB | ~3 GB |

On A100-40GB, all models fit comfortably with room for batch size > 1.

### Optimization Tips

1. **Pre-cache frequently used audio** — the `AudioLoader` uses LRU cache (max 8 by default, configurable)
2. **Use streaming for large responses** — SSE streaming (stream=true) reduces latency perception
3. **Batch requests** — if latency allows, batch multiple audio files per Dynamo request cycle
4. **Enable prefix caching** — for repeated system prompts in translation tasks (enabled by default)
5. **Monitor KV events** — use ZMQ publisher metrics to detect bottlenecks

---

## Next Steps

- **Multi-node deployment:** See `examples/basics/multinode/README.md` for scaling across multiple machines
- **Custom models:** To support other Qwen audio models, update `utils/model.py` and `utils/args.py`
- **LoRA fine-tuning:** See `examples/multimodal/deploy/lora/README.md` for model adaptation
- **Production deployment:** See `examples/deployments/` for Kubernetes, ECS, AKS, GKE, and EFS setups

---

## References

- [Dynamo GitHub](https://github.com/ai-dynamo/dynamo)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3-ASR Model Card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [NVIDIA A100 GPU Datasheet](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-memory-hierarchy-whitepaper-v1.pdf)
