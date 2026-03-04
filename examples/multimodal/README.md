# Multimodal ASR + SpeechLLM with Dynamo

End-to-end guide for running **Qwen3-ASR** (and Qwen2-Audio) on Dynamo with vLLM — covering installation, single-node multi-GPU deployment, and validation.

---

## Contents

```
examples/multimodal/
├── components/
│   ├── asr_encode_worker.py   # Qwen3-ASR encoder + projector (GPU, NIXL out)
│   ├── audio_encode_worker.py # Qwen2-Audio encoder (GPU, NIXL out)
│   ├── processor.py           # OpenAI chat parser → encoder client
│   ├── worker.py              # vLLM LLM decoder (prefill / decode / both)
│   └── publisher.py           # Stats logger
├── launch/
│   ├── install.sh             # One-time setup: venv, etcd, NATS, model download
│   ├── asr_speechllm_agg.sh   # E/PD topology (2 GPUs) — recommended
│   ├── asr_speechllm_disagg.sh# E/P/D topology (3 GPUs) — high throughput
│   ├── asr_8gpu_replicas.sh   # 4× E/PD replicas (8 GPUs) — A100×8 scale-out
│   ├── audio_agg.sh           # Qwen2-Audio E/PD (2 GPUs)
│   ├── audio_disagg.sh        # Qwen2-Audio E/P/D (3 GPUs)
│   └── validate_asr_speechllm_agg.sh  # Smoke-test suite
└── utils/
    ├── protocol.py            # Pydantic request/response types + NIXL metadata
    ├── audio_loader.py        # Async cached audio downloader (HTTP/HTTPS)
    ├── model.py               # SupportedModels registry + construct_mm_data()
    └── args.py                # Shared CLI parsing + vLLM defaults
```

---

## Architecture

```
Audio URL
   │
   ▼
HTTP Frontend (:8000)        ← OpenAI-compatible /v1/chat/completions
   │
   ▼
Processor                    ← tokenizes text, extracts audio URL
   │  (Dynamo round-robin)
   ▼
ASR Encode Worker  (GPU N)
  • downloads audio (16 kHz)
  • Qwen3ASRProcessor → mel spectrogram
  • Qwen3OmniMoeAudioEncoder (encoder + proj1 + proj2)
  • creates NIXL descriptor
   │  (NIXL RDMA — GPU-to-GPU, zero-copy)
   ▼
vLLM Worker  (GPU M)         ← receives audio embeddings via RDMA
  • injects embeddings at <|AUDIO|> token positions
  • runs LLM decoder (continuous batching, PagedAttention)
  • streams CompletionOutput tokens
   │
   ▼
HTTP Response (JSON / SSE)
```

### Topology options

| Name | Script | GPUs | Best for |
|---|---|---|---|
| E/PD aggregated | `asr_speechllm_agg.sh` | 2 | Low-latency, streaming-first |
| E/P/D disaggregated | `asr_speechllm_disagg.sh` | 3 | High-throughput batch |
| 4× E/PD replicas | `asr_8gpu_replicas.sh` | 8 | A100×8 scale-out |

---

## Supported Models

| Model | Size | Notes |
|---|---|---|
| `Qwen/Qwen3-ASR-1.7B` | 1.7B | Default, recommended |
| `Qwen/Qwen3-ASR-0.6B` | 0.6B | Faster, lower VRAM (~1.2 GB fp16) |
| `Qwen/Qwen2-Audio-7B-Instruct` | 7B | Older audio model, use `audio_agg.sh` |

---

## Installation (one-time)

### Requirements

| Item | Minimum |
|---|---|
| GPUs | 1× NVIDIA GPU (A100 recommended) |
| CUDA | 12.x |
| Python | 3.10 / 3.11 / 3.12 |
| RAM | 32 GB system memory |
| Disk | 20 GB free (model + packages) |

### Quick setup (no Docker, no sudo)

`install.sh` creates a local venv, downloads etcd and nats-server binaries, installs all Python packages, and pre-downloads the model:

```bash
cd examples/multimodal
bash launch/install.sh
# Optional flags:
#   --venv /custom/venv/path
#   --model Qwen/Qwen3-ASR-0.6B
```

After it finishes, activate the venv for all subsequent commands:

```bash
source ./venv/bin/activate
```

### Manual installation

If you prefer to manage your own environment:

```bash
# 1. Python packages
pip install uv
pip install ai-dynamo 'vllm[audio]' accelerate cupy-cuda12x uvloop safetensors huggingface_hub

# 2. vllm-omni — REQUIRED for Qwen3-ASR
#    Provides: vllm.transformers_utils.configs.qwen3_asr
#              vllm.model_executor.models.qwen3_omni_moe_thinker
pip install vllm-omni==0.16.0rc1 || {
    # Fall back to building from source if not on PyPI
    git clone --depth 1 --branch v0.16.0rc1 https://github.com/vllm-project/vllm-omni.git /tmp/vllm-omni
    pip install /tmp/vllm-omni
}
python -c "from vllm.transformers_utils.configs.qwen3_asr import Qwen3ASRConfig; print('vllm-omni OK')"

# 3. Infrastructure (etcd + NATS)
#    Via Docker:
docker run -d --name etcd --restart unless-stopped -p 2379:2379 \
    quay.io/coreos/etcd:v3.5.0 etcd \
    --listen-client-urls http://0.0.0.0:2379 \
    --advertise-client-urls http://localhost:2379
docker run -d --name nats --restart unless-stopped -p 4222:4222 nats:latest

#    Or bare-metal (same binaries install.sh downloads):
#    etcd v3.5.21 from github.com/etcd-io/etcd/releases
#    nats-server v2.12.4 from github.com/nats-io/nats-server/releases

# 4. Pre-download model (optional but recommended — avoids startup timeout)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-ASR-1.7B', ignore_patterns=['*.msgpack','*.h5'])
"
```

---

## Running (2 GPUs — standard E/PD)

```bash
cd examples/multimodal
source ./venv/bin/activate   # if using install.sh venv

bash launch/asr_speechllm_agg.sh
# or the smaller model:
bash launch/asr_speechllm_agg.sh --model Qwen/Qwen3-ASR-0.6B
```

Processes started:

| Process | GPU | Port |
|---|---|---|
| `dynamo.frontend` | CPU | 8000 (HTTP) |
| `processor.py` | CPU | — |
| `asr_encode_worker.py` | GPU 0 | — |
| `worker.py --worker-type prefill` | GPU 1 | 20097 (NIXL) |

---

## Running (3 GPUs — E/P/D disaggregated)

Use when prefill and decode contend for GPU compute at high batch sizes:

```bash
bash launch/asr_speechllm_disagg.sh
```

Adds a dedicated decode worker on GPU 2 with separate KV event and NIXL ports.

---

## Running (8 GPUs — A100×8 scale-out)

Launches four independent E/PD replica pairs. Dynamo's `round_robin` routing distributes requests across all replicas automatically.

```bash
bash launch/asr_8gpu_replicas.sh
# optional flags:
#   --model Qwen/Qwen3-ASR-0.6B
#   --tp 2          (tensor parallelism per LLM worker, uses 2 GPUs per pair)
```

GPU assignment:

| GPU | Process |
|---|---|
| 0 | Encoder replica 1 |
| 1 | LLM (PD) replica 1 |
| 2 | Encoder replica 2 |
| 3 | LLM (PD) replica 2 |
| 4 | Encoder replica 3 |
| 5 | LLM (PD) replica 3 |
| 6 | Encoder replica 4 |
| 7 | LLM (PD) replica 4 |

Each replica pair uses its own `DYN_VLLM_KV_EVENT_PORT` and `VLLM_NIXL_SIDE_CHANNEL_PORT` to avoid conflicts.

---

## Checking Readiness

```bash
# Poll until the pipeline is ready
until curl -sf http://localhost:8000/v1/models > /dev/null; do
    echo "Waiting..."; sleep 5
done
curl http://localhost:8000/v1/models | python3 -m json.tool
```

---

## Sending Requests

### Non-streaming transcription

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-ASR-1.7B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"}},
        {"type": "text", "text": "Transcribe this audio."}
      ]
    }],
    "max_tokens": 256,
    "temperature": 0.0,
    "stream": false
  }' | python3 -m json.tool
```

### Streaming transcription (SSE)

```bash
curl -sN -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-ASR-1.7B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"}},
        {"type": "text", "text": "Transcribe this audio."}
      ]
    }],
    "max_tokens": 256,
    "temperature": 0.0,
    "stream": true
  }'
```

### Translation (pass a system prompt via the text field)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-ASR-1.7B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"}},
        {"type": "text", "text": "Transcribe this audio and translate it to English."}
      ]
    }],
    "max_tokens": 256,
    "temperature": 0.0
  }' | python3 -m json.tool
```

---

## Validation

```bash
# Run the full smoke-test suite against a running pipeline
bash launch/validate_asr_speechllm_agg.sh

# Custom port or model
bash launch/validate_asr_speechllm_agg.sh --port 8000 --model Qwen/Qwen3-ASR-0.6B
```

The suite runs 6 checks: frontend health, model discovery, non-streaming ASR, streaming ASR, invalid-URL error handling, and text-only request handling.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | all | GPU(s) for this process |
| `DYN_VLLM_KV_EVENT_PORT` | `20080` | ZMQ port for KV cache events (must be unique per worker) |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | — | NIXL side-channel port (must be unique per worker) |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | auto | Host for NIXL negotiation (auto-resolved from hostname) |
| `DYN_NAMESPACE` | `dynamo` | Dynamo service namespace for etcd/NATS routing |
| `DYN_HTTP_PORT` | `8000` | Frontend HTTP port |

---

## Troubleshooting

**`ModuleNotFoundError: vllm.transformers_utils.configs.qwen3_asr`**
vllm-omni is not installed. Run:
```bash
pip install vllm-omni==0.16.0rc1
```

**Workers fail to discover each other**
Check etcd is reachable:
```bash
curl http://localhost:2379/health
```

**`NATS connection refused`**
Check NATS is running on port 4222:
```bash
docker ps | grep nats   # if using Docker
# or check the log: cat /tmp/dynamo-nats.log
```

**Encoder falls back to numpy (slow NIXL)**
cupy is not installed or CUDA is unavailable:
```bash
pip install cupy-cuda12x
python -c "import cupy; print(cupy.cuda.is_available())"
```

**Port conflict between replicas**
Each worker must have a unique `DYN_VLLM_KV_EVENT_PORT` and `VLLM_NIXL_SIDE_CHANNEL_PORT`. Check `asr_8gpu_replicas.sh` uses ports 20081–20084 and 20091–20094 respectively.

**CUDA out of memory on a single A100**
Switch to the 0.6B model:
```bash
bash launch/asr_speechllm_agg.sh --model Qwen/Qwen3-ASR-0.6B
```

**Model list empty after startup**
The processor registers the model only after the encoder worker is ready. Check the processor log:
```bash
# look for: "Waiting for Encoder Worker Instances ..." → "Starting to serve the endpoint..."
```

---

## Further Reading

- `speechllm_architecture.md` — comparison of Triton, vLLM+Dynamo, SGLang, TRT-LLM, RIVA, and edge-VAD architectures for streaming ASR
- `deploy/lora/README.md` — LoRA adapter hot-swap for Qwen multimodal models
- `launch/lora/README.md` — LoRA launch scripts
