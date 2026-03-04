---
title: "Review: Streaming SpeechLLM (SALM Duplex) Architecture for ASR & Translation"
date: 2026-03-02
tags: ["SpeechLLM", "ASR", "SALM", "streaming", "Dynamo", "vLLM", "Triton", "TRT-LLM", "architecture", "translation"]
translationKey: "review-salm-duplex-2026-03-02"
draft: false
---

# Streaming SpeechLLM Architecture: ASR & Translation

A comparison of serving architectures for **streaming SpeechLLM (SALM duplex)** — covering ASR and translation use cases across Triton, vLLM, Dynamo, RIVA, SGLang, TRT-LLM, Faster-Whisper, Ray Serve, and hybrid edge approaches.

---

## Architecture Comparison

### 1. Triton Only
```
Audio → Triton Ensemble:
  [Preprocessor] → [Encoder model] → [LLM (BLS or TRT-LLM backend)] → Tokens
```

**Strengths:**
- Mature, production-grade, single system to operate
- Ensemble pipelines chain stages cleanly
- gRPC streaming support out of the box
- TRT-LLM backend gives excellent throughput on NVIDIA hardware

**Weaknesses:**
- LLM efficiency is lower than vLLM (no PagedAttention-style dynamic KV management natively)
- Encoder → LLM data transfer is in-process but less flexible for inter-GPU RDMA
- Scaling encoder and LLM independently is hard (tied to ensemble topology)
- State management for chunked streaming audio across requests is awkward in BLS

---

### 2. Triton + vLLM
```
Audio → Triton (encoder/preprocessor) --HTTP/gRPC--> vLLM (LLM decoder) → Tokens
```

**Strengths:**
- Best-in-class LLM decoder (PagedAttention, continuous batching)
- Clean separation: Triton handles audio, vLLM handles generation

**Weaknesses:**
- **Serialization overhead**: Embeddings cross process boundaries over gRPC/HTTP — not GPU-to-GPU direct
- No unified scheduler — Triton and vLLM have no awareness of each other's load
- Chunked streaming audio requires explicit state management between the two systems

---

### 3. vLLM + Dynamo (current approach)
```
HTTP → Frontend → Processor
                → ASR Encode Worker (GPU 0) --NIXL RDMA--> PD Worker/vLLM (GPU 1) → Tokens
```

**Strengths:**
- **NIXL RDMA**: GPU-to-GPU embedding transfer with near-zero latency (no serialization, no CPU copy)
- **Independent scaling**: Encoder and LLM workers can be on separate GPUs, scaled independently
- **E/PD is ideal for ASR**: ASR input is processed once (no iterative prefill benefit from P/D separation), so E/PD avoids unnecessary round-trips
- **Streaming-first design**: explicitly designed for chunked audio in future iterations
- **KV-aware routing**: useful for multi-turn sessions (translation w/ context)
- Unified observability, dynamic GPU scheduling via planner

**Weaknesses:**
- Most complex operationally (etcd/NATS/ZMQ, service discovery)
- Still pre-release (1.0.0 not on PyPI yet)
- Overkill for a single-GPU deployment

---

## Streaming-Specific Considerations

For **real-time streaming ASR** (meeting transcription, live translation), the critical path is:

```
Chunk 1 → Encode → Transfer → Decode partial tokens  →  stream output
Chunk 2 → Encode → Transfer → Append to KV cache     →  continue stream
...
```

| Property | Triton Only | Triton+vLLM | vLLM+Dynamo |
|---|---|---|---|
| Encoder→LLM transfer | In-process | gRPC (CPU) | NIXL RDMA (GPU-GPU) |
| Incremental KV updates | Difficult | Not native | Native (planned) |
| Encoder/LLM GPU isolation | No | Partial | Yes |
| First-token latency | Low–Medium | Medium | Low |
| Multi-stream throughput | Medium | Medium–High | High |
| Operational simplicity | High | Medium | Low |

---

## Recommendation

| Use Case | Best Fit |
|---|---|
| Single GPU, simple deployment | **Triton only** (with TRT-LLM backend) |
| Max LLM throughput, existing Triton infra | **Triton + vLLM** |
| Multi-GPU, high concurrency, streaming-first | **vLLM + Dynamo** (current design) |

The current **E/PD Dynamo design is the right long-term architecture** for a streaming SpeechLLM at scale — especially once chunked audio streaming is wired in (the encoder worker already accepts incremental requests). The main practical pain point right now is the operational complexity of the stack and the fact that 1.0.0 isn't packaged yet.

The key architectural gap in the current implementation that would unlock true streaming is **VAD (Voice Activity Detection)** at the frontend to segment audio into chunks, feeding them incrementally to the encode worker rather than waiting for the full utterance.

---

## Additional Candidates

### 4. NVIDIA RIVA
```
Mic/Audio → RIVA ASR Service (streaming gRPC) → Text → RIVA NMT → Translation
```
- Purpose-built for streaming ASR + translation pipelines
- Chunk-based streaming via gRPC bidirectional stream (true word-by-word output)
- TensorRT-optimized conformer/Citrinet/Whisper models
- Handles VAD, diarization, punctuation natively
- **Limitation**: not LLM-based — traditional encoder-decoder, not autoregressive generative model
- **Best for**: production deployments where accuracy of established models is sufficient and you don't need an LLM backbone

---

### 5. SGLang + Dynamo (or standalone)
```
Audio → Encode Worker → SGLang (prefill+decode) with RadixAttention
```
- Available as a Dynamo backend (`components/src/dynamo/sglang/`)
- **RadixAttention**: automatic prefix KV cache sharing — great for translation where system prompt is long/repeated
- Generally comparable throughput to vLLM, sometimes better for high-concurrency with shared prefixes
- More mature multimodal support in recent releases (0.4+)
- Streaming-first design with native SSE
- **Best for**: translation with long shared context/system prompts, or when you want an alternative to vLLM's PagedAttention

---

### 6. TRT-LLM + Triton (without Dynamo)
```
Audio → Triton Preprocessor → TRT-LLM backend (in-flight batching) → Tokens
```
- Maximum GPU efficiency — TRT-LLM is the fastest LLM inference engine on NVIDIA hardware
- In-flight batching, paged KV cache, speculative decoding all built in
- First-class Triton integration (no glue code needed)
- **Limitation**: model compilation step is non-trivial, especially for new models like Qwen3-ASR
- Dynamo already supports TRT-LLM as a backend if you want RDMA + disaggregation later
- **Best for**: known stable models in production where you want peak throughput

---

### 7. Faster-Whisper / CTranslate2
```
Audio → Faster-Whisper (encoder+decoder, CTranslate2 backend) → Tokens streamed
```
- Specifically for Whisper-family models (encoder-decoder, not decoder-only)
- 4× faster than OpenAI Whisper with same accuracy, ~half memory
- True streaming: yields tokens as they're generated
- Supports batching, beam search, VAD integration (`silero-vad`)
- **Limitation**: only Whisper-architecture models; Qwen3-ASR is decoder-only LLM style
- **Best for**: pure ASR without translation, or when the model is Whisper-based and you want simplicity

---

### 8. Ray Serve (composition layer)
```
Audio → Ray Serve DAG:
  [VAD actor] → [Encoder replica pool] → [vLLM replica pool] → Tokens
```
- Ray Serve handles inter-service communication, autoscaling, and load balancing
- Can wrap any backend (vLLM, TRT-LLM, Faster-Whisper) as Ray actors
- Built-in replica autoscaling based on request queue depth
- **Limitation**: no RDMA (embeddings go through Ray object store, not GPU-to-GPU), adds latency vs Dynamo's NIXL
- **Best for**: teams already on Ray ecosystem, multi-model pipelines where flexibility > raw latency

---

### 9. Edge-VAD + Cloud LLM (Hybrid Architecture)
```
Client (WebRTC/WebSocket):
  [Mic] → [VAD + feature extraction] → compressed features → Cloud
Cloud:
  Dynamo/vLLM worker pool (receive features, run LLM decoder only)
```
- Offload VAD and Fbank feature extraction to edge (runs on CPU/mobile efficiently)
- Only send compressed acoustic features over network (much smaller than raw audio)
- Cloud only runs the expensive LLM decoder portion
- Eliminates encoder GPU entirely from the cloud side
- **Best for**: mobile/meeting apps, when network bandwidth is a concern

---

## Full Summary Matrix

| Architecture | Latency | Throughput | Streaming | Complexity | LLM Quality |
|---|---|---|---|---|---|
| RIVA | Very Low | High | Native | Low | Traditional ASR |
| SGLang + Dynamo | Low | High | Good | High | Full LLM |
| TRT-LLM + Triton | Low | Highest | Good | Medium | Full LLM |
| Faster-Whisper | Low | Medium | Native | Low | Whisper only |
| Ray Serve | Medium | Medium | Via actors | Medium | Any backend |
| Edge-VAD + Cloud | Lowest* | High | Native | High | Full LLM |
| vLLM + Dynamo (current) | Low | High | Planned | High | Full LLM |

---

## Recommendation for Qwen3-ASR

Given the use of **Qwen3-ASR** (decoder-only LLM with audio encoder):

1. **Short term / simplest**: TRT-LLM + Triton — compile the model once, maximum efficiency
2. **Best long-term for streaming**: vLLM + Dynamo (current work) — once chunked audio streaming is implemented, NIXL RDMA makes the encoder→decoder path nearly free
3. **If translation context matters**: SGLang + Dynamo — RadixAttention prefix sharing pays dividends when you have long shared system prompts for translation tasks
4. **If you need truly real-time** (sub-300ms first token for live speech): Edge-VAD hybrid — remove the encoder from the hot path entirely
