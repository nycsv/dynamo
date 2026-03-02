#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Validation / smoke-test script for the Qwen3-ASR SpeechLLM pipeline.
#
# Tests the full E/PD or E/P/D pipeline end-to-end:
#   1. Health check (frontend reachable)
#   2. Model discovery (/v1/models)
#   3. ASR transcription with a public audio sample
#   4. Response structure validation
#   5. Streaming response validation
#   6. Error handling (invalid audio URL)
#
# Prerequisites:
#   A running ASR SpeechLLM pipeline via asr_speechllm_agg.sh or asr_speechllm_disagg.sh
#
# Usage:
#   ./validate_asr_speechllm_agg.sh                          # defaults: port=8000
#   ./validate_asr_speechllm_agg.sh --port 9000              # custom frontend port
#   ./validate_asr_speechllm_agg.sh --model Qwen/Qwen3-ASR-0.6B  # custom model

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────

FRONTEND_PORT="${DYN_HTTP_PORT:-8000}"
MODEL_NAME=""
# Public Qwen audio samples for testing
AUDIO_URL="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"
CURL_TIMEOUT=120
PASS=0
FAIL=0
SKIP=0
TOTAL=0

# ── Parse args ───────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)          FRONTEND_PORT=$2; shift 2 ;;
        --model)         MODEL_NAME=$2; shift 2 ;;
        --audio-url)     AUDIO_URL=$2; shift 2 ;;
        --timeout)       CURL_TIMEOUT=$2; shift 2 ;;
        -h|--help)
            cat <<USAGE
Usage: $0 [OPTIONS]

Validates a running Qwen3-ASR SpeechLLM pipeline (E/PD or E/P/D).

Options:
  --port <port>       Frontend HTTP port (default: 8000)
  --model <name>      Model name to use in requests (auto-detected if omitted)
  --audio-url <url>   Audio URL for transcription test
  --timeout <secs>    Curl timeout per request (default: 120)
  -h, --help          Show this help message

Examples:
  # Start the pipeline first:
  bash asr_speechllm_agg.sh --model Qwen/Qwen3-ASR-1.7B

  # Then validate:
  ./validate_asr_speechllm_agg.sh
  ./validate_asr_speechllm_agg.sh --model Qwen/Qwen3-ASR-0.6B
USAGE
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

FRONTEND="http://localhost:$FRONTEND_PORT"

# ── Helpers ──────────────────────────────────────────────────────────────

pass() { PASS=$((PASS + 1)); TOTAL=$((TOTAL + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); TOTAL=$((TOTAL + 1)); echo "  FAIL: $1"; }
skip() { SKIP=$((SKIP + 1)); TOTAL=$((TOTAL + 1)); echo "  SKIP: $1"; }

# Curl wrapper with timeout
api() {
    curl -sf --max-time "$CURL_TIMEOUT" "$@" 2>/dev/null
}

check_json_field() {
    local json=$1 field=$2 expected=$3 name=$4
    local actual
    actual=$(echo "$json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('$field',''))" 2>/dev/null || echo "PARSE_ERROR")
    if [[ "$actual" == "$expected" ]]; then
        pass "$name"
    else
        fail "$name (expected '$expected', got '$actual')"
    fi
}

# ── Banner ───────────────────────────────────────────────────────────────

echo "=================================================="
echo "Qwen3-ASR SpeechLLM Pipeline Validation"
echo "=================================================="
echo "Frontend:  $FRONTEND"
echo "Model:     ${MODEL_NAME:-<auto-detect>}"
echo "Audio URL: $AUDIO_URL"
echo "Timeout:   ${CURL_TIMEOUT}s"
echo "=================================================="

# ── 1. Frontend health ──────────────────────────────────────────────────

echo ""
echo "[1/6] Checking frontend health..."
if api "$FRONTEND/v1/models" > /dev/null; then
    pass "Frontend is reachable"
else
    fail "Frontend is NOT reachable at $FRONTEND"
    echo "  Ensure asr_speechllm_agg.sh is running. Aborting."
    exit 1
fi

# ── 2. Model discovery ──────────────────────────────────────────────────

echo ""
echo "[2/6] Discovering model..."
MODELS_RESP=$(api "$FRONTEND/v1/models" || echo '{}')

if [[ -z "$MODEL_NAME" ]]; then
    # Auto-detect the model name from the running server
    MODEL_NAME=$(echo "$MODELS_RESP" | python3 -c "
import sys, json
data = json.load(sys.stdin).get('data', [])
# Prefer ASR models, fall back to first model
for m in data:
    if 'asr' in m['id'].lower() or 'audio' in m['id'].lower():
        print(m['id']); sys.exit(0)
if data:
    print(data[0]['id'])
" 2>/dev/null || echo "")
fi

if [[ -n "$MODEL_NAME" ]]; then
    pass "Model detected: $MODEL_NAME"
else
    fail "Could not detect model name from /v1/models"
    echo "  Response: $MODELS_RESP"
    exit 1
fi

# Verify the model is in the models list
MODEL_FOUND=$(echo "$MODELS_RESP" | python3 -c "
import sys, json
data = json.load(sys.stdin).get('data', [])
ids = [m['id'] for m in data]
print('yes' if '$MODEL_NAME' in ids else 'no')
" 2>/dev/null || echo "no")

if [[ "$MODEL_FOUND" == "yes" ]]; then
    pass "Model '$MODEL_NAME' is listed in /v1/models"
else
    fail "Model '$MODEL_NAME' NOT found in /v1/models"
fi

# ── 3. ASR transcription (non-streaming) ────────────────────────────────

echo ""
echo "[3/6] Testing ASR transcription (non-streaming)..."
RESP=$(api -X POST "$FRONTEND/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL_NAME\",
      \"messages\": [{\"role\": \"user\", \"content\": [
        {\"type\": \"audio_url\", \"audio_url\": {\"url\": \"$AUDIO_URL\"}},
        {\"type\": \"text\", \"text\": \"Transcribe this audio.\"}
      ]}],
      \"max_tokens\": 256,
      \"temperature\": 0.0,
      \"stream\": false
    }" || echo '{"error":"request_failed"}')

# Check response has choices
if echo "$RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'choices' in d and len(d['choices']) > 0
content = d['choices'][0].get('message', {}).get('content', '')
assert len(content) > 0, 'Empty transcription'
print(content[:200])
" 2>/dev/null; then
    pass "ASR transcription returned non-empty result"
else
    fail "ASR transcription failed or returned empty result"
    echo "  Response: $(echo "$RESP" | head -c 500)"
fi

# Check response has usage stats
if echo "$RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
usage = d.get('usage', {})
assert usage.get('prompt_tokens', 0) > 0, 'No prompt tokens'
assert usage.get('completion_tokens', 0) > 0, 'No completion tokens'
" 2>/dev/null; then
    pass "Response includes valid usage statistics"
else
    fail "Response missing valid usage statistics"
fi

# ── 4. ASR transcription (streaming) ────────────────────────────────────

echo ""
echo "[4/6] Testing ASR transcription (streaming)..."
STREAM_RESP=$(curl -sN --max-time "$CURL_TIMEOUT" \
    -X POST "$FRONTEND/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL_NAME\",
      \"messages\": [{\"role\": \"user\", \"content\": [
        {\"type\": \"audio_url\", \"audio_url\": {\"url\": \"$AUDIO_URL\"}},
        {\"type\": \"text\", \"text\": \"Transcribe this audio.\"}
      ]}],
      \"max_tokens\": 256,
      \"temperature\": 0.0,
      \"stream\": true
    }" 2>/dev/null || echo "")

# Verify SSE format and content
if echo "$STREAM_RESP" | python3 -c "
import sys

lines = sys.stdin.read().strip().split('\n')
data_lines = [l for l in lines if l.startswith('data: ') and l != 'data: [DONE]']
assert len(data_lines) > 0, 'No data lines in stream'

import json
# Check first chunk has valid structure
first = json.loads(data_lines[0].removeprefix('data: '))
assert 'choices' in first, 'First chunk missing choices'

# Collect all content deltas
content = ''
for line in data_lines:
    chunk = json.loads(line.removeprefix('data: '))
    delta = chunk.get('choices', [{}])[0].get('delta', {})
    content += delta.get('content', '')

assert len(content) > 0, 'No content in stream'
print(f'Streamed {len(data_lines)} chunks, content: {content[:200]}')
" 2>/dev/null; then
    pass "Streaming ASR transcription works"
else
    fail "Streaming ASR transcription failed"
    echo "  First 500 chars: $(echo "$STREAM_RESP" | head -c 500)"
fi

# Check the stream ends with [DONE]
if echo "$STREAM_RESP" | grep -q 'data: \[DONE\]'; then
    pass "Stream properly terminates with [DONE]"
else
    fail "Stream missing [DONE] terminator"
fi

# ── 5. Error handling (invalid audio URL) ────────────────────────────────

echo ""
echo "[5/6] Testing error handling (invalid audio URL)..."
ERR_RESP=$(curl -s --max-time "$CURL_TIMEOUT" \
    -w "\n%{http_code}" \
    -X POST "$FRONTEND/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL_NAME\",
      \"messages\": [{\"role\": \"user\", \"content\": [
        {\"type\": \"audio_url\", \"audio_url\": {\"url\": \"https://nonexistent.invalid/audio.wav\"}},
        {\"type\": \"text\", \"text\": \"Transcribe this audio.\"}
      ]}],
      \"max_tokens\": 64,
      \"temperature\": 0.0
    }" 2>/dev/null || echo -e "\n000")

HTTP_CODE=$(echo "$ERR_RESP" | tail -1)
ERR_BODY=$(echo "$ERR_RESP" | sed '$d')

# We expect either an HTTP error (4xx/5xx) or an error in the JSON body
if [[ "$HTTP_CODE" -ge 400 ]] || echo "$ERR_BODY" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'error' in d or d.get('choices', [{}])[0].get('finish_reason') == 'error'
" 2>/dev/null; then
    pass "Invalid audio URL returns error (HTTP $HTTP_CODE)"
else
    # Some pipelines may still return 200 with an error message in content
    fail "Invalid audio URL did not return expected error (HTTP $HTTP_CODE)"
    echo "  Response: $(echo "$ERR_BODY" | head -c 300)"
fi

# ── 6. Request without audio (text-only fallback) ────────────────────────

echo ""
echo "[6/6] Testing text-only request handling..."
TEXT_RESP=$(curl -s --max-time "$CURL_TIMEOUT" \
    -w "\n%{http_code}" \
    -X POST "$FRONTEND/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL_NAME\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Hello, what model are you?\"}],
      \"max_tokens\": 64,
      \"temperature\": 0.0
    }" 2>/dev/null || echo -e "\n000")

TEXT_HTTP_CODE=$(echo "$TEXT_RESP" | tail -1)
TEXT_BODY=$(echo "$TEXT_RESP" | sed '$d')

# ASR models may or may not support text-only input
if [[ "$TEXT_HTTP_CODE" == "200" ]]; then
    if echo "$TEXT_BODY" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'choices' in d and len(d['choices']) > 0
" 2>/dev/null; then
        pass "Text-only request returns valid response (HTTP 200)"
    else
        fail "Text-only request returned 200 but invalid body"
    fi
elif [[ "$TEXT_HTTP_CODE" -ge 400 ]]; then
    pass "Text-only request correctly rejected (HTTP $TEXT_HTTP_CODE) — ASR model requires audio"
else
    fail "Text-only request returned unexpected HTTP $TEXT_HTTP_CODE"
fi

# ── Summary ──────────────────────────────────────────────────────────────

echo ""
echo "=================================================="
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped (out of $TOTAL)"
echo "=================================================="

if [[ $FAIL -gt 0 ]]; then
    echo ""
    echo "Some tests FAILED. Check the pipeline logs for details."
    exit 1
else
    echo ""
    echo "All tests PASSED!"
fi
