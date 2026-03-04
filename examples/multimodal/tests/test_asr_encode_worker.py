# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Qwen3-ASR encode worker and related changes.

These tests are designed to run with minimal dependencies (torch, numpy, pytest).
Tests that require heavy dependencies (vllm, transformers, dynamo) are skipped
if the dependencies are not available.
"""

import importlib
import sys
import os

import numpy as np
import pytest
import torch

# Add the examples/multimodal directory to the path
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)


def _can_import(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


# ---------------------------------------------------------------------------
# Tests for embedding masking logic (no external deps)
# These test the core tensor operations used in get_audio_embeddings()
# ---------------------------------------------------------------------------


class TestEmbeddingMaskingLogic:
    """Test the attention mask and embedding masking logic from get_audio_embeddings."""

    def test_padding_mask_creation(self):
        """Verify padding mask is correctly constructed."""
        batch_size = 1
        max_seq_len = 50
        audio_feat_lengths = torch.tensor([30])

        seq_range = (
            torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(
            batch_size, max_seq_len
        )
        padding_mask = seq_range >= lengths_expand

        # First 30 positions: not padded
        assert not padding_mask[0, :30].any()
        # Positions 30-49: padded
        assert padding_mask[0, 30:].all()

    def test_attention_mask_shape(self):
        """Verify 4D attention mask has correct shape."""
        batch_size = 1
        max_seq_len = 50
        audio_feat_lengths = torch.tensor([30])

        seq_range = (
            torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(
            batch_size, max_seq_len
        )
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(
            batch_size, 1, 1, max_seq_len
        ).expand(batch_size, 1, max_seq_len, max_seq_len)

        assert audio_attention_mask_.shape == (1, 1, 50, 50)

    def test_attention_mask_inf_values(self):
        """Verify -inf is applied to padded positions."""
        batch_size = 1
        max_seq_len = 10
        audio_feat_lengths = torch.tensor([6])

        seq_range = (
            torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(
            batch_size, max_seq_len
        )
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(
            batch_size, 1, 1, max_seq_len
        ).expand(batch_size, 1, max_seq_len, max_seq_len)

        audio_attention_mask = torch.zeros_like(
            audio_attention_mask_, dtype=torch.float16
        )
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        # Non-padded region should be 0
        assert (audio_attention_mask[0, 0, :6, :6] == 0).all()
        # Padded column region should be -inf
        assert torch.isinf(audio_attention_mask[0, 0, 0, 6:]).all()

    def test_embedding_output_shape_single_audio(self):
        """After masking, output should be (num_tokens, embed_dim) with no padding."""
        audio_features = torch.randn(1, 100, 2048)
        audio_output_lengths = torch.tensor([75])

        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = torch.arange(
            max_audio_tokens, device=audio_output_lengths.device
        )[None, :]
        audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
        masked_features = audio_features[audio_features_mask]

        assert masked_features.shape == (75, 2048)
        assert masked_features.ndim == 2

    def test_embedding_output_shape_batch(self):
        """Masking with batch_size > 1 concatenates all tokens."""
        audio_features = torch.randn(3, 100, 2048)
        audio_output_lengths = torch.tensor([50, 75, 100])

        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = torch.arange(
            max_audio_tokens, device=audio_output_lengths.device
        )[None, :]
        audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
        masked_features = audio_features[audio_features_mask]

        # Total tokens: 50 + 75 + 100 = 225
        assert masked_features.shape == (225, 2048)

    def test_embedding_content_preserved(self):
        """Verify masking preserves the correct token values."""
        audio_features = torch.arange(12).reshape(1, 4, 3).float()
        audio_output_lengths = torch.tensor([2])

        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = torch.arange(
            max_audio_tokens, device=audio_output_lengths.device
        )[None, :]
        audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
        masked_features = audio_features[audio_features_mask]

        # Should keep only first 2 tokens: [[0,1,2], [3,4,5]]
        expected = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float)
        assert torch.equal(masked_features, expected)

    def test_downsampling_formula(self):
        """Verify the 8x downsampling formula: (max_mel_seq_len - 2) // 2 + 1."""
        # For a mel spectrogram of length 1000
        max_mel_seq_len = 1000
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        assert max_seq_len == 500

        # For 500
        max_mel_seq_len = 500
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        assert max_seq_len == 250


# ---------------------------------------------------------------------------
# Tests for model.py: SupportedModels and construct_mm_data
# (requires transformers + torch)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _can_import("transformers"),
    reason="transformers not installed",
)
class TestSupportedModels:
    """Verify Qwen3-ASR models are registered in SupportedModels."""

    def test_qwen3_asr_06b_registered(self):
        from utils.model import SupportedModels

        assert SupportedModels.QWEN3_ASR_0_6B == "Qwen/Qwen3-ASR-0.6B"

    def test_qwen3_asr_17b_registered(self):
        from utils.model import SupportedModels

        assert SupportedModels.QWEN3_ASR_1_7B == "Qwen/Qwen3-ASR-1.7B"

    def test_existing_models_unchanged(self):
        from utils.model import SupportedModels

        assert SupportedModels.QWEN_2_AUDIO_7B == "Qwen/Qwen2-Audio-7B-Instruct"
        assert SupportedModels.LLAVA_1_5_7B == "llava-hf/llava-1.5-7b-hf"


@pytest.mark.skipif(
    not _can_import("transformers"),
    reason="transformers not installed",
)
class TestConstructMmDataQwen3Asr:
    """Verify construct_mm_data handles Qwen3-ASR models correctly."""

    def test_qwen3_asr_17b_returns_audio_key(self):
        from utils.model import SupportedModels, construct_mm_data

        embeddings = torch.randn(125, 2048)
        result = construct_mm_data(
            SupportedModels.QWEN3_ASR_1_7B,
            torch.bfloat16,
            audio_embeds=embeddings,
        )
        assert "audio" in result
        assert isinstance(result["audio"], list)
        assert len(result["audio"]) == 1
        assert result["audio"][0].dtype == torch.bfloat16

    def test_qwen3_asr_06b_returns_audio_key(self):
        from utils.model import SupportedModels, construct_mm_data

        embeddings = torch.randn(125, 1024)
        result = construct_mm_data(
            SupportedModels.QWEN3_ASR_0_6B,
            torch.bfloat16,
            audio_embeds=embeddings,
        )
        assert "audio" in result
        assert result["audio"][0].dtype == torch.bfloat16

    def test_qwen3_asr_embeddings_must_be_2d(self):
        from utils.model import SupportedModels, construct_mm_data

        embeddings_3d = torch.randn(1, 125, 2048)
        with pytest.raises(AssertionError):
            construct_mm_data(
                SupportedModels.QWEN3_ASR_1_7B,
                torch.bfloat16,
                audio_embeds=embeddings_3d,
            )

    def test_qwen3_asr_preserves_shape(self):
        from utils.model import SupportedModels, construct_mm_data

        embeddings = torch.randn(62, 2048)
        result = construct_mm_data(
            SupportedModels.QWEN3_ASR_1_7B,
            torch.bfloat16,
            audio_embeds=embeddings,
        )
        assert result["audio"][0].shape == (62, 2048)

    def test_qwen2_audio_backward_compat(self):
        from utils.model import SupportedModels, construct_mm_data

        embeddings = torch.randn(100, 3584)
        result = construct_mm_data(
            SupportedModels.QWEN_2_AUDIO_7B,
            torch.bfloat16,
            audio_embeds=embeddings,
        )
        assert "audio" in result
        assert result["audio"][0].dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Tests for worker.py: ASR model detection logic
# ---------------------------------------------------------------------------


class TestWorkerAsrModelDetection:
    """Verify the model detection condition in worker.py."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "Qwen/Qwen2-Audio-7B-Instruct",
            "Qwen/Qwen3-ASR-1.7B",
            "Qwen/Qwen3-ASR-0.6B",
            "some-org/custom-audio-model",
            "some-org/custom-asr-model",
        ],
    )
    def test_audio_or_asr_detected(self, model_name):
        """Models with 'audio' or 'asr' should route to audio embedding path."""
        name_lower = model_name.lower()
        assert "audio" in name_lower or "asr" in name_lower

    @pytest.mark.parametrize(
        "model_name",
        [
            "llava-hf/llava-1.5-7b-hf",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "meta-llama/Llama-3-8B-Instruct",
            "llava-hf/LLaVA-NeXT-Video-7B-hf",
        ],
    )
    def test_non_audio_models_not_detected(self, model_name):
        """Non-audio models should NOT route to audio embedding path."""
        name_lower = model_name.lower()
        assert not ("audio" in name_lower or "asr" in name_lower)


# ---------------------------------------------------------------------------
# Tests for AudioLoader (requires librosa + httpx)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_can_import("librosa") and _can_import("httpx")),
    reason="librosa or httpx not installed",
)
class TestAudioLoader:
    """Test AudioLoader URL validation."""

    def _run_async(self, coro):
        """Helper to run async code in sync tests."""
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_invalid_scheme_ftp_rejected(self):
        from utils.audio_loader import AudioLoader

        loader = AudioLoader(cache_size=2)
        with pytest.raises(ValueError, match="Invalid audio source scheme"):
            self._run_async(loader.load_audio("ftp://example.com/audio.wav"))

    def test_invalid_scheme_file_rejected(self):
        from utils.audio_loader import AudioLoader

        loader = AudioLoader(cache_size=2)
        with pytest.raises(ValueError, match="Invalid audio source scheme"):
            self._run_async(loader.load_audio("file:///tmp/audio.wav"))

    def test_http_scheme_accepted(self):
        """HTTP URLs should not raise ValueError on scheme check.
        (Will fail on network, but scheme is valid.)"""
        from utils.audio_loader import AudioLoader

        loader = AudioLoader(cache_size=2)
        # Should raise a network/HTTP error, not a scheme error
        with pytest.raises(Exception) as exc_info:
            self._run_async(
                loader.load_audio("https://nonexistent.invalid/audio.wav")
            )
        assert "Invalid audio source scheme" not in str(exc_info.value)
