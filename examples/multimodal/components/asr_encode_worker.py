# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import logging
import os
import signal
import sys
from typing import AsyncIterator, Tuple

import json

import torch
import uvloop
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)
from vllm.transformers_utils.configs.qwen3_asr import Qwen3ASRConfig
from vllm.transformers_utils.processors.qwen3_asr import Qwen3ASRProcessor
from vllm.utils.argparse_utils import FlexibleArgumentParser

import dynamo.nixl_connect as connect
from dynamo.runtime import Client, DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.args import Config, base_parse_args, parse_endpoint
from utils.audio_loader import AudioLoader
from utils.protocol import MyRequestOutput, vLLMMultimodalRequest

configure_dynamo_logging()
logger = logging.getLogger(__name__)

try:
    import cupy as array_module

    if not array_module.cuda.is_available():
        raise ImportError("CUDA is not available.")
    DEVICE = "cuda"
    logger.info("Using cupy for array operations (GPU mode).")
except ImportError as e:
    logger.warning(f"Failed to import cupy, falling back to numpy: {e}.")
    import numpy as array_module

    DEVICE = "cpu"

CACHE_SIZE_MAXIMUM = 8


class AsrEncodeWorker:
    """
    Encode worker for Qwen3-ASR models.

    Extracts audio encoder + projector embeddings from Qwen3-ASR and transfers
    them via NIXL RDMA to a downstream PD worker running the Qwen3 LLM decoder.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        engine_args: AsyncEngineArgs,
        pd_worker_client: Client,
    ) -> None:
        self.pd_worker_client = pd_worker_client
        self.engine_args = engine_args
        self.model = self.engine_args.model

        self.audio_loader = AudioLoader(cache_size=CACHE_SIZE_MAXIMUM)

        # Use vLLM's processor since transformers 4.x doesn't support qwen3_asr.
        # Qwen3ASRProcessor requires a text argument; we pass a dummy placeholder.
        self.audio_processor = Qwen3ASRProcessor.from_pretrained(self.model)
        self._processor_dummy_text = "<|im_start|>"

        # Load config and instantiate only the audio tower (encoder + projector).
        # The LLM decoder runs on the downstream PD worker.
        # vLLM's Qwen3OmniMoeAudioEncoder already includes proj1/proj2, so there is
        # no separate multi_modal_projector to call.
        hf_config = Qwen3ASRConfig.from_pretrained(self.model)
        self.audio_tower = Qwen3OmniMoeAudioEncoder(
            hf_config.thinker_config.audio_config
        )
        self._load_audio_tower_weights(self.model)
        self.audio_tower = self.audio_tower.to(torch.float16).cuda().eval()

    def _load_audio_tower_weights(self, model_id: str) -> None:
        """Load audio_tower weights from the model's safetensors shards."""
        model_dir = snapshot_download(model_id, ignore_patterns=["*.msgpack", "*.h5"])
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        # Collect shard files that contain audio_tower weights
        audio_prefix = "thinker.audio_tower."
        shard_files = {
            shard
            for key, shard in index["weight_map"].items()
            if key.startswith(audio_prefix)
        }

        state_dict = {}
        for shard_file in shard_files:
            shard_path = os.path.join(model_dir, shard_file)
            tensors = load_file(shard_path)
            for key, tensor in tensors.items():
                if key.startswith(audio_prefix):
                    new_key = key[len(audio_prefix):]
                    state_dict[new_key] = tensor

        missing, unexpected = self.audio_tower.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys in audio_tower: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in audio_tower: {unexpected}")

    def get_audio_embeddings(self, processed) -> torch.Tensor:
        """
        Extract encoder + projector output from Qwen3-ASR.

        Uses vLLM's Qwen3OmniMoeAudioEncoder which includes the projector
        (proj1 → proj2) internally.  The forward signature differs from Qwen2-Audio:
          - input_features: (num_mel, seq_len) — batch dim is squeezed
          - feature_lens:   non-padded frame counts, shape (batch,)
          - aftercnn_lens:  post-CNN lengths, computed by _get_feat_extract_output_lengths

        Returns embeddings tensor of shape (num_tokens, embed_dim).
        """
        # (1, num_mel, seq_len) → (num_mel, seq_len)
        input_features = processed.input_features.squeeze(0).to(
            self.audio_tower.dtype, device=self.audio_tower.device
        )
        feature_lens = processed.feature_attention_mask.sum(-1).to(
            device=self.audio_tower.device
        )
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)

        with torch.no_grad():
            # audio_tower returns a flat (total_tokens, embed_dim) tensor
            audio_embeddings = self.audio_tower(
                input_features,
                feature_lens=feature_lens,
                aftercnn_lens=aftercnn_lens,
            )
        return audio_embeddings

    def cleanup(self):
        pass

    async def generate(
        self, request: vLLMMultimodalRequest
    ) -> AsyncIterator[MyRequestOutput]:
        logger.debug(f"Got raw request: {request}")
        if not isinstance(request, vLLMMultimodalRequest):
            if isinstance(request, str):
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)
        logger.debug(f"Received encode request: {{ id: {request.request_id} }}.")

        request_id = request.request_id

        # The following steps encode the requested audio and produce embeddings:
        # 1. Load the audio from the provided URL.
        # 2. Process the audio using the Qwen3-ASR processor.
        # 3. Run the audio through the ASR model's audio encoder.
        # 4. Run the encoder output through the projector.
        # 5. Create a NIXL descriptor for the embeddings.
        # 6. Transfer embeddings to the downstream PD worker via RDMA.
        # 7. Stream the decode response back.

        try:
            audio, sr = await self.audio_loader.load_audio(
                request.multimodal_input.audio_url
            )

            processed = self.audio_processor(
                audio=audio,
                text=self._processor_dummy_text,
                return_tensors="pt",
                sampling_rate=sr,
            )
            audio_embeddings = self.get_audio_embeddings(processed)
            descriptor = connect.Descriptor(audio_embeddings)
            with await self._connector.create_readable(descriptor) as readable:
                request.serialized_request = readable.metadata()
                # Clear the audio URL as hint that the audio is passed as embeddings.
                request.multimodal_input.audio_url = None
                request.embeddings_shape = tuple(audio_embeddings.shape)
                logger.debug(f"Request: {request.model_dump_json()}")

                response_generator = await self.pd_worker_client.round_robin(
                    request.model_dump_json()
                )

                await readable.wait_for_completion()

                async for response in response_generator:
                    output = MyRequestOutput.model_validate_json(response.data())
                    yield MyRequestOutput(
                        request_id=output.request_id,
                        prompt=output.prompt,
                        prompt_token_ids=output.prompt_token_ids,
                        prompt_logprobs=output.prompt_logprobs,
                        outputs=output.outputs,
                        finished=output.finished,
                    ).model_dump_json()

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            raise

    async def async_init(self, runtime: DistributedRuntime):
        logger.info("Startup started.")
        # Create and initialize a dynamo connector for this worker.
        # We'll need this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector()

        logger.info("Startup completed.")

    @classmethod
    def parse_args(cls) -> Tuple[argparse.Namespace, Config]:
        DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")
        DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.encoder.generate"
        DEFAULT_DOWNSTREAM_ENDPOINT = f"dyn://{DYN_NAMESPACE}.llm.generate"

        parser = FlexibleArgumentParser(
            description="Qwen3-ASR based encoder for Dynamo LLM."
        )
        parser.add_argument(
            "--endpoint",
            type=str,
            default=DEFAULT_ENDPOINT,
            help=f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Default: '{DEFAULT_ENDPOINT}'",
        )
        parser.add_argument(
            "--downstream-endpoint",
            type=str,
            default=DEFAULT_DOWNSTREAM_ENDPOINT,
            help=f"The endpoint string of the downstream LLM in 'dyn://namespace.component.endpoint' format. Default: '{DEFAULT_DOWNSTREAM_ENDPOINT}'",
        )

        args, config = base_parse_args(parser)

        return args, config


async def graceful_shutdown(runtime):
    """
    By calling `runtime.shutdown()`, the endpoints will immediately be unavailable.
    However, in-flight requests will still be processed until they are finished.
    After all in-flight requests are finished, the `serve_endpoint` functions will return
    and the engine will be shutdown by Python's garbage collector.
    """
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    # Runtime setup
    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    # worker setup
    args, config = AsrEncodeWorker.parse_args()
    await init(runtime, args, config)


async def init(runtime: DistributedRuntime, args: argparse.Namespace, config: Config):
    """
    Instantiate and serve
    """

    generate_endpoint = runtime.namespace(config.namespace).component(config.component).endpoint(config.endpoint)

    parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
        args.downstream_endpoint
    )
    pd_worker_client = await runtime.namespace(parsed_namespace).component(parsed_component_name).endpoint(parsed_endpoint_name).client()

    handler = AsrEncodeWorker(args, config.engine_args, pd_worker_client)
    await handler.async_init(runtime)

    logger.info("Waiting for PD Worker Instances ...")
    await pd_worker_client.wait_for_instances()

    logger.info(f"Starting to serve the {args.endpoint} endpoint...")

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate, metrics_labels=[("model", config.model)]
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
