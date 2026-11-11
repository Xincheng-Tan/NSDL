# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for tokenization."""
import warnings

__all__ = ['hf_tokenizer', 'hf_processor']


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f'tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f'tokenizer.pad_token is None. Now set to {tokenizer.eos_token}')


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer
    if correct_gemma2 and isinstance(name_or_path, str) and 'gemma-2-2b-it' in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn('Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.')
        kwargs['eos_token'] = '<end_of_turn>'
        kwargs['eos_token_id'] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor
    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception:
        processor = None
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    if processor is not None:
        # Input-resolution cap (A0b, 2026-08-27). This is the single funnel for
        # every RL-side processor -- actor (fsdp_workers:352), ref/rollout
        # (fsdp_workers:917), and RLHFDataset, which is handed the processor built
        # here. Note the usual case needs no env var at all: a ckpt trained under a
        # cap ships max_pixels in its own preprocessor_config.json, so both this
        # call and vLLM's internal processor inherit it. The env var exists to
        # force a cap onto an *uncapped* ckpt, which is what the chain-i ablation
        # did; setting it here keeps that override on the same code path as SFT
        # and eval instead of a fourth copy of "clamp max_pixels".
        from nsdl.image_pixel_cap import apply_image_pixel_cap

        apply_image_pixel_cap(processor, tag="verl")
    return processor
