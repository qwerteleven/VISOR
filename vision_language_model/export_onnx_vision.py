"""
Prefill export wrapper: full multimodal forward pass (pixel_values -> vision
tower -> fused embeddings -> text decoder), starting from an EMPTY cache,
producing both the first generated token's logits AND the populated cache
state to feed into the decode engine for subsequent steps.

This wrapper has no cache *inputs* -- StaticCache is constructed fresh (zeros)
inside forward(), since
prefill by definition starts a new conversation/image.

Two-engine design:
  - THIS wrapper -> prefill engine: pixel_values + prompt -> logits + cache
  -                       -> decode engine: token + cache -> logits + cache
Both must produce cache tensors in IDENTICAL flat order (owning_indices order)
so the decode engine can consume what this one produces without remapping.
"""

import torch
import os
import sys
from transformers import AutoProcessor
from transformers import AutoModelForMultimodalLM
from typing import Tuple, List
from torch.export import Dim

root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)

from _layer_inspection import (  # noqa: E402
    get_owning_layer_indices,
    get_layer_types,
    patch_clamp_limit,
    patch_reduce,
    patch_split_sequence,
)
from utils.io import get_config, set_logger, save_onnx, check_onnx  # noqa: E402


config = get_config("config.json", "onnx_export_vision")
set_logger("../logs", os.path.basename(sys.argv[0]))


class Gemma4_vision_wrapper(torch.nn.Module):
    """

    Forward signature:
        input_ids:          (batch, seq_len)              int64
        attention_mask:     (batch, seq_len)              int64
        mm_token_type_ids:  (batch, seq_len)              int64
        pixel_values:       (batch, num_images, C, H, W)  float/bfloat16
        image_position_ids: (batch, num_image_tokens)     int64
        position_ids:       (batch, seq_len)              int64

    Returns:
        logits:            (batch, seq_len, vocab_size)
        *flat_cache_out:    2 tensors per owning layer (key, value)
                            flat order as the decode wrapper expects as input
    """

    def __init__(
        self, model: AutoModelForMultimodalLM, max_batch_size: int, max_cache_len: int
    ):
        """

            Creates the object wrapper

        Args:
            model (AutoModelForMultimodalLM): internal vision model
            max_batch_size (int): max batch size
            max_cache_len (int): max cache len
        """

        super().__init__()

        assert model is not None
        assert max_batch_size > 0
        assert max_cache_len > 0

        self.model = model
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len

        self.owning_indices = get_owning_layer_indices(model)
        self.layer_types = get_layer_types(model, self.owning_indices)
        assert len(self.owning_indices) == 24

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, List]:
        """

            forward of model, to trace computation flow

        Args:
            input_ids (torch.Tensor): inputs layers identifiers
            attention_mask (torch.Tensor): sequence filter attention
            mm_token_type_ids (torch.Tensor): token type
            pixel_values (torch.Tensor): preprocess image
            image_position_ids (torch.Tensor): position to attention on image
            position_ids (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, List]: logits, cache tensors
        """

        """
        past_key_values = StaticCache(
            config=self.model.config.text_config,
            max_batch_size=self.max_batch_size,
            max_cache_len=self.max_cache_len,
            device=input_ids.device,
            dtype=self.model.dtype,
        )
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            position_ids=position_ids,
            # past_key_values=past_key_values,
            use_cache=False,  # use_cache=True,
        )

        logits = outputs.logits

        """
        softcap = self.model.config.text_config.final_logit_softcapping

        if softcap is not None:
            logits = logits / softcap
            logits = torch.tanh(logits)
            logits = logits * softcap
        """
        """
        flat_cache_out = []

        for layer_idx in self.owning_indices:
            layer = past_key_values.layers[layer_idx]
            flat_cache_out.append(layer.keys)
            flat_cache_out.append(layer.values)
        """
        return (logits, 1)  # *flat_cache_out)


def build_dummy_inputs(
    processor: AutoProcessor, sample_messages: List, device: torch.device
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """

        output as dummy/trace input

    Args:
        processor (AutoProcessor): preprocessor for Gemma
        sample_messages (List): representative message
        device (torch.device): computation device

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            needed inputs for inference
    """

    tokenizer_output = processor.apply_chat_template(
        sample_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    tokenizer_output = {k: v.to(device) for k, v in tokenizer_output.items()}

    input_ids = tokenizer_output["input_ids"]
    attention_mask = tokenizer_output["attention_mask"]
    mm_token_type_ids = tokenizer_output["mm_token_type_ids"]
    pixel_values = tokenizer_output["pixel_values"]
    image_position_ids = tokenizer_output["image_position_ids"]

    seq_len = input_ids.shape[-1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    return (
        input_ids,
        attention_mask,
        mm_token_type_ids,
        pixel_values,
        image_position_ids,
        position_ids,
    )


if __name__ == "__main__":
    model = AutoModelForMultimodalLM.from_pretrained(
        config["model_id"], dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(config["model_id"])
    max_batch_size = config["MAX_BATCH_SIZE"]
    sample_messages = config["message"]
    max_cache_len = config["PREFILL_LEN"]

    model = patch_clamp_limit(model)

    wrapper = Gemma4_vision_wrapper(model, max_batch_size, max_cache_len).eval()

    (
        input_ids,
        attention_mask,
        mm_token_type_ids,
        pixel_values,
        image_position_ids,
        position_ids,
    ) = build_dummy_inputs(processor, sample_messages, model.device)

    with torch.no_grad():
        eager_out = wrapper(
            input_ids,
            attention_mask,
            mm_token_type_ids,
            pixel_values,
            image_position_ids,
            position_ids,
        )

    print("try to export a minimal model")
    try:
        exported = torch.export.export(
            wrapper,
            (
                input_ids,
                attention_mask,
                mm_token_type_ids,
                pixel_values,
                image_position_ids,
                position_ids,
            ),
            strict=False,
        )
        print("EXPORT SUCCEEDED")
    except Exception as e:
        print("EXPORT FAILED:", type(e), e)
        raise

    print("checking numerical divergence")

    exported_out = exported.module()(
        input_ids,
        attention_mask,
        mm_token_type_ids,
        pixel_values,
        image_position_ids,
        position_ids,
    )

    with torch.no_grad():
        eager_out = wrapper(
            input_ids,
            attention_mask,
            mm_token_type_ids,
            pixel_values,
            image_position_ids,
            position_ids,
        )

    logits_diff = (eager_out[0].float() - exported_out[0].float()).abs().max()
    print("max logits diff:", logits_diff.item())

    """
    for i in [0, 1, 46, 47]:
        diff = (eager_out[1+i].float() - exported_out[1+i].float()).abs().max()
        print(f"cache tensor {i} max diff:", diff.item())

    """
    print("Try complete export")

    seq_len_dim = Dim("seq_len", min=37, max=config["PREFILL_LEN"])

    dynamic_shapes = {
        "input_ids": {1: seq_len_dim},
        "attention_mask": {1: seq_len_dim},
        "mm_token_type_ids": {1: seq_len_dim},
        "pixel_values": {},
        "image_position_ids": {},
        "position_ids": {1: seq_len_dim},
    }
    print("input_ids", input_ids.shape)

    try:
        exported = torch.onnx.export(
            wrapper,
            (
                input_ids,
                attention_mask,
                mm_token_type_ids,
                pixel_values,
                image_position_ids,
                position_ids,
            ),
            config["output_path"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            external_data=True,
            optimize=True,
        )
        print("EXPORT SUCCEEDED")
    except Exception as e:
        print("EXPORT FAILED:", type(e), e)
        raise

    check_onnx(config["output_path"])

    print("Patch reduce")
    output_reduce = f"{config['output_path'].replace('.onnx', '')}_reduce.onnx"
    m = patch_reduce(config["output_path"])
    save_onnx(output_reduce, m)
    check_onnx(output_reduce)

    print("Patch split sequence")
    output_reduce_split = f"{output_reduce.replace('.onnx', '')}_split.onnx"
    m = patch_split_sequence(output_reduce)
    save_onnx(output_reduce_split, m)
    check_onnx(output_reduce_split)
