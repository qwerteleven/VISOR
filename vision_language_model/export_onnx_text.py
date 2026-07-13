"""
Export wrapper for Gemma4ForConditionalGeneration (text path) with StaticCache,
exposing per-layer K/V cache tensors as flat, explicit inputs/outputs so
torch.export -> ONNX -> TensorRT gets a real KV-cache I/O contract.

Key architectural facts:

- 42 total decoder layers (text_config.num_hidden_layers), but only 24 layers
  actually own/store cache (`is_kv_shared_layer=False`). The other 18 layers
  reuse one of two "donor" layers' K/V via `shared_kv_states`, keyed by
  layer_type ("sliding_attention" / "full_attention"), not by layer index.

- `shared_kv_states` is fully call-local: it's built fresh from `past_key_values`
  on every forward() call (see Gemma4TextAttention.forward), so it does NOT
  need to be threaded through as extra export I/O -- only `past_key_values`
  (the 24 owning layers' cache) needs to cross the export boundary.

- Layer types have different shapes:
    sliding_attention layers: cache shape (batch, 2, sliding_window=512, head_dim=256)
    full_attention layers:    cache shape (batch, 2, max_cache_len,      global_head_dim=512)

- num_key_value_heads = 2 for all layers (GQA).

IMPORTANT: the exact list of which of the 42 layer indices are "owning" (cache-bearing)
vs "shared" must be read from the live model (see OWNING_LAYER_INDICES derivation below)
-- do not hardcode without verifying against your actual loaded model, since this is
read from `layer.self_attn.is_kv_shared_layer` per layer.

"""

import torch
import sys
import os
import logging
from transformers import AutoModelForCausalLM
from transformers.cache_utils import StaticCache
from typing import List, Tuple
from torch.export import Dim

root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)

from _layer_inspection import get_owning_layer_indices, get_layer_types  # noqa: E402
from utils.io import get_config, set_logger, check_onnx  # noqa: E402

config = get_config("config.json", "onnx_export_text")
set_logger("../logs", os.path.basename(sys.argv[0]))


class Gemma4_text_wrapper(torch.nn.Module):
    """

    Wraps Gemma4ForConditionalGeneration's language model for cache-aware export.

    Forward signature (all positional, all tensors -- export-friendly):
        input_ids:         (batch, seq_len)             int64
        attention_mask:    (batch, total_seq_len)       int64   -- covers cache + new tokens
        cache_position:    (seq_len,)                   int64   -- absolute positions of input_ids
        *flat_cache_in:    2 tensors per owning layer (key, value), in owning-layer order

    Returns:
        logits:            (batch, seq_len, vocab_size)
        *flat_cache_out:   2 tensors per owning layer (updated key, value), same order

    """

    def __init__(
        self, model: AutoModelForCausalLM, max_batch_size: int, max_cache_len: int
    ):
        """

            Creates the object wrapper

        Args:
            model (AutoModelForCausalLM): internal text model
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

        assert len(self.owning_indices) == 24, (
            f"expected 24 owning layers, got {len(self.owning_indices)} -- "
            f"architecture assumption violated"
        )

    def _build_cache_from_flat(
        self, flat_cache_tensors: List, device: torch.device, dtype: torch.dtype
    ) -> StaticCache:
        """

            Construct a StaticCache and populate its owning layers' key/value
            tensors directly from the flat input list (no copy needed beyond the
            assignment -- these become the live tensors the model reads/writes).

        Args:
            flat_cache_tensors (List): list of tensors to make cache
            device (torch.device): device destination to compute
            dtype (AutoModelForCausalLM.dtype): numeric precision of the model

        Returns:
            StaticCache: defined cache space
        """

        cache = StaticCache(
            config=self.model.config.text_config
            if hasattr(self.model.config, "text_config")
            else self.model.config,
            max_batch_size=self.max_batch_size,
            max_cache_len=self.max_cache_len,
            device=device,
            dtype=dtype,
        )

        assert len(flat_cache_tensors) == 2 * len(self.owning_indices), (
            f"expected {2 * len(self.owning_indices)} flat cache tensors, "
            f"got {len(flat_cache_tensors)}"
        )

        for pos, layer_idx in enumerate(self.owning_indices):
            key_in = flat_cache_tensors[2 * pos]
            value_in = flat_cache_tensors[2 * pos + 1]
            cache.layers[layer_idx].keys = key_in
            cache.layers[layer_idx].values = value_in

        return cache

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        *flat_cache_in: List,
    ) -> Tuple[torch.Tensor, List]:
        """

            forward of model, to trace computation flow

        Args:
            input_ids (torch.Tensor): inputs layers identifiers
            attention_mask (torch.Tensor): sequence filter attention
            cache_position (torch.Tensor): position of tensor to generate a cache
            *flat_cache_in (List): agrupation of cache tensors IO
        Returns:
            output (Tuple[torch.Tensor, List]): logits, cache tensors
        """

        assert len(flat_cache_in) > 0

        past_key_values = self._build_cache_from_flat(
            flat_cache_in, device=input_ids.device, dtype=self.model.dtype
        )

        outputs = self.model.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.model.lm_head(hidden_states)

        try:
            softcap = self.model.config.text_config.final_logit_softcapping
        except Exception as e:
            msg = f"can not access to language config text_config final_logit_softcapping, error: {e}"
            logging.error(msg)
            print(msg)
            softcap = None

        if softcap is not None:
            logits = logits / softcap
            logits = torch.tanh(logits)
            logits = logits * softcap

        flat_cache_out = []

        for layer_idx in self.owning_indices:
            layer = past_key_values.layers[layer_idx]
            flat_cache_out.append(layer.keys)
            flat_cache_out.append(layer.values)

        output = (logits, *flat_cache_out)

        return output


def build_dummy_inputs(
    model: AutoModelForCausalLM,
    wrapper: Gemma4_text_wrapper,
    batch_size: int,
    prefill_len: int,
    device: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List]:
    """

        Construct dummy inputs for a prefill-shaped export trace.
        Cache tensors start as zeros (empty cache) -- correct for prefill.

    Args:
        model (AutoModelForCausalLM): target model
        wrapper (Gemma4_text_wrapper): wrapper to make traceable computation flow
        batch_size (int): number of inputs, normally 1, text generation
        prefill_len (int): prompt text lenght
        device (torch.dtype): computation device target

    Returns:
        output (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List]): needed parameters for wrapper forward
    """

    assert batch_size > 0
    assert prefill_len > 0

    try:
        config = model.config.text_config
    except Exception as e:
        msg = f"can not access to language config text_config, error: {e}"
        logging.error(msg)
        print(msg)

    assert config.vocab_size > 0

    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, prefill_len), device=device
    )
    attention_mask = torch.ones(
        (batch_size, prefill_len), dtype=torch.int64, device=device
    )
    cache_position = torch.arange(prefill_len, device=device)

    flat_cache_in = []

    try:
        num_kv_heads = config.num_key_value_heads
    except Exception as e:
        msg = f"can not access to language config num_key_value_heads, error: {e}"
        logging.error(msg)
        print(msg)

    assert num_kv_heads > 0

    for layer_idx, layer_type in zip(wrapper.owning_indices, wrapper.layer_types):
        if layer_type == "sliding_attention":
            seq_dim = config.sliding_window
            head_dim = config.head_dim
        else:
            seq_dim = wrapper.max_cache_len
            head_dim = config.global_head_dim

        k = torch.zeros(
            (batch_size, num_kv_heads, seq_dim, head_dim),
            dtype=model.dtype,
            device=device,
        )
        v = torch.zeros(
            (batch_size, num_kv_heads, seq_dim, head_dim),
            dtype=model.dtype,
            device=device,
        )
        flat_cache_in.append(k)
        flat_cache_in.append(v)

    output = (input_ids, attention_mask, cache_position, flat_cache_in)

    return output


if __name__ == "__main__":
    model_id: str = config["model_id"]
    assert len(model_id) > 0

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval()

    MAX_BATCH_SIZE: int = config["MAX_BATCH_SIZE"]
    PREFILL_LEN: int = config["PREFILL_LEN"]
    PREFILL_LEN_TRACE: int = config["PREFILL_LEN_TRACE"]

    assert MAX_BATCH_SIZE > 0
    assert PREFILL_LEN > 2
    assert 2 < PREFILL_LEN_TRACE < PREFILL_LEN

    wrapper = Gemma4_text_wrapper(model, MAX_BATCH_SIZE, PREFILL_LEN).eval()

    input_ids, attention_mask, cache_position, flat_cache_in = build_dummy_inputs(
        model, wrapper, MAX_BATCH_SIZE, PREFILL_LEN_TRACE, model.device
    )

    print(
        f"owning layer indices ({len(wrapper.owning_indices)}): {wrapper.owning_indices}"
    )
    print(f"layer types per owning layer: {wrapper.layer_types}")
    print(f"flat_cache_in count: {len(flat_cache_in)} tensors")

    with torch.no_grad():
        eager_out = wrapper(input_ids, attention_mask, cache_position, *flat_cache_in)
    print("eager forward OK, logits shape:", eager_out[0].shape)
    print("eager forward OK, num cache outputs:", len(eager_out) - 1)

    print("try to export a minimal model")
    try:
        exported = torch.export.export(
            wrapper,
            (input_ids, attention_mask, cache_position, *flat_cache_in),
            strict=False,
        )
        print("EXPORT SUCCEEDED")
        print("num user inputs:", len(exported.graph_signature.user_inputs))
        print("num user outputs:", len(exported.graph_signature.user_outputs))
    except Exception as e:
        print("EXPORT FAILED:", type(e), e)
        raise

    print("checking numerical divergence")

    with torch.no_grad():
        eager_out = wrapper(input_ids, attention_mask, cache_position, *flat_cache_in)

    exported_out = exported.module()(
        input_ids, attention_mask, cache_position, *flat_cache_in
    )

    logits_diff = (eager_out[0].float() - exported_out[0].float()).abs().max()
    print("max logits diff:", logits_diff.item())

    for i in [0, 1, 46, 47]:  # first layer's k/v, last layer's k/v
        diff = (eager_out[1 + i].float() - exported_out[1 + i].float()).abs().max()
        print(f"cache tensor {i} max diff:", diff.item())

    print("Try complete export")

    seq_len_dim = Dim("seq_len", min=2, max=PREFILL_LEN)
    attn_len_dim = Dim("attn_len", min=2, max=PREFILL_LEN)

    dynamic_shapes = {
        "input_ids": {1: seq_len_dim},
        "attention_mask": {1: attn_len_dim},
        "cache_position": {0: seq_len_dim},
        "flat_cache_in": tuple({} for _ in flat_cache_in),
    }

    try:
        exported = torch.onnx.export(
            wrapper,
            (input_ids, attention_mask, cache_position, *flat_cache_in),
            config["output_path"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            external_data=True,
        )
        print("EXPORT SUCCEEDED")
    except Exception as e:
        print("EXPORT FAILED:", type(e), e)
        raise

    check_onnx(config["output_path"])
