"""
KV-cache manager for Gemma4 TensorRT inference.

Wraps the 24 owning layers' key/value tensors in a dict keyed by layer_idx,
so harness code can reason about "layer 22's key tensor" instead of
"flat_cache[44]". Provides translation to/from the flat positional order
that both the torch.export wrapper and the TensorRT engine's I/O actually use
(ONNX/TensorRT don't have a native concept of "named dict of tensors" --
everything is positional/name-string at that layer).

Usage:
    cache = KVCacheManager(owning_indices, layer_types, config, max_cache_len,
                            batch_size, device, dtype)
    cache.reset()  # zero-initialize for a fresh prefill

    # ... run engine with cache.to_flat_inputs() bound as the cache input tensors ...

    cache.update_from_flat_outputs(flat_outputs)  # after each inference call
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class LayerCacheSpec:
    layer_idx: int
    layer_type: str            # "sliding_attention" | "full_attention"
    seq_dim: int               # 512 (sliding_window) or max_cache_len (full)
    head_dim: int              # 256 (sliding) or global_head_dim (full)
    num_kv_heads: int


class KVCacheManager:
    def __init__(self, owning_indices, layer_types, config, max_cache_len,
                 batch_size, dtype, np_dtype=None):
        """
        owning_indices: list[int]  -- the 24 real layer indices, in wrapper order
        layer_types:    list[str] -- matching layer_type per owning_indices entry
        config:         text_config object (needs sliding_window, head_dim,
                        global_head_dim, num_key_value_heads)

        dtype:           numpy dtype to allocate buffers as (e.g. ml_dtypes.bfloat16)
        """
        assert len(owning_indices) == len(layer_types)
        self.owning_indices = list(owning_indices)
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        self.dtype = np_dtype if np_dtype is not None else dtype

        self.specs: dict[int, LayerCacheSpec] = {}
        for layer_idx, layer_type in zip(owning_indices, layer_types):
            if layer_type == "sliding_attention":
                seq_dim = config["sliding_window"]
                head_dim = config["head_dim"]
            else:
                seq_dim = max_cache_len
                head_dim = config["global_head_dim"]
            self.specs[layer_idx] = LayerCacheSpec(
                layer_idx=layer_idx,
                layer_type=layer_type,
                seq_dim=seq_dim,
                head_dim=head_dim,
                num_kv_heads=config["num_key_value_heads"],
            )


        self.tensors: dict[int, dict[str, np.ndarray]] = {}
        self.reset()

    def reset(self):
        """Zero-initialize all cache tensors -- call before a fresh prefill
        (new image/conversation, not continuing an existing one)."""
        for layer_idx, spec in self.specs.items():
            shape = (self.batch_size, spec.num_kv_heads, spec.seq_dim, spec.head_dim)
            self.tensors[layer_idx] = {
                "key": np.zeros(shape, dtype=self.dtype),
                "value": np.zeros(shape, dtype=self.dtype),
            }

    def to_flat_inputs(self):
        """Return the 48 cache tensors in flat (key0, value0, key1, value1, ...)
        order matching wrapper.owning_indices / the exported model's input order.
        This is the order torch.export's *flat_cache_in collected them in."""
        flat = []
        for layer_idx in self.owning_indices:
            flat.append(self.tensors[layer_idx]["key"])
            flat.append(self.tensors[layer_idx]["value"])
        return flat

    def update_from_flat_outputs(self, flat_outputs):
        """Given the 48 cache output tensors (same flat order as to_flat_inputs,
        per the wrapper's flat_cache_out construction), update internal storage."""
        assert len(flat_outputs) == 2 * len(self.owning_indices), (
            f"expected {2 * len(self.owning_indices)} cache outputs, got {len(flat_outputs)}"
        )
        for pos, layer_idx in enumerate(self.owning_indices):
            self.tensors[layer_idx]["key"] = flat_outputs[2 * pos]
            self.tensors[layer_idx]["value"] = flat_outputs[2 * pos + 1]

    def get(self, layer_idx: int, kind: str):
        """kind: 'key' or 'value'. Direct dict-style access for inspection/debugging."""
        return self.tensors[layer_idx][kind]

    # --- name mapping for TensorRT binding ---
    def flat_input_names(self, name_template="past_key_values.{idx}.{kind}"):
        """Names matching what the ONNX export should produce for cache inputs,
        in flat order. Adjust name_template if your actual export uses a
        different naming convention -- check with `engine.get_tensor_name(i)`
        after building and update this to match exactly."""
        names = []
        for layer_idx in self.owning_indices:
            names.append(name_template.format(idx=layer_idx, kind="key"))
            names.append(name_template.format(idx=layer_idx, kind="value"))
        return names

    def flat_output_names(self, name_template="present_key_values.{idx}.{kind}"):
        names = []
        for layer_idx in self.owning_indices:
            names.append(name_template.format(idx=layer_idx, kind="key"))
            names.append(name_template.format(idx=layer_idx, kind="value"))
        return names