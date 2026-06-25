
from transformers import AutoModelForCausalLM
from transformers import AutoModelForMultimodalLM  
from typing import List
import logging


def get_owning_layer_indices(model: AutoModelForCausalLM) -> List:
    """ 
    
        (is_kv_shared_layer == False), in order. Should be length 24 for Gemma4 text.

    Args:
        model (AutoModelForCausalLM): internal text model 

    Returns:
        List: the list of layer indices that actually own/store cache
    """    

    try:
        layers = model.model.language_model.layers
    except Exception as e:
        
        msg = f"can not access to language layers, error: {e}"
        logging.error(msg)
        print(msg)

    owning = [
        i 
        for i, 
        layer in enumerate(layers) 
        if not layer.self_attn.is_kv_shared_layer
    ]

    assert len(owning) > 0

    return owning


def get_layer_types(model: AutoModelForCausalLM, owning_indices: List) -> List:
    """
    
        layer_type string ('sliding_attention' / 'full_attention') per owning layer index.

    Args:
        model (AutoModelForCausalLM): internal text model 
        owning_indices (List): _description_

    Returns:
        List: model list types per layers
    """  

    try:
        layer_types = model.config.text_config.layer_types
    except Exception as e:
        msg = f"can not access to language text_config layer_types, error: {e}"
        logging.error(msg)
        print(msg)

    list_types = [layer_types[i] for i in owning_indices]

    assert len(list_types) > 0

    return list_types


def patch_clamp_limit(model: AutoModelForMultimodalLM) -> AutoModelForMultimodalLM:
    """
    
        creates static limits for be traced by ONNX

    Args:
        model (AutoModelForMultimodalLM): model to process

    Returns:
        AutoModelForMultimodalLM: modified model
    """    
    patched_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "input_min") and hasattr(module, "input_max"):
            
            input_min_val = float(module.input_min.item())
            input_max_val = float(module.input_max.item())
            output_min_val = float(module.output_min.item())
            output_max_val = float(module.output_max.item())

            del module._buffers["input_min"]
            del module._buffers["input_max"]
            del module._buffers["output_min"]
            del module._buffers["output_max"]

            module.input_min = input_min_val
            module.input_max = input_max_val
            module.output_min = output_min_val
            module.output_max = output_max_val

            patched_count += 1

    print(f"patched {patched_count} clipped-linear modules (vision + audio + any others)")

    return model

