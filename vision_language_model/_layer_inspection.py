
from transformers import AutoModelForCausalLM
from transformers import AutoModelForMultimodalLM  
from typing import List, Dict
import logging
import os.path
import onnx
import numpy as np
from collections import defaultdict



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


def _get_const_value(const_node: Dict) -> None | np.array :
    """
    
        get the constant value over shapes and scalars

    Args:
        const_node (Dict): ONNX nodes

    Returns:
        None | np.array: if not found constant values | shape ->  None
    """    
    for a in const_node.attribute:

        if a.name == "value":
            return onnx.numpy_helper.to_array(a.t)
        
        if a.name == "value_ints":
            return np.array(list(a.ints), dtype=np.int64)
        
        if a.name == "value_int":
            return np.array([a.i], dtype=np.int64)
        
    return None


def _topo_sort(graph):
    """
    
        sort the topology of the ONNX graph, trace for aisolated nodes

    Args:
        graph (ONNX.graph): graph to sort

    Raises:
        RuntimeError: graph route have not retonable nodes

    Returns:
        ONNX.graph: sort graph
    """    
    available = set(i.name for i in graph.input) | set(i.name for i in graph.initializer)
    sorted_nodes, remaining = [], list(graph.node)

    while remaining:
        progressed = False
        still = []

        for n in remaining:
            if all(inp == "" or inp in available for inp in n.input):
                sorted_nodes.append(n)
                available.update(n.output)
                progressed = True
            else:
                still.append(n)

        if not progressed:
            raise RuntimeError(
                f"stuck: {
                    [
                    (n.name, [i for i in n.input if i not in available]) 
                    for n in still[:5]
                    ]
                }"
            )
        
        remaining = still

    del graph.node[:]
    graph.node.extend(sorted_nodes)

    return graph
  

def patch_reduce(onnx_path: str):
    """
        reduce sequence across ONNX graph

    Args:
        onnx_path (str): path to load model

    Returns:
        ONNX: modified graph
    """    


    m = onnx.load(onnx_path, load_external_data=True)
    graph = m.graph

    output_to_node = {}
    for n in graph.node:
        for o in n.output:
            output_to_node[o] = n

    consumer_count = defaultdict(int)
    for n in graph.node:
        for inp in n.input:
            if inp:
                consumer_count[inp] += 1

    to_delete = set()
    patched = 0

    for node in list(graph.node):
        if node.op_type != "Reshape":
            continue

        if node.input[0] not in output_to_node or node.input[1] not in output_to_node:
            continue

        data_producer = output_to_node[node.input[0]]
        shape_producer = output_to_node[node.input[1]]

        if data_producer.op_type != "Constant" or shape_producer.op_type != "Constant":
            continue

        scalar_val = _get_const_value(data_producer)
        new_shape = _get_const_value(shape_producer)

        if scalar_val is None or new_shape is None:
            continue

        if scalar_val.dtype not in (np.int32, np.int64):
            continue

        folded_array = scalar_val.astype(np.int64).reshape(new_shape)
        folded_name = f"{node.output[0]}_folded"

        new_const = onnx.helper.make_node(
            "Constant",
            inputs = [],
            outputs = [folded_name],
            name = f"{node.name}_folded_const",
            value = onnx.numpy_helper.from_array(folded_array, name=folded_name),
        )

        old_output = node.output[0]
        for consumer in graph.node:
            for i, inp in enumerate(consumer.input):
                if inp == old_output:
                    consumer.input[i] = folded_name

        graph.node.append(new_const)
        to_delete.add(node.name)  
        

        consumer_count[node.input[0]] -= 1
        consumer_count[node.input[1]] -= 1

        if consumer_count[node.input[0]] == 0:
            to_delete.add(data_producer.name)

        if consumer_count[node.input[1]] == 0:
            to_delete.add(shape_producer.name)

        patched += 1

    print(f"folded {patched} constant-reshape patterns")

    new_nodes = [n for n in graph.node if n.name not in to_delete]
    del graph.node[:]
    graph.node.extend(new_nodes)
        
    graph = _topo_sort(graph)

    return m


def patch_split_sequence(onnx_path: str):
    """
    
        split sequence across ONNX graph

    Args:
        onnx_path (str): path to load model

    Returns:
        ONNX: modified model
    """    

    m = onnx.load(onnx_path, load_external_data=True)
    graph = m.graph
    output_to_node = {}
    consumer_map = {}

    for n in graph.node:

        for o in n.output:
            output_to_node[o] = n

        for inp in n.input:
            consumer_map.setdefault(inp, []).append(n)

    sts_nodes = [n for n in graph.node if n.op_type == "SplitToSequence"]
    print(f"processing {len(sts_nodes)} SplitToSequence nodes")

    to_delete = set()
    new_nodes = []
    patched = 0

    for n in sts_nodes:
        data_input = n.input[0]
        split_sizes_node = output_to_node.get(n.input[1])
        sizes = _get_const_value(split_sizes_node)

        if sizes is None:
            print(f"SKIP {n.name}: couldn't resolve static split sizes")
            continue

        axis = -1
        for a in n.attribute:
            if a.name == "axis":
                axis = a.i

        consumers = consumer_map.get(n.output[0], [])
        seq_at_consumers = [c for c in consumers if c.op_type == "SequenceAt"]

        if len(seq_at_consumers) != len(sizes):
            print(f"SKIP {n.name}: {len(seq_at_consumers)} SequenceAt consumers != {len(sizes)} split sizes")
            continue

        idx_to_seqat = {}
        ok = True

        for c in seq_at_consumers:
            idx_node = output_to_node.get(c.input[1])
            idx_val = _get_const_value(idx_node)
            if idx_val is None:
                ok = False
                break
        
            idx_to_seqat[int(idx_val.reshape(-1)[0])] = c

        if not ok or set(idx_to_seqat.keys()) != set(range(len(sizes))):
            print(f"SKIP {n.name}: non-static or non-contiguous indices")
            continue

        split_outputs = [f"{n.name}_split_out_{i}" for i in range(len(sizes))]
        split_node = onnx.helper.make_node(
            "Split",
            inputs=[data_input, n.input[1]],
            outputs=split_outputs,
            name=n.name + "_as_split",
            axis=axis,
        )
        new_nodes.append(split_node)

        for i, seqat_node in idx_to_seqat.items():
            old_out = seqat_node.output[0]
            new_out = split_outputs[i]
            for other in graph.node:
                for j, inp in enumerate(other.input):
                    if inp == old_out:
                        other.input[j] = new_out
            to_delete.add(seqat_node.name)
        to_delete.add(n.name)
        patched += 1

    print(f"rewrote {patched} SplitToSequence groups into plain Split")

    remaining_nodes = [nd for nd in graph.node if nd.name not in to_delete]
    del graph.node[:]
    graph.node.extend(remaining_nodes)
    graph.node.extend(new_nodes)

    _topo_sort(graph)

    return m

