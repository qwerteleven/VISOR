
import os
import tensorrt as trt
import numpy as np
import sys
import logging
import ml_dtypes    
from transformers import AutoModelForCausalLM
import torch

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)


from utils import func                            # noqa: E402
from utils import cuda_handler                    # noqa: E402
from utils.kv_cache_manager import KVCacheManager # noqa: E402
from export_onnx_text import Gemma4_text_wrapper  # noqa: E402
from utils.io import get_config, set_logger       # noqa: E402
from utils.trt_logger import trt_logger           # noqa: E402


config = get_config("config.json", "export_text_tensorRT") 
set_logger("../logs", os.path.basename(sys.argv[0]))


logger = trt_logger()
trt.init_libnvinfer_plugins(logger, "")
model_path = config["model_path"] 


builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

success = parser.parse_from_file(model_path)

for idx in range(parser.num_errors):
    print(parser.get_error(idx))
    logging.error(parser.get_error(idx))

if not success:
    print(parser.get_error(idx))
    logging.error(parser.get_error(idx))


config_build = builder.create_builder_config()
config_build.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, config["memory_size_gb"] << 30)

if builder.platform_has_fast_fp16:
    config_build.set_flag(trt.BuilderFlag.FP16)


try:
    config_build.builder_optimization_level = 5      # 0–5; 5 = longest build, fastest engine
except AttributeError:
    msg = "older TRT version"
    logging.error(msg)
    print(msg)

profile = builder.create_optimization_profile()

for key, value in config["dynamic_shape"].items():
    if (type(value["min"]) is list and
        type(value["opt"]) is list and
        type(value["max"]) is list):
        profile.set_shape(
            key, 
            min = value["min"],
            opt = value["opt"],  
            max = value["max"]
        )
    if (type(value["min"]) is int and
        type(value["opt"]) is int and
        type(value["max"]) is int):
        profile.set_shape(
            key, 
            min = (value["min"], ),
            opt = (value["opt"], ),  
            max = (value["max"], )
        )


config_kv_cache = get_config("config.json", "kv_cache") 

cache = KVCacheManager(
    owning_indices=config_kv_cache["owning_indices"],
    layer_types=config_kv_cache["layer_types"],
    config=config_kv_cache["config_model"],
    max_cache_len=config["MAX_CACHE_LEN"],
    batch_size=config["MAX_BATCH_SIZE"],
    dtype=None,
    np_dtype=ml_dtypes.bfloat16,
)


for layer_idx, spec in cache.specs.items():
    shape = (cache.batch_size, spec.num_kv_heads, spec.seq_dim, spec.head_dim)
    key_name   = f"past_key_values.{layer_idx}.key" 
    value_name = f"past_key_values.{layer_idx}.value"
    profile.set_shape(key_name,   min=shape, opt=shape, max=shape)
    profile.set_shape(value_name, min=shape, opt=shape, max=shape)

config_build.add_optimization_profile(profile)
serialized_engine = builder.build_serialized_network(network, config_build)


try: 
    with open(config["output_name"], "wb") as f:
        f.write(serialized_engine)
except FileNotFoundError:
    msg = "can not write the engine, check disk space, and logs"
    logging.error(msg)
    print(msg)
except Exception as e:
    msg = f"can not write the engine, error {e}"
    logging.error(msg)
    print(msg)





print("CHECK NUMERICAL DIFFERENCE")
cache.reset() 


model = AutoModelForCausalLM.from_pretrained(config["model_label"], dtype=torch.bfloat16).eval()
wrapper_decode = Gemma4_text_wrapper(model, 1, config["MAX_CACHE_LEN"])


input_ids_dec = torch.randint(0, config["vocab_size"], (1, config["MAX_CACHE_LEN"]), dtype=torch.int64)
attention_mask_dec = torch.ones((1, config["MAX_CACHE_LEN"]), dtype=torch.int64)
cache_position_dec = torch.arange(config["MAX_CACHE_LEN"], dtype=torch.int64)

flat_cache_in = [cuda_handler.to_host_bytes(t) for t in cache.to_flat_inputs()]

with torch.no_grad():
    eager_out = wrapper_decode(
        input_ids_dec, attention_mask_dec, cache_position_dec, *flat_cache_in
    )


eager_logits = eager_out[0].float().cpu().numpy()


with open(config["output_name"], "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())


context = engine.create_execution_context()


context.set_input_shape("input_ids",      (1, config["MAX_CACHE_LEN"]))
context.set_input_shape("attention_mask", (1, config["MAX_CACHE_LEN"]))
context.set_input_shape("cache_position", (config["MAX_CACHE_LEN"],))


inputs_trt, outputs_trt, bindings, stream = cuda_handler.allocate_buffers(
    engine, profile_idx=0, context=context
)


input_name_to_idx, output_name_to_idx = cuda_handler.get_input_output(engine)


for name, tensor in [
    ("input_ids",      input_ids_dec),
    ("attention_mask", attention_mask_dec),
    ("cache_position", cache_position_dec),
]:
    idx = input_name_to_idx[name]
    np.copyto(inputs_trt[idx].host, tensor.cpu().numpy().ravel())


flat_np = cache.to_flat_inputs()
for pos, cache_tensor in enumerate(flat_np):
    idx = input_name_to_idx[f"flat_cache_in_{pos}"]
    np.copyto(inputs_trt[idx].host, cache_tensor.view(np.uint16).ravel())


engine_out = cuda_handler.do_inference(
    context, engine=engine, bindings=bindings,
    inputs=inputs_trt, outputs=outputs_trt, stream=stream
)


logits_idx = output_name_to_idx["mul_13306"]
trt_logits = engine_out[logits_idx][:config["MAX_CACHE_LEN"] * config["vocab_size"]].reshape(
    1, config["MAX_CACHE_LEN"], config["vocab_size"]
).astype(np.float32)


diff = np.abs(trt_logits - eager_logits).max()
print(f"max logits diff (TRT decode engine vs eager): {diff:.6f}")

diff = np.abs(func.sigmoid(trt_logits) - func.sigmoid(eager_logits)).max()
print(f"max logits diff, after sigmoid (TRT decode engine vs eager): {diff:.6f}")

trt_argmax = trt_logits[0, -1, :].argmax()
eager_argmax = eager_logits[0, -1, :].argmax()

print(f"last-position argmax — TRT: {trt_argmax}, eager: {eager_argmax}, match: {trt_argmax == eager_argmax}")

