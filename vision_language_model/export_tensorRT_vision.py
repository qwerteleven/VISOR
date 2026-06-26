
import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import tensorrt as trt
import sys
import logging
import ml_dtypes    
from transformers import AutoModelForCausalLM
import torch


root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from utils.io import get_config, set_logger
from utils.trt_logger import trt_logger 
from utils.kv_cache_manager import KVCacheManager
from export_onnx_vision import Gemma4_vision_wrapper
config = get_config("config.json", "export_vision_tensorRT") 
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
    config_build.builder_optimization_level = 5 
except AttributeError:
    msg = "older TRT version"
    logging.error(msg)
    print(msg)


profile = builder.create_optimization_profile()



profile.set_shape("input_ids", min=(1, 1), opt=(1, 296), max=(1, 512))
profile.set_shape("attention_mask", min=(1, 1), opt=(1, 296), max=(1, 512))
profile.set_shape("mm_token_type_ids", min=(1, 1), opt=(1, 296), max=(1, 512))
profile.set_shape("position_ids", min=(1, 1), opt=(1, 296), max=(1, 512))
profile.set_shape("pixel_values", min=(1, 2520, 768), opt=(1, 2520, 768), max=(1, 2520, 768))
profile.set_shape("image_position_ids", min=(1, 2520, 2), opt=(1, 2520, 2), max=(1, 2520, 2))


config_build.add_optimization_profile(profile)


serialized_engine = builder.build_serialized_network(network, config_build)


try: 
    with open(config["output_name"], "wb") as f:
        f.write(serialized_engine)
except FileNotFoundError:
    msg = "can not write the engine, check disk space, and logs"
    logging.error(msg)
    print(msg)

