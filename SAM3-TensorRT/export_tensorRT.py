
import json

with open("config.json") as f:
    config = json.load(f)

config = config["export_tensorRT"]


import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import tensorrt as trt


logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")
model_path = config["model_path"] 


class MyLogger(trt.ILogger):
    def __init__(self):
       trt.ILogger.__init__(self)

    def log(self, severity, msg):
        print(severity, msg)
        pass # Your custom logging implementation here

logger = MyLogger()

builder = trt.Builder(logger)

network = builder.create_network()

parser = trt.OnnxParser(network, logger)

success = parser.parse_from_file(model_path)

for idx in range(parser.num_errors):
    print(parser.get_error(idx))

if not success:
    print(parser.get_error(idx))


config_build = builder.create_builder_config()
config_build.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GiB

if builder.platform_has_fast_fp16:
    config_build.set_flag(trt.BuilderFlag.FP16)


try:
    config_build.builder_optimization_level = 5      # 0–5; 5 = longest build, fastest engine
except AttributeError:
    pass  # older TRT version


serialized_engine = builder.build_serialized_network(network, config_build)


with open(config["output_name"], "wb") as f:
    f.write(serialized_engine)



