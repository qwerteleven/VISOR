
import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import tensorrt as trt
import sys
import logging

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from utils.io import get_config, set_logger
config = get_config("config.json", "export_tensorRT") 
set_logger("../logs", os.path.basename(sys.argv[0]))


logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")
model_path = config["model_path"] 



class MyLogger(trt.ILogger):
    def __init__(self):
       trt.ILogger.__init__(self)

    def log(self, severity, msg):
        logging.info(msg)
        print(severity, msg)

logger = MyLogger()

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
config_build.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GiB

if builder.platform_has_fast_fp16:
    config_build.set_flag(trt.BuilderFlag.FP16)


try:
    config_build.builder_optimization_level = 5      # 0–5; 5 = longest build, fastest engine
except AttributeError:
    msg = "older TRT version"
    logging.error(msg)
    print(msg)

serialized_engine = builder.build_serialized_network(network, config_build)


try: 
    with open(config["output_name"], "wb") as f:
        f.write(serialized_engine)
except:
    msg = "can not write the engine, check disk space, and logs"
    logging.error(msg)
    print(msg)

