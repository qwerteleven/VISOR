
import logging
import tensorrt as trt


class trt_logger(trt.ILogger):
    def __init__(self):
       trt.ILogger.__init__(self)

    def log(self, severity, msg):
        logging.info(msg)
        print(severity, msg)