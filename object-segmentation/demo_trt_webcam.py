import numpy as np
import cv2
import torch
import os
import sys
import logging
from typing import Dict, List, Tuple
from PIL import Image
import tensorrt as trt

# loads all the plugins
logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")

from transformers import Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3ImageSegmentationOutput


root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from utils.io import get_config, set_logger
config = get_config("config.json", "demo_trt_webcam") 
set_logger("../logs", os.path.basename(sys.argv[0]))

import utils.cuda_handler as cuda_handler
from utils.image_processing import overlay_masks, draw_user_rectangle
from utils.wrappers import timer
from utils.streaming import streaming_pipeline_OpenCV


class sam3_model():
    """

        SAM3 model, implementation on tensorRT

    """    

    def __init__(self: object, engine_path: str, config: Dict, overlay_config: Dict) -> None:
        """

            Create a object for handle SAM model
        
        Args:
            self (object): self
            engine_path (str): path to the engine file
            config (Dict): config of the model

        Raises:
            Exception: check if the model file exists
        """        
        if not os.path.isfile(engine_path):
            msg = f"engine file not exists: {engine_path}"
            logging.error(msg)
            print(msg)
            raise Exception("msg")
    
        self.engine_path = engine_path
        self.context = None
        self.engine = None
        self.inputs = None
        self.outputs = None 
        self.bindings = None 
        self.stream = None
        self.processor = None
        
        # attention image region
        self.user_ref_point = []
        self.config = config
        self.overlay_config = overlay_config
        self.input_boxes_labels = [[1]] 
        self.input_boxes = [[[]]]

    def load(self: object) -> None:
        """
        load the engine on the device 

        Args:
            self (object): self

        Raises:
            Exception: check if the model can be loaded on the device
        """        
        
        try:
            with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        except FileNotFoundError:
            msg = "can not load engine on the device"
            logging.error(msg)
            print(msg)
            raise Exception(msg)

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = cuda_handler.allocate_buffers(self.engine)

        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    def _copy_to_device(self: object, tokenizer_output: Dict) -> None:
        """
            copy on device inputs for the engine model

        Args:
            self (object): self
            tokenizer_output (Dict): output of the preprocessing, type hugging face
        """        
        np.copyto(self.inputs[0].host, tokenizer_output["pixel_values"].ravel())
        np.copyto(self.inputs[1].host, tokenizer_output["input_ids"].ravel())
        np.copyto(self.inputs[2].host, tokenizer_output["attention_mask"].ravel())
        np.copyto(self.inputs[3].host, tokenizer_output["input_boxes"].ravel())
        np.copyto(self.inputs[4].host, tokenizer_output["input_boxes_labels"].ravel())

    def update_attention_region(self: object, user_ref_point: List, image_shape: Tuple) -> None:
        """
        
            update the region of attention for the model

        Args:
            user_ref_point (List): list of points touched by the user
        """        
        self.user_ref_point = user_ref_point

        if len(self.user_ref_point) == 2:
            self.box1_xyxy = []
            for point in self.user_ref_point:
                self.box1_xyxy += [int(point[0] * image_shape[1])]
                self.box1_xyxy += [int(point[1] * image_shape[0])]

            self.input_boxes = [[self.box1_xyxy]]

    @timer
    def __call__(self, image: np.array, **kwds) -> np.array:
        """
        
            Make inference over a numpy image

        Args:
            image (np.array): raw image for the inference

        Returns:
            np.array: image overlay with the inference result
        """       

        if len(self.user_ref_point) == 2:
            image = draw_user_rectangle(image, self.user_ref_point, self.overlay_config)

        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)


        tokenizer_output = self.processor(
            images=image, 
            text=self.config["prompt"], 
            input_boxes=self.input_boxes,
            input_boxes_labels=self.input_boxes_labels,
            return_tensors="pt"
        )

        self._copy_to_device(tokenizer_output)

        output_model = cuda_handler.do_inference(
            self.context, 
            engine=self.engine, 
            bindings=self.bindings, 
            inputs=self.inputs, 
            outputs=self.outputs, 
            stream=self.stream
        )

        output_model = [
            out.reshape(*outputs_shapes)
            for out, outputs_shapes in zip(output_model, self.config["outputs_shapes"])
        ]

        sam_output = Sam3ImageSegmentationOutput()

        sam_output.pred_masks =              torch.from_numpy(output_model[0])
        sam_output.pred_boxes =              torch.from_numpy(output_model[1])
        sam_output.pred_logits =             torch.from_numpy(output_model[2])
        sam_output.presence_logits =         torch.from_numpy(output_model[3])
        sam_output.semantic_seg =            torch.from_numpy(output_model[4])
        sam_output.decoder_reference_boxes = torch.from_numpy(output_model[5])

        results = self.processor.post_process_instance_segmentation(
            sam_output,
            threshold=self.config["score_thr"],
            mask_threshold=self.config["mask_threshold"],
            target_sizes=tokenizer_output.get("original_sizes").tolist()
        )[0]


        output_image = overlay_masks(image, results["masks"])

        output_image = np.array(output_image)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

        return output_image


if __name__ == "__main__":
    
    overlay_config = get_config("../config.json", "streaming_overlay")
    ml_model = sam3_model(config["engine_file_path"], config, overlay_config)
    ml_model.load()
    streaming_pipeline_OpenCV(config, ml_model)

