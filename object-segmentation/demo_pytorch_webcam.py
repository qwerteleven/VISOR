import numpy as np
import cv2
import torch
import os
import sys
from typing import Dict, List, Tuple
from PIL import Image

from transformers import Sam3Processor, Sam3Model

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from utils.io import get_config, set_logger
config = get_config("config.json", "demo_pytorch_webcam") 
set_logger("../logs", os.path.basename(sys.argv[0]))

from utils.image_processing import overlay_masks, draw_user_rectangle
from utils.wrappers import timer
from utils.streaming import streaming_pipeline_OpenCV


class sam3_model():
    """

        SAM3 model, implementation on pytorch

    """    

    def __init__(self: object, config: Dict) -> None:
        """

            Create a object for handle SAM model
        
        Args:
            self (object): self
            config (Dict): config of the model

        """        

        # attention image region
        self.user_ref_point = []
        self.config = config
        self.input_boxes_labels = [[1]] 
        self.input_boxes = [[[]]]

    def load(self: object) -> None:
        """
        load the engine on the device 

        Args:
            self (object): self

        """        
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")


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
            image = draw_user_rectangle(image, self.user_ref_point)

        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)


        tokenizer_output = self.processor(
            images=image,
            text=self.config["prompt"], 
            input_boxes=self.input_boxes,
            input_boxes_labels=self.input_boxes_labels,
            return_tensors="pt"
        ).to(self.device)


        with torch.no_grad():
            outputs = self.model(**tokenizer_output)


        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.config["score_thr"],
            mask_threshold=self.config["mask_threshold"],
            target_sizes=tokenizer_output.get("original_sizes").tolist()
        )[0]


        output_image = overlay_masks(image, results["masks"])

        output_image = np.array(output_image)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

        return output_image


if __name__ == "__main__":
    
    ml_model = sam3_model(config)
    ml_model.load()
    streaming_pipeline_OpenCV(config, ml_model)



