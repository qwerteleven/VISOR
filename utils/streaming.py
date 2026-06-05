import cv2
from typing import Dict
import logging

from utils.user_event import click_and_crop
import utils.globals as globals

def streaming_pipeline_OpenCV(config: Dict, ml_model: object, cam: object = None) -> None:
    """
    
        Create a streaming processing with OpenCV cam handles

    Args:
        config (Dict): config of the windows streaming
        ml_model (object): process frame of streaming
        cam (object, optional): camera device. Defaults to None.

    Raises:
        Exception: check if can open webcam device
        Exception: check if can take a frame from device
        Exception: check if can make inference on model

    """    

    globals.init()
    globals.VISOR_NAME = config["VISOR_NAME"]
    if cam is None:

        try:
            cam = cv2.VideoCapture(0)
        except:
            msg("Can not open webcam")
            logging.error(msg)
            print(msg)
            raise Exception(msg)

    
    cv2.namedWindow(config["VISOR_NAME"])
    cv2.setMouseCallback(config["VISOR_NAME"], click_and_crop)

    for _ in range(config["max_iteration"]):
        ret, image = cam.read()


        if not ret:
            msg = "failed to grab frame"
            logging.error(msg)
            print(msg)
            raise Exception(msg)
        
        try:
            ml_model.update_attention_region(globals.user_ref_point, image.shape)
            output_image = ml_model(image)
        except:
            msg = "can not make inference with ml_model"
            logging.error(msg)
            print(msg)
            raise Exception(msg)

        cv2.imshow(config["VISOR_NAME"], output_image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    cam.release()
