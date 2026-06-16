import cv2
from typing import Dict
import logging
import traceback

from vidgear.gears import CamGear
from vidgear.gears import WriteGear


from utils.io import get_config
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
        except SystemError:
            msg = "Can not open webcam"
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
        except Exception as e:
            msg = f"can not make inference with ml_model, error: {e}"
            logging.error(msg)
            print(msg)
            print(traceback.format_exc())
            raise Exception(msg)

        cv2.imshow(config["VISOR_NAME"], output_image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    cam.release()



def streaming_pipeline_vidgear(config: Dict, ml_model: object, input_source: str, input_config: Dict, output_source: str, output_config: Dict) -> None:
    """
    
        Takes a input of streaming, executes the ml model over the frames and stremaing the result to rtsp-server

    Args:
        config (Dict): config of the interface and user inputs
        ml_model (object): ml model tha inference overr the frames
        input_source (str): url to streaming, can be a disk path
        input_config (Dict): config for the FFMPEG, by default use h264_cuid
        output_source (str): url to streaming the procesed frames, can be a disk path
        output_config (Dict): config for the FFMPEG, by default use h264_nvenc

    Raises:
        Exception: check if can read frames from input source
        Exception: check if can do the inference over the frame
    """    
    
    custom_ffmpeg = get_config("config.json", "custom_ffmpeg") 

    stream = CamGear(
        source = input_source, 
        **input_config, 
        custom_ffmpeg = custom_ffmpeg
    ).start() 

    streamer = WriteGear(
        output = output_source, 
        **output_config, 
        custom_ffmpeg = custom_ffmpeg
    )  


    globals.init()

    for _ in range(config["max_iteration"]):
        image = stream.read()

        if image is None:
            msg = "failed to grab frame"
            logging.error(msg)
            print(msg)
            raise Exception(msg)

        try:
            ml_model.update_attention_region(globals.user_ref_point, image.shape)
            output_image = ml_model(image)
        except Exception as e:
            msg = f"can not make inference with ml_model, error: {e}"
            logging.error(msg)
            print(msg)
            print(traceback.format_exc())
            raise Exception(msg)
        
        try:
            streamer.write(output_image)
        except Exception as e:
            msg = f"can not write to the output_source: {output_source}, error: {e}"
            print(msg)
            logging.error(msg)
            print(traceback.format_exc())
            raise Exception(msg)


    stream.stop()
    streamer.close()

