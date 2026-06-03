import numpy as np
import cv2
import  torch
import json
from PIL import Image
import tensorrt as trt
import matplotlib
logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")
import utils.common as common
from transformers import Sam3Processor
import time
from utils.image_preprocess import preprocess_image
from utils.simplify_tokenizer import SimpleCLIPBPETokenizer
from utils.detect_postprocess import process_sam3_results, draw_sam3_results

from transformers.models.sam3.modeling_sam3 import Sam3ImageSegmentationOutput


with open("config.json") as f:
    config = json.load(f)

config_metadata = config["metadata"]
config = config["infer_trt_webcam"]



def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image


def click_and_crop(event, x, y, flags, param):
      
    global user_ref_point

    if event == cv2.EVENT_LBUTTONDOWN:
        windowWidth = cv2.getWindowImageRect(VISOR_NAME)[2]
        windowHeight = cv2.getWindowImageRect(VISOR_NAME)[3]
        user_ref_point = [(x / windowWidth, y / windowHeight)]

    if event == cv2.EVENT_LBUTTONUP:
        windowWidth = cv2.getWindowImageRect(VISOR_NAME)[2]
        windowHeight = cv2.getWindowImageRect(VISOR_NAME)[3]
        user_ref_point += [(x / windowWidth, y / windowHeight)]

    


def main_loop():
    
    global prompt 
    cummulative_time = 0
    n_iteration = 0
    box1_xyxy = [0, 0, 0, 0]  # Dial box
    input_boxes = [[box1_xyxy]]
    input_boxes_labels = [[1]] 

    for _ in range(config["max_iteration"]):
        n_iteration += 1
        start = time.time()
        ret, image = cam.read()

        if not ret:
            print("failed to grab frame")
            break


        if len(user_ref_point) == 2:
            image_shape = image.shape
            a = (
                int(user_ref_point[0][0] * image_shape[1]), 
                int(user_ref_point[0][1] * image_shape[0])
            )
            b = (
                int(user_ref_point[1][0] * image_shape[1]), 
                int(user_ref_point[1][1] * image_shape[0])
            )
            box1_xyxy = [*a, *b]  # Dial box
            input_boxes = [[box1_xyxy]]
            input_boxes_labels = [[1]] 
            radius = 5          # In pixels
            color = (0, 0, 255) # BGR format
            thickness = 2     # Thickness in pixels (-1 fills the circle)
            image = cv2.rectangle(image, a, b, color, thickness)
        
         
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if (n_iteration + 1) % 2 == 0:
            tokenizer_output = processor(
                images=image, 
                text=prompt, 
                input_boxes=input_boxes,
                input_boxes_labels=input_boxes_labels,
                return_tensors="pt"
            )

    
            np.copyto(inputs[0].host, tokenizer_output["pixel_values"].ravel())
            np.copyto(inputs[1].host, tokenizer_output["input_ids"].ravel())
            np.copyto(inputs[2].host, tokenizer_output["attention_mask"].ravel())
            np.copyto(inputs[3].host, tokenizer_output["input_boxes"].ravel())
            np.copyto(inputs[4].host, tokenizer_output["input_boxes_labels"].ravel())


            output_model = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

            output_model = [
                out.reshape(*outputs_shapes)
                for out, outputs_shapes in zip(output_model, config["outputs_shapes"])
            ]

    
        sam_output = Sam3ImageSegmentationOutput()

        sam_output.pred_masks =              torch.from_numpy(output_model[0])
        sam_output.pred_boxes =              torch.from_numpy(output_model[1])
        sam_output.pred_logits =             torch.from_numpy(output_model[2])
        sam_output.presence_logits =         torch.from_numpy(output_model[3])
        sam_output.semantic_seg =            torch.from_numpy(output_model[4])
        sam_output.decoder_reference_boxes = torch.from_numpy(output_model[5])




        results = processor.post_process_instance_segmentation(
            sam_output,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=tokenizer_output.get("original_sizes").tolist()
        )[0]



        output_image = overlay_masks(image, results["masks"])

        output_image = np.asarray(output_image)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    


        cv2.imshow(VISOR_NAME, output_image)
        cv2.waitKey(1)  # Wait for a key press to close the window

        end = time.time()

        cummulative_time += end-start

        if n_iteration % 100 == 0:

            print(f"Process: time: {cummulative_time /  100}")
            n_iteration = 0 
            cummulative_time = 0




if __name__ == "__main__":

    cam = cv2.VideoCapture(0)
    engine_file_path = config["engine_file_path"]
    VISOR_NAME = config["VISOR_NAME"]

    global user_ref_point 
    user_ref_point = []
    
    cv2.namedWindow(VISOR_NAME)
    cv2.setMouseCallback(VISOR_NAME, click_and_crop)


    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    

    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    vocab_file = config_metadata["vocab_file"] 
    merges_file = config_metadata["merges_file"]

    global prompt 
    prompt = config["prompt"]
    
    processor = Sam3Processor.from_pretrained("facebook/sam3")


    main_loop()

    cv2.destroyAllWindows()  # Close the window
    cam.release()

