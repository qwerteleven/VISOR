import torch
from pathlib import Path
from transformers.models.sam3 import Sam3Processor, Sam3Model
from PIL import Image
import requests
import cv2
import numpy as np
import sys
import os
import logging
import traceback

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from utils.io import get_config, set_logger
config = get_config("config.json", "onnx_export") 
set_logger("../logs", os.path.basename(sys.argv[0]))


device = config["device"]

try:
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
except Exception as e:
    msg = "can not load model"
    logging.error(msg)
    print(msg)
    print(traceback.format_exc())

model.eval()


try:
    image = Image.open(requests.get(config["image_url"], stream=True).raw).convert("RGB")
except FileNotFoundError:
    msg = "can not load image for traking ONNX graph"
    logging.error(msg)
    print(msg)


## inputs for create a inference with ONNX that activates a significative input
image = np.asarray(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image = cv2.rectangle(image, [225, 86], [540, 414],  (0, 0, 255), 2 )
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
input_boxes = [[[225, 86, 540, 414]]]
input_boxes_labels = [[1]] 


inputs = processor(
    images=image,
    text=config["prompt"], 
    input_boxes=input_boxes,
    input_boxes_labels=input_boxes_labels,
    return_tensors="pt"
).to(device)


class Sam3ONNXWrapper(torch.nn.Module):
    def __init__(self, sam3):
        super().__init__()
        self.sam3 = sam3

    def forward(self, pixel_values, input_ids, attention_mask, input_boxes, input_boxes_labels):
        outputs = self.sam3(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels
            )
        
        return outputs.pred_masks, outputs.pred_boxes, outputs.pred_logits, outputs.presence_logits, outputs.semantic_seg, outputs.decoder_reference_boxes

wrapper = Sam3ONNXWrapper(model).to(device).eval()

# 5. Export to ONNX
output_dir = Path(f"onnx_weights")
output_dir.mkdir(exist_ok=True)
onnx_path = str(output_dir /  config["onnx_path"])

pixel_values = inputs["pixel_values"]
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
input_boxes = inputs["input_boxes"]
input_boxes_labels = inputs["input_boxes_labels"]

try: 

    msg = "Start ONNX export..."
    logging.info(msg)
    print(msg)

    torch.onnx.export(
        model,
        (pixel_values, input_ids, attention_mask, input_boxes, input_boxes_labels),
        onnx_path,
        input_names=["pixel_values", "input_ids", "attention_mask", "input_boxes", "input_boxes_labels"],
        output_names=["pred_masks", "pred_boxes", "pred_logits", 'presence_logits', 'semantic_seg', 'decoder_reference_boxes'],
        dynamo=config["dynamo"],
        opset_version=config["opset_version"],
    )
    msg = f"Exported to {onnx_path}"
    logging.info(msg)
    print(msg)
    
except Exception as e:
    msg = "can not export ONNX model"
    logging.error(msg)
    print(msg)
    print(traceback.format_exc())
    

