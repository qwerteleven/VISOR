import torch
from pathlib import Path
from transformers.models.sam3 import Sam3Processor, Sam3Model
from PIL import Image
import requests
import json
import cv2
import numpy as np


with open("config.json") as f:
    config = json.load(f)

config = config["onnx_export"]



device = config["device"]

# 1. Load model & processor
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")
model.eval()

prompt = config["prompt"]

# 2. Build a sample batch (same as your example)
image_url = config["image_url"]
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")


radius = 5          # In pixels
color = (0, 0, 255) # BGR format
thickness = 2     # Thickness in pixels (-1 fills the circle)


## inputs for create a inference with ONNX that activates a significative input
image = np.asarray(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image = cv2.rectangle(image, [225, 86], [540, 414], color, thickness)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
input_boxes = [[[225, 86, 540, 414]]]
input_boxes_labels = [[1]] 


inputs = processor(images=image, text=prompt, 
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt").to(device)

pixel_values = inputs["pixel_values"]
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
input_boxes = inputs["input_boxes"]
input_boxes_labels = inputs["input_boxes_labels"]


print("input_ids", input_ids.shape, input_ids.dtype)
print(input_ids)
print()
print("attention_mask", attention_mask.shape, attention_mask.dtype)
print(attention_mask)

# 3. Wrap Sam3Model so the ONNX graph has clean inputs/outputs
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

torch.onnx.export(
    model,
    (pixel_values, input_ids, attention_mask, input_boxes, input_boxes_labels),
    onnx_path,
    input_names=["pixel_values", "input_ids", "attention_mask", "input_boxes", "input_boxes_labels"],
    output_names=["pred_masks", "pred_boxes", "pred_logits", 'presence_logits', 'semantic_seg', 'decoder_reference_boxes'],
    dynamo=config["dynamo"],
    opset_version=config["opset_version"],
)
print(f"Exported to {onnx_path}")
