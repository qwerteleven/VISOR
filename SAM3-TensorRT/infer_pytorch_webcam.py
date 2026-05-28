from transformers.models.sam3 import Sam3Processor, Sam3Model
import torch
import cv2
from PIL import Image
import numpy as np
from utils.pytorch_visualize import visualize_sam3_results
import time
import json


with open("config.json") as f:
    config = json.load(f)

config = config["infer_pytorch_webcam"]


device = config["device"] 
prompt = config["prompt"] 
benchmark = config["benchmark"] 


model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")
cam = cv2.VideoCapture(0)


while True:
    start = time.time()
    ret, image = cam.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    if not ret:
        print("failed to grab frame")
        break


    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=config["postprocess_threshold"],
        mask_threshold=config["postprocess_mask_threshold"],
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    image = visualize_sam3_results(
        image,
        results,
        out_path = 'results_.jpg',
        mask_alpha = config["mask_alpha"],
        mask_threshold = config["mask_threshold"],
        box_width = config["box_width"],
        benchmark=benchmark
    )

    image = np.array(image)
    # Convert RGB to BGR
    image = image[:, :, ::-1].copy()

    cv2.imshow("test", image)
    cv2.waitKey(1) 
    end = time.time()

    print(f"Process: time: {end-start}")

cv2.destroyAllWindows()  # Close the window

cam.release()

