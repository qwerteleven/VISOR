from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import cv2
import time

import numpy as np
import matplotlib

global user_ref_point
user_ref_point = []


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

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")


VISOR_NAME = "test"
cam = cv2.VideoCapture(0)

cv2.namedWindow(VISOR_NAME)

cv2.setMouseCallback(VISOR_NAME, click_and_crop)


box1_xyxy = [59, 144, 76, 163]  # Dial box
box2_xyxy = [87, 148, 104, 159]  # Button box
input_boxes = None
input_boxes_labels = None



cummulative_time = 0
n_iteration = 0




for _ in range(20000):
    ret, image = cam.read()
    start = time.time()
    n_iteration += 1
    
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
    # Segment using text prompt
    inputs = processor(
        images=image,
        text="object inside box", 
        input_boxes=input_boxes,
        input_boxes_labels=input_boxes_labels,
        return_tensors="pt"
    ).to(device)



    with torch.no_grad():
        outputs = model(**inputs)


    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
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


cv2.destroyAllWindows()  # Close the window
cam.release()


