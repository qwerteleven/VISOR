import numpy as np
import cv2

# Pre-compute constants once (outside the function)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INV_STD = 1.0 / STD  # multiply is faster than divide

def preprocess_image(image):
    image = cv2.resize(image, (1008, 1008), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image *= (1.0 / 255.0)               # in-place, avoids temp array
    image -= MEAN
    image *= INV_STD                      # in-place multiply vs divide
    return image[np.newaxis].transpose(0, 3, 1, 2)  # NHWC → NCHW in one shot

