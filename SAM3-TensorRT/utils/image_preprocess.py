import numpy as np
import cv2

# Pre-compute constants once (outside the function)
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
INV_STD = 1.0 / STD  # multiply is faster than divide

def preprocess_image(image):
    image = cv2.resize(image, (1008, 1008), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image *= (1.0 / 255.0)               # in-place, avoids temp array
    image -= MEAN
    image *= INV_STD                      # in-place multiply vs divide
    return image[np.newaxis].transpose(0, 3, 1, 2)  # NHWC → NCHW in one shot

