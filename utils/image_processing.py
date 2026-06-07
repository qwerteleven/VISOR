
import matplotlib
from PIL import Image
import numpy as np
import torch
from typing import List, Dict
import cv2

def overlay_masks(image: np.array, masks: torch.Tensor) -> np.array:
    """
    
        blend mask with the mask over a RGBA image

    Args:
        image (np.array): frame from the streaming
        masks (torch.Tensor): mask segmentations

    Returns:
        np.array: image overlaped with masks
    """    
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


def draw_user_rectangle(image: np.array, config: Dict, user_ref_point: List) -> np.array:
    """
    
        Draw  the rectagle of interest of the user on overlay streaming

    Args:
        image (np.array): frame of streaming
        user_ref_point (List): point upper left, down right of box of interest

    Returns:
        np.array: blended image with box of interest
    """    

    image_shape = image.shape

    a = (
        int(user_ref_point[0][0] * image_shape[1]), 
        int(user_ref_point[0][1] * image_shape[0])
    )

    b = (
        int(user_ref_point[1][0] * image_shape[1]), 
        int(user_ref_point[1][1] * image_shape[0])
    )

    image = cv2.rectangle(image, a, b, config["color"], config["thickness"])

    return image
