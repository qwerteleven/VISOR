import numpy as np


def sigmoid(z: np.array) -> np.array:
    """
    
        sigmoid of a numpy array, probs between (0 - 1)

    Args:
        z (np.array): logits matrix 

    Returns:
        np.array: output probs
    """   

    return 1/(1 + np.exp(-z))