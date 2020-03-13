import numpy as np
from sklearn.model_selection import ParameterSampler, ParameterGrid


def clip(img):
    return np.clip(img, a_min=0.0, a_max=255.0).astype(np.int32)


def mse(original_img: np.ndarray, cleaned_img: np.ndarray):
    diff = original_img - cleaned_img
    diff = diff ** 2
    return 0.5 * np.mean(diff)


