from skimage.filters import gaussian
import numpy as np
import torch

def normalise(x):
    if np.sum(x) == 0:
        raise Exception("Divided by zero. Attempted to normalise a zero tensor")

    return x/np.sum(x**2)

IMAGE_SIZE = 256
SIGMA = 10 
CHANNELS_IMG = 1 
MAX_JITTER = 1.5

kernal = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
kernal[IMAGE_SIZE//2, IMAGE_SIZE//2] = 1
PSF = torch.from_numpy(normalise(gaussian(kernal, SIGMA)))
