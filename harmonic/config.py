from skimage.filters import gaussian
import numpy as np
import torch

def normalise(x):
    if np.sum(x) == 0:
        raise Exception("Divided by zero. Attempted to normalise a zero tensor")

    return x/np.sum(x**2)

IMAGE_SIZE = 256 
CORRELATION_LENGTH = 2
SIGMA = 10 
CHANNELS_IMG = 1 
MAX_JITTER = 0.4
PADDING_WIDTH = 15

psf_hight = IMAGE_SIZE-2*PADDING_WIDTH
kernal = np.zeros((psf_hight, psf_hight))
kernal[psf_hight//2, psf_hight//2] = 1
PSF = torch.from_numpy(normalise(gaussian(kernal, SIGMA)))
