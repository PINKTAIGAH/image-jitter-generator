from skimage.filters import gaussian
import numpy as np
import torch
import utils

IMAGE_SIZE = 256
SIGMA = 7 
MAX_JITTER = 4

kernal = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
kernal[IMAGE_SIZE//2, IMAGE_SIZE//2] = 1
PSF = torch.from_numpy(utils.normalise(gaussian(kernal, SIGMA)))
