from torch.utils.data import Dataset
import scipy.ndimage as ndimg
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from kornia.geometry.transform import translate
from torchvision.transforms import Pad

class ImageGenerator(object):

    def __init__(self, psf, maxJitter, imageHeight, correlationLength, padding_width):
        
        if not torch.is_tensor(psf):
            psf = torch.from_numpy(psf)

        self.psf = psf
        self.ftPsf = torch.fft.fft2(self.psf)
        self.maxJitter = maxJitter
        self.imageHight = imageHeight
        self.correlationLength = correlationLength
        self.pad = Pad(padding_width)

    def generateGroundTruth(self):

        whiteNoise = torch.randn(*self.ftPsf.shape)
        groundTruth = torch.fft.ifft2(self.ftPsf * torch.fft.fft2(whiteNoise))  
        return self.pad(torch.real(groundTruth).type(torch.float32)).numpy()

