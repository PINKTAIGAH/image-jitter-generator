from ctypes import alignment
from torch.utils.data import Dataset
import scipy.ndimage as ndimg
import torch
import config
import utils
import wavelets
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from kornia.geometry.transform import translate
from torchvision.transforms import Pad

class ImageGenerator(object):

    def __init__(self, psf, maxJitter, imageHeight, correlationLength, padding_width):
        
        self.psf = psf
        self.ftPsf = torch.fft.fft2(self.psf)
        self.maxJitter = maxJitter
        self.imageHight = imageHeight
        self.correlationLength = correlationLength
        self.pad = Pad(padding_width)

    def generateGroundTruth(self):

        whiteNoise = torch.randn(*self.ftPsf.shape)
        groundTruth = torch.fft.ifft2(self.ftPsf * torch.fft.fft2(whiteNoise))  
        return self.pad(torch.real(groundTruth).type(torch.float32))

    def generateShifts(self):
        shiftsX = torch.from_numpy(wavelets.generateShiftMatrix(self.imageHight,
                                    self.correlationLength,
                                    self.maxJitter))
        shiftsX = torch.unsqueeze(shiftsX, 2)
        shiftsY = torch.zeros_like(shiftsX)
        return torch.cat([shiftsX, shiftsY], dim=2)
        

    def shiftImage(self, input, shifts, isBatch=True):
        if not isBatch:
            input = torch.unsqueeze(input, 0)
            shifts = torch.unsqueeze(shifts, 0)

        if len(input.shape) != 4:
            raise Exception("Input image must be of dimention 4: (B, C, H, W)")
        if len(shifts.shape) !=4:
            raise Exception("Shifts must be of the shape (B, H, W, 2)")

        B, _, H, W = input.shape
        output = torch.zeros_like(input)

        for i in range(B):
            singleImage = torch.unsqueeze(torch.clone(input[i]), 0)
            for j in range(H):
                singleVectorRow = torch.cumsum(shifts[i, j], dim=0).type(torch.float32)
                for k in range(W):
                    if k > 0:
                        shift = torch.unsqueeze(singleVectorRow[k], 0).type(torch.float32)
                        output[:, :, j, :k] = translate(torch.unsqueeze(singleImage[:, :, j, :k], 0),
                                        shift,
                                        mode="bilinear",
                                        align_corners=False) 

        """
        for i in range(B):
            singleImage = torch.unsqueeze(torch.clone(input[i]),0)
            singleShift = torch.clone(shifts[i])
            for j in range(H):
                output[i, :, j, :] = translate(singleImage[:, :, j, :],
                                               torch.unsqueeze(singleShift[j], 0),
                                               padding_mode="reflection",
                                               align_corners=False)
        """
        return output

def test():

    filter = ImageGenerator(config.PSF, config.MAX_JITTER, config.IMAGE_SIZE,
                            config.CORRELATION_LENGTH, config.PADDING_WIDTH)

    groundTruth = filter.generateGroundTruth()  # C*H*W 
    shiftMatrix = filter.generateShifts()       # H*W*2

    groundTruth = torch.unsqueeze(groundTruth, 0)

    shifted = filter.shiftImage(groundTruth, shiftMatrix, isBatch=False)

    unshifted = filter.shiftImage(shifted[0], -shiftMatrix, isBatch=False)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(groundTruth[0], cmap="gray")
    ax2.imshow(shifted[0,0], cmap="gray")
    ax3.imshow(unshifted[0,0], cmap="gray")
    ax4.imshow(groundTruth[0]-unshifted[0,0], cmap="gray")
    plt.show()
if __name__ == "__main__":
    test()
