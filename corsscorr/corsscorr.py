from time import time
import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
from torch.nn.modules import padding
from ImageGenerator import ImageGenerator
from skimage.filters import gaussian

def corrImageScipy(img1, img2):
    corr = correlate2d(img1, img2, mode="full",)
    return corr.sum()

def corrImagePython(img1, img2):
    return np.sum((img1*img2)**2)/np.sqrt((img1**2).sum() * (img2**2).sum())

def normalise(array):
    if np.sum(array) == 0:
        raise Exception("Divided by zero. Attempted to normalise a zero array")
    return array/np.sum(array**2)

def createPSF(noiseSize, sigma):
    kernal = np.zeros((noiseSize, noiseSize))
    kernal[noiseSize//2, noiseSize//2] = 1
    return normalise(gaussian(kernal, sigma))

def test():
    noiseSize = 256
    paddingWidth = 15
    maxJitter = 7
    sigma = 15
    psf = createPSF(noiseSize, sigma)

    filter = ImageGenerator(psf, maxJitter, noiseSize, 4, paddingWidth)

    img1 = filter.generateGroundTruth()
    img2 = filter.generateGroundTruth()

    t1 = time()
    corrScipy = corrImageScipy(img2, img2)
    t2 = time()
    corrPython = corrImagePython(img1, img2)
    t3 = time()

    print(f"Using scipy: {t2-t1} s")
    print(f"Using numpy: {t3-t2} s")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img1, cmap="gray")
    ax2.imshow(img2, cmap="gray")
    plt.show()

if __name__ == "__main__":
    test()

