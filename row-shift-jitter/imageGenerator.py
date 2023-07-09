import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import gaussian
from scipy.signal import convolve2d

class ImageGenerator(object):
    def __init__(self, N=64):
        self.N = N

    def generateLines(self, nLines=3):
        self.array = np.zeros((self.N, self.N))
        for _ in range(nLines):
            randomInt = np.random.randint(0, self.N-1)
            self.array[:, randomInt-1] = 1
            self.array[:, randomInt] = 1
            self.array[:, randomInt+1] = 1
        return self.array

    def generateGenericNoise(self, nConvolutions=3, kernalSize=4):
        self.kernalSize = kernalSize
        self.createKernalPSF()

        self.array = np.random.random((self.N, self.N))

        for _ in range(nConvolutions):
            self.array = convolve2d(self.array, self.kernal, boundary="wrap")
        return self.array 

    def createKernalPSF(self):
        self.kernal = np.zeros((self.kernalSize, self.kernalSize))
        self.kernal[self.kernalSize//2, self.kernalSize//2] = 1 
        self.kernal = gaussian(self.kernal)

    def generateRandomMnist(self, root="/media/giorgio/HDD/GAN/dcgan/mnist_dataset/"):
        nImages = len(os.listdir(root))
        randomImage = os.listdir(root)[np.random.randint(0, nImages-1)]
        image = Image.open(root + randomImage)
        imageArray = np.asarray((image))
        return imageArray
        

if __name__ == "__main__":
    cmap = "gray"
    testClass = ImageGenerator()
    plt.imshow(testClass.generateRandomMnist(), cmap=cmap)
    plt.show()
