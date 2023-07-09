import numpy as np
import matplotlib.pyplot as plt


class ImageGenerator(object):
    def __init__(self, N=64):
        self.N = N

    def generateLines(self, nLines=3):
        self.array = np.zeros((self.N, self.N))
        for _ in range(nLines):
            randomInt = np.random.random_integers(0, self.N-1)
            self.array[:, randomInt-1] = 1
            self.array[:, randomInt] = 1
            self.array[:, randomInt+1] = 1
        return self.array


if __name__ == "__main__":
    testClass = ImageGenerator()
    plt.imshow(testClass.generateLines())
    plt.show()
