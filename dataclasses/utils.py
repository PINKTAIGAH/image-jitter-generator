import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.misc import face

SOBEL_KERNAL = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
)

def arrayConcatinate(arrayLeft, arrayRight):
    return np.concatenate((arrayLeft, arrayRight), axis=1)

def verticalEdges(array):
    return convolve2d(array, SOBEL_KERNAL, "same")


if __name__ == "__main__":
    image = face(gray=True,)
    imageEdges = verticalEdges(image)
    imageConcat = arrayConcatinate(image, imageEdges)
    plt.imshow(imageConcat, cmap = "gray")
    plt.show()
