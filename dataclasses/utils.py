import numpy as np 
from scipy.signal import convolve2d

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
