import numpy as np 
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.datasets import face

SOBEL_KERNAL = torch.tensor(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32
)

def tensorConcatinate(tensorLeft, tensorRight):
    tensorRight = tensorRight.view(-1, tensorRight.shape[-1])
    tensorLeft = tensorLeft.view(-1, tensorLeft.shape[-1])
    return torch.cat((tensorLeft, tensorRight), dim=1) 

def verticalEdges(inputTensor):
    print(inputTensor.shape)
    print(SOBEL_KERNAL.shape)
    return F.conv2d(inputTensor.reshape(1, 1, *inputTensor.shape),
                    SOBEL_KERNAL.reshape(1, 1, *SOBEL_KERNAL.shape), padding="same")


if __name__ == "__main__":
    image = torch.from_numpy(face(gray=True,)).type(torch.float32)
    imageEdges = verticalEdges(image)
    imageConcat = tensorConcatinate(image, imageEdges)
    plt.imshow(imageConcat.numpy(), cmap = "gray")
    plt.show()
