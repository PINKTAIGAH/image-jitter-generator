from kornia.geometry.transform import warp_perspective
from scipy.ndimage import shift
from ImageGenerator import ImageGenerator
import config
import utils
import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.ndimage import shift

def generateRandomFrequency(mode="m"):
    match mode:
        case "l":
            mean, std = 30, 15
        case "h":
            mean, std = 400, 50
        case "m":
            mean, std = 200, 150
        case _:
            raise Exception(f"Input was {mode}, when accepted inputs are 'm', 'l' or 'h'")

    return np.random.normal(mean, std)

def generateRandomAmplitude(maxAmplitude):
    return np.random.rand()*maxAmplitude

def generateRandoPhase():
    return np.random.rand()*2*np.pi

def generateMainJitter():
    return generateRandomAmplitude(3)*np.sin(2*np.pi*generateRandomFrequency()*generateRandoPhase())

def generateLowJitter():
    return generateRandomAmplitude(3)*np.sin(2*np.pi*generateRandomFrequency('l')*generateRandoPhase())

def generateHighJitter():
    return generateRandomAmplitude(3)*np.sin(2*np.pi*generateRandomFrequency('h')*generateRandoPhase())
                
def message(x):
    f_m, A_m, P_m = generateRandomFrequency("m"), generateRandomAmplitude(5), generateRandoPhase()
    f_l, A_l, P_l = generateRandomFrequency("l"), generateRandomAmplitude(5), generateRandoPhase()
    f_h, A_h, P_h = generateRandomFrequency("h"), generateRandomAmplitude(5), generateRandoPhase()
    return A_m*np.sin(2*np.pi*f_m*x+P_m) + A_l*np.sin(2*np.pi*f_l*x+P_l) + A_h*np.sin(2*np.pi*f_h*x+P_h)

def carrier(x, image_size):
    p = generateRandoPhase()
    f = generateRandomFrequency('l')
    return np.sin(f*np.pi*x/image_size + p)

def decay(x, x_0=0.0, std=1.0):
    return np.exp(-(x-x_0)**2/(2*std**2))

def generateShiftVector(imageSize, correlationLength, maxJitter):

    shift_vector = np.empty((imageSize, imageSize))
    wavelet_centers = np.arange(0, imageSize, correlationLength*3)

    for j in range(imageSize):
        x = np.arange(imageSize)
        y_final = np.zeros_like(x, dtype=np.float64)
        for i, val in enumerate(wavelet_centers):
            y = message(x)
            y_decay = decay(x, val, correlationLength)
            y_carrier = carrier(x, imageSize)
            y_final += utils.adjustArray(y * y_decay * y_carrier)*maxJitter
        shift_vector[j] = y_final
    return shift_vector

def shiftImage(input, shiftMatrix):

    output = np.copy(input)
    for i in range(config.IMAGE_SIZE):
        for j in range(config.IMAGE_SIZE):
            shifts = np.cumsum(shiftMatrix[i])
            output[i, :j] = shift(input[i, :j], shifts[j], output=None, 
                                    order=3, mode="constant", cval=0, prefilter=True)

    return output

filter = ImageGenerator(config.PSF, config.MAX_JITTER, config.IMAGE_SIZE,
                        config.CORRELATION_LENGTH, config.PADDING_WIDTH)
groundTruth = filter.generateGroundTruth().numpy()
print(f"ground truth generated")

shiftMatrix = generateShiftVector(config.IMAGE_SIZE, config.CORRELATION_LENGTH,
                                   config.MAX_JITTER)
print(f"shift matrix generated")
shiftMatrix[:, 0] = 0
t1 = time()
shifted = shiftImage(groundTruth, shiftMatrix)
print(f"shift image generated")
t2 = time()
unshifted = shiftImage(shifted, -shiftMatrix)
print(f"unshift image generated")

# padding_mask = shifted==0.0

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

print(f"Time taken to shift image of {config.IMAGE_SIZE}p is {t2-t1} s")
ax1.imshow(groundTruth, cmap="gray")
ax2.imshow(shifted, cmap="gray")
ax3.imshow(unshifted, cmap="gray")
ax4.imshow(groundTruth-unshifted, cmap="gray")
plt.show()

    

