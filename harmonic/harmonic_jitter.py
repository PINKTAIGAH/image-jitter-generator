from kornia.geometry.transform import warp_perspective
from scipy.ndimage import shift
from ImageGenerator import ImageGenerator
import config
import utils
import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time

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

t1 = time()


IMAGE_SIZE = 256
CORRELATION_LENGTH = 5
MAX_JITTER =  5
shift_vector = np.empty((IMAGE_SIZE, IMAGE_SIZE))
wavelet_centers = np.arange(0, IMAGE_SIZE, CORRELATION_LENGTH*3)

for j in range(IMAGE_SIZE):
    x = np.arange(IMAGE_SIZE)
    y_final = np.zeros_like(x, dtype=np.float64)
    for i, val in enumerate(wavelet_centers):
        y = message(x)
        y_decay = decay(x, val, CORRELATION_LENGTH)
        y_carrier = carrier(x, IMAGE_SIZE)
        y_final += utils.adjustArray(y * y_decay * y_carrier)*MAX_JITTER
    shift_vector[j] = y_final
t2 = time()
print(f"time taken per image is {t2 - t1} s")

print(shift_vector)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.scatter(x, shift_vector[0], s=5, marker='x')
ax2.plot(x, shift_vector[0])
plt.show()
    

