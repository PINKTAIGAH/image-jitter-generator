from kornia.geometry.transform import warp_perspective
from ImageGenerator import ImageGenerator
import config
import utils
import torch
import numpy as np
import matplotlib.pyplot as plt


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

def carrier(x):
    f = generateRandomFrequency('l')
    return np.sin(f*np.pi*x/256)

def decay(x, x_0=0.0, std=1.0):
    return np.exp(-(x-x_0)**2/(2*std**2))

CORRELATION_LENGTH = 4
MAX_JITTER =  5
wavelet_centers = np.arange(0, 256, CORRELATION_LENGTH*3)
x = np.arange(256)
y_final = np.zeros_like(x, dtype=np.float64)

for i, val in enumerate(wavelet_centers):
    y = message(x)
    y_decay = decay(x, val, CORRELATION_LENGTH)
    y_carrier = carrier(x)
    y_final += utils.adjustArray(y * y_decay * y_carrier)*MAX_JITTER

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.scatter(x, y_final, s=5, marker='x')
ax2.plot(x, y_final)
plt.show()
    

