from ImageGenerator import ImageGenerator
import config
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

displacementConstX = 1.27
displacementConstY = 1
shiftList = []

filter = ImageGenerator(config.PSF, config.MAX_JITTER, config.IMAGE_SIZE)
groundTruth, _ = filter.generateGroundTruth()

for _ in range(1000):
    altitudeJitter = generateMainJitter() + sum([generateLowJitter() for _ in range(5)]) +\
        sum([generateHighJitter() for _ in range(5)])
                
    shiftX = altitudeJitter*displacementConstX
    shiftList.append(shiftX)

plt.plot(shiftList)
plt.show()
    

