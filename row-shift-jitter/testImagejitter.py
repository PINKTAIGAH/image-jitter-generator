import numpy as np
import matplotlib.pyplot as plt
from jitterFilter import JitterFilter
from imageGenerator import ImageGenerator

cmap = "gnuplot"
N = 600 
testImage = ImageGenerator(N).generateGenericNoise() 
N = testImage.shape[0]

testClass = JitterFilter(testImage, N, 3)
jitterImage = testClass.linearJitter()
# testClass.printJitterVector()

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(testImage, cmap=cmap)
ax1.set_title("test image")
ax2.imshow(jitterImage, cmap=cmap)
ax2.set_title("jittered image")

plt.show()
