import numpy as np
import matplotlib.pyplot as plt
from jitterFilter import JitterFilter
from imageGenerator import ImageGenerator

cmap = "gnuplot"
N = 600 
testGeneratorClass = ImageGenerator(N)
testImage = testGeneratorClass.generateGenericNoise() 
N = testImage.shape[0]

testJitterClass = JitterFilter(testImage, N, 3)
jitterImage = testJitterClass.linearJitter()
# testJitterClass.printJitterVector()

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(testImage, cmap=cmap)
ax1.set_title("test image")
ax2.imshow(jitterImage, cmap=cmap)
ax2.set_title("jittered image")

concatImage = testGeneratorClass.concatenateArray(jitterImage,
                                                  testImage)
testGeneratorClass.saveArrayToImage(concatImage)


plt.show()
