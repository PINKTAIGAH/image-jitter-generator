import numpy as np
import matplotlib.pyplot as plt
from jitterFilter import JitterFilter
from imageGenerator import ImageGenerator

N = 64 
testImage = ImageGenerator(N).generateLines(6) 

testClass = JitterFilter(testImage, N, 3)
jitterImage = testClass.linearJitter()
testClass.printJitterVector()

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(testImage, cmap="gnuplot")
ax2.imshow(jitterImage, cmap="gnuplot")

plt.show()
