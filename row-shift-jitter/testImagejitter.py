import numpy as np
import matplotlib.pyplot as plt
from jitterFilter import JitterFilter

N = 10
testImage = np.arange(0, N**2).reshape(N,N)

testClass = JitterFilter(testImage, N, 3)
jitterImage = testClass.linearJitter()
testClass.printJitterVector()

fig, (ax1, ax2) = plt.subplots(2,1)

ax1.imshow(testImage, cmap="gnuplot")
ax2.imshow(jitterImage, cmap="gnuplot")

plt.show()
