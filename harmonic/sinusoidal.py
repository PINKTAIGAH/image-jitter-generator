import torch
import config
import numpy as np
import matplotlib.pyplot as plt
from ImageGenerator import ImageGenerator
from torchvision.transforms import Pad

filter = ImageGenerator(config.PSF, config.MAX_JITTER, config.IMAGE_SIZE)
transform = Pad((20, 20))

x,_ = filter.generateGroundTruth()
x = transform(x)
y = torch.clone(x)
A = 5
F = 6
PHI = 2

for i in range(config.IMAGE_SIZE):
    # shift = A * np.sin(20*np.pi*i/276) * np.sin(5*np.pi*i/276)
    shift = A * np.sin(F*2*np.pi*i/276)
    y[i, :] = torch.roll(x[i, :], shifts=int(shift))

for i in range(config.IMAGE_SIZE):
    # shift = A * np.sin(20*np.pi*i/276) * np.sin(5*np.pi*i/276)
    shift = A * np.sin(15*2*np.pi*i/276)
    y[:, i] = torch.roll(x[:, i], shifts=int(shift))

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(x)
ax2.imshow(y)
plt.show()
